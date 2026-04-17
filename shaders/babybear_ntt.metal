// BabyBear field + forward NTT (DIT: bit-reverse + butterfly stages).
// All kernels work on ROW-MAJOR matrices: data[row * width + col].
// A single dispatch handles all columns — no CPU transpose needed.

#include <metal_stdlib>
using namespace metal;

constant uint BB_P = 0x78000001u;

// P^{-1} mod 2^32 — the Montgomery constant for BabyBear.
constant uint BB_MONTY_MU = 0x88000001u;

struct Bb {
    uint v;
};

// All values are in Montgomery form: v represents (v_actual * 2^32) mod P.
// Add and sub are unchanged; Montgomery form is additive-compatible.

Bb bb_add(Bb a, Bb b) {
    uint sum = a.v + b.v;
    return Bb{sum >= BB_P ? sum - BB_P : sum};
}

Bb bb_sub(Bb a, Bb b) {
    if (a.v >= b.v) return Bb{a.v - b.v};
    return Bb{a.v + BB_P - b.v};
}

// Montgomery multiplication: monty_reduce(a * b) = a*b * R^{-1} mod P.
// Inputs and output are all in Montgomery form.
Bb bb_mul(Bb a, Bb b) {
    ulong x = ulong(a.v) * ulong(b.v);
    uint t = uint(x) * BB_MONTY_MU;
    ulong u = ulong(t) * ulong(BB_P);
    ulong x_sub_u = x - u;
    uint hi = uint(x_sub_u >> 32);
    return Bb{u > x ? hi + BB_P : hi};
}

// ── Bit-reverse permutation (batched, row-major) ──────────────────────
// 2D grid: gid.x = column, gid.y = row.
// Column as x gives coalesced global reads (adjacent threads → adjacent cols).
kernel void bb_ntt_bitrev(
    device Bb* data                [[buffer(0)]],
    constant uint& height          [[buffer(1)]],
    constant uint& width           [[buffer(2)]],
    constant uint& log_height      [[buffer(3)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= height || col >= width) return;

    uint rev = 0;
    uint val = row;
    for (uint i = 0; i < log_height; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    if (row < rev) {
        uint idx_a = row * width + col;
        uint idx_b = rev * width + col;
        Bb tmp = data[idx_a];
        data[idx_a] = data[idx_b];
        data[idx_b] = tmp;
    }
}

// ── Phase B: fused butterfly stages in threadgroup shared memory ──────
// One thread per ROW.  Columns are processed in tiles of
//   tile_w = min(8192 / tg_size, width)
// so that log_block stays at its maximum regardless of matrix width.
// Within each tile, the twiddle factor depends only on the row position
// and is loaded once and reused across all columns in the tile — the GPU
// analogue of Plonky3's SIMD packing.
//
// block_size = threads_per_threadgroup (= 1 << log_block).
kernel void bb_ntt_shared_mem(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& log_block       [[buffer(4)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup Bb sdata[8192];

    uint base_row = tgid * tg_size;
    uint global_row = base_row + tid;
    uint gm_row = global_row * width;
    uint tile_w = 8192 / tg_size;

    for (uint col_base = 0; col_base < width; col_base += tile_w) {
        uint cw = min(tile_w, width - col_base);

        // Load tile into shared memory.
        uint sm_base = tid * tile_w;
        for (uint c = 0; c < cw; c++) {
            sdata[sm_base + c] = data[gm_row + col_base + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stage = 0; stage < log_block; stage++) {
            uint stride = 1u << stage;

            if ((tid & stride) == 0) {
                uint partner = tid | stride;
                uint local_idx = tid & (stride - 1);
                uint tw_idx = local_idx * (height >> (stage + 1));

                uint a_base = tid * tile_w;
                uint b_base = partner * tile_w;

                if (tw_idx == 0) {
                    for (uint c = 0; c < cw; c++) {
                        Bb a = sdata[a_base + c];
                        Bb b = sdata[b_base + c];
                        sdata[a_base + c] = bb_add(a, b);
                        sdata[b_base + c] = bb_sub(a, b);
                    }
                } else {
                    Bb w = twiddles[tw_idx];
                    for (uint c = 0; c < cw; c++) {
                        Bb a = sdata[a_base + c];
                        Bb b = sdata[b_base + c];
                        Bb wb = bb_mul(w, b);
                        sdata[a_base + c] = bb_add(a, wb);
                        sdata[b_base + c] = bb_sub(a, wb);
                    }
                }
            }

            if (stride >= 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Store tile back to global memory.
        for (uint c = 0; c < cw; c++) {
            data[gm_row + col_base + c] = sdata[sm_base + c];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Phase A: single-stage global-memory butterfly (batched) ───────────
// 2D grid: gid.x = column, gid.y = butterfly index.
// Column as x gives coalesced access for adjacent threads.
kernel void bb_ntt_butterfly(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint butterfly_id = gid.y;
    uint num_butterflies = height >> 1;
    if (butterfly_id >= num_butterflies || col >= width) return;

    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint block_idx = butterfly_id / half_block;
    uint local_idx = butterfly_id % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;
    uint twiddle_idx = local_idx * (height / block_size);

    uint idx_a = i * width + col;
    uint idx_b = j * width + col;
    Bb a = data[idx_a];
    Bb b = data[idx_b];
    if (twiddle_idx == 0) {
        data[idx_a] = bb_add(a, b);
        data[idx_b] = bb_sub(a, b);
    } else {
        Bb w = twiddles[twiddle_idx];
        Bb wb = bb_mul(w, b);
        data[idx_a] = bb_add(a, wb);
        data[idx_b] = bb_sub(a, wb);
    }
}

// ── Phase A: radix-4 fused two-stage global butterfly (batched) ──────
// Processes stages `stage` and `stage + 1` in a single dispatch.
// Each thread handles one radix-4 unit: reads 4 elements, performs two
// layers of butterflies, writes 4 elements — halving the number of
// dispatches and global memory round-trips compared to radix-2.
// 2D grid: gid.x = column, gid.y = radix-4 unit index.
kernel void bb_ntt_butterfly_r4(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 2;
    if (unit_id >= num_units || col >= width) return;

    uint half_s = 1u << stage;
    uint block_r4 = half_s << 2;
    uint block_idx = unit_id / half_s;
    uint j = unit_id % half_s;

    uint base = block_idx * block_r4;
    uint row0 = base + j;
    uint row1 = row0 + half_s;
    uint row2 = row0 + 2 * half_s;
    uint row3 = row0 + 3 * half_s;

    uint idx0 = row0 * width + col;
    uint idx1 = row1 * width + col;
    uint idx2 = row2 * width + col;
    uint idx3 = row3 * width + col;

    Bb a = data[idx0];
    Bb b = data[idx1];
    Bb c = data[idx2];
    Bb d = data[idx3];

    // Stage s: butterfly pairs (a,b) and (c,d) with the same twiddle.
    uint tw1_idx = j * (height >> (stage + 1));
    Bb wb, wd;
    if (tw1_idx == 0) {
        wb = b; wd = d;
    } else {
        Bb w1 = twiddles[tw1_idx];
        wb = bb_mul(w1, b);
        wd = bb_mul(w1, d);
    }
    Bb A = bb_add(a, wb);
    Bb B = bb_sub(a, wb);
    Bb C = bb_add(c, wd);
    Bb D = bb_sub(c, wd);

    // Stage s+1: butterfly pairs (A,C) and (B,D).
    uint step_s1 = height >> (stage + 2);
    uint tw2_idx = j * step_s1;
    uint tw3_idx = (j + half_s) * step_s1;

    Bb wC, wD;
    if (tw2_idx == 0) {
        wC = C;
    } else {
        wC = bb_mul(twiddles[tw2_idx], C);
    }
    wD = bb_mul(twiddles[tw3_idx], D);

    data[idx0] = bb_add(A, wC);
    data[idx1] = bb_add(B, wD);
    data[idx2] = bb_sub(A, wC);
    data[idx3] = bb_sub(B, wD);
}

// ── Phase A: radix-8 fused three-stage global butterfly (batched) ────
// Processes stages `stage`, `stage+1`, `stage+2` in a single dispatch.
// Each thread handles one radix-8 unit: reads 8 elements, performs three
// layers of butterflies (7 twiddle muls), writes 8 elements.
// 2D grid: gid.x = column, gid.y = radix-8 unit index.
kernel void bb_ntt_butterfly_r8(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& stage           [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 3;
    if (unit_id >= num_units || col >= width) return;

    uint half_s = 1u << stage;
    uint block_r8 = half_s << 3;
    uint block_idx = unit_id / half_s;
    uint j = unit_id % half_s;
    uint base = block_idx * block_r8 + j;

    // Load 8 elements at stride half_s.
    Bb e0 = data[(base              ) * width + col];
    Bb e1 = data[(base +     half_s ) * width + col];
    Bb e2 = data[(base + 2 * half_s ) * width + col];
    Bb e3 = data[(base + 3 * half_s ) * width + col];
    Bb e4 = data[(base + 4 * half_s ) * width + col];
    Bb e5 = data[(base + 5 * half_s ) * width + col];
    Bb e6 = data[(base + 6 * half_s ) * width + col];
    Bb e7 = data[(base + 7 * half_s ) * width + col];

    // ── Stage s: 4 butterflies, stride = half_s ──
    uint tw_s = j * (height >> (stage + 1));
    if (tw_s != 0) {
        Bb w = twiddles[tw_s];
        e1 = bb_mul(w, e1); e3 = bb_mul(w, e3);
        e5 = bb_mul(w, e5); e7 = bb_mul(w, e7);
    }
    Bb a0 = bb_add(e0, e1); Bb a1 = bb_sub(e0, e1);
    Bb a2 = bb_add(e2, e3); Bb a3 = bb_sub(e2, e3);
    Bb a4 = bb_add(e4, e5); Bb a5 = bb_sub(e4, e5);
    Bb a6 = bb_add(e6, e7); Bb a7 = bb_sub(e6, e7);

    // ── Stage s+1: 4 butterflies, stride = 2·half_s ──
    // Pairs: (a0,a2) (a1,a3) | (a4,a6) (a5,a7)
    uint step1 = height >> (stage + 2);
    uint tw1a = j * step1;
    uint tw1b = (j + half_s) * step1;
    if (tw1a != 0) {
        Bb wa = twiddles[tw1a];
        a2 = bb_mul(wa, a2); a6 = bb_mul(wa, a6);
    }
    { Bb wb = twiddles[tw1b];
      a3 = bb_mul(wb, a3); a7 = bb_mul(wb, a7); }
    Bb b0 = bb_add(a0, a2); Bb b2 = bb_sub(a0, a2);
    Bb b1 = bb_add(a1, a3); Bb b3 = bb_sub(a1, a3);
    Bb b4 = bb_add(a4, a6); Bb b6 = bb_sub(a4, a6);
    Bb b5 = bb_add(a5, a7); Bb b7 = bb_sub(a5, a7);

    // ── Stage s+2: 4 butterflies, stride = 4·half_s ──
    // Pairs: (b0,b4) (b1,b5) (b2,b6) (b3,b7)
    uint step2 = height >> (stage + 3);
    uint t0 = j * step2;
    if (t0 != 0) b4 = bb_mul(twiddles[t0], b4);
    b5 = bb_mul(twiddles[(j +     half_s) * step2], b5);
    b6 = bb_mul(twiddles[(j + 2 * half_s) * step2], b6);
    b7 = bb_mul(twiddles[(j + 3 * half_s) * step2], b7);

    data[(base              ) * width + col] = bb_add(b0, b4);
    data[(base +     half_s ) * width + col] = bb_add(b1, b5);
    data[(base + 2 * half_s ) * width + col] = bb_add(b2, b6);
    data[(base + 3 * half_s ) * width + col] = bb_add(b3, b7);
    data[(base + 4 * half_s ) * width + col] = bb_sub(b0, b4);
    data[(base + 5 * half_s ) * width + col] = bb_sub(b1, b5);
    data[(base + 6 * half_s ) * width + col] = bb_sub(b2, b6);
    data[(base + 7 * half_s ) * width + col] = bb_sub(b3, b7);
}
