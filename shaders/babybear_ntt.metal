// BabyBear field + forward NTT + Poseidon2 Merkle hashing + PoW grinding.
// All kernels work on ROW-MAJOR matrices: data[row * width + col].
// A single dispatch handles all columns — no CPU transpose needed.
//
// GPU/CPU speedup (best per n, across fold ∈ {1-8}, rate ∈ {1-3}):
//
//        │ Opt 1  │ Opt 2  │ Opt 3  │ Opt 4  │ Opt 5  │ Opt 6  │ Opt 7  │
//   n    │GPU NTT │Radix-16│+Merkle │+Fused  │+Rounds │+Thresh │+Grind  │
//   ─────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
//   18   │ 1.08x  │ 1.08x  │ 0.99x  │ 0.95x  │ 0.91x  │ 0.98x  │ 0.99x  │
//   20   │ 0.97x  │ 1.01x  │ 1.19x  │ 1.25x  │ 1.26x  │ 1.40x  │ 1.49x  │
//   22   │ 1.09x  │ 1.46x  │ 2.04x  │ 2.18x  │ 2.10x  │ 2.33x  │ 2.24x  │
//   24   │ 1.24x  │ 1.37x  │ 1.91x  │ 1.94x  │ 2.04x  │ 2.08x  │ 2.58x  │
//
// Full per-parameter tables: docs/gpu-optimizations.md
//
// ┌─────────────────────────────┬──────┬─────────────────────────────────────────┐
// │ Kernel                      │ Line │ Purpose                                 │
// ├─────────────────────────────┼──────┼─────────────────────────────────────────┤
// │ bb_bandwidth_test           │    6 │ Read-write bandwidth microbenchmark     │
// │ bb_bitrev_gather            │   67 │ Bit-reversal permutation (gather)       │
// │ bb_ntt_bitrev               │   86 │ Bit-reversal + twiddle (legacy)         │
// │ bb_ntt_shared_mem           │  121 │ Shared-memory radix-2 NTT block         │
// │ bb_ntt_shared_mem_gs        │  204 │ Shared-memory Gentleman-Sande NTT       │
// │ bb_ntt_butterfly            │  303 │ Global radix-2 butterfly                │
// │ bb_ntt_butterfly_r4         │  345 │ Global radix-4 butterfly                │
// │ bb_ntt_butterfly_r8         │  418 │ Global radix-8 butterfly                │
// │ bb_ntt_stockham             │  509 │ Stockham auto-sort NTT block            │
// │ bb_stockham_global_r2       │  656 │ Global Stockham radix-2 pass            │
// ├─────────────────────────────┼──────┼─────────────────────────────────────────┤
// │ bb_dif_r8                   │  707 │ DIF radix-8 in-place stage              │
// │ bb_dif_r4                   │  806 │ DIF radix-4 in-place stage              │
// │ bb_dif_r2                   │  869 │ DIF radix-2 in-place stage              │
// │ bb_dif_r16                  │  904 │ DIF radix-16 in-place stage             │
// │ bb_dif_r16_oop              │  992 │ DIF radix-16 out-of-place (first pass)  │
// │ bb_dif_r32                  │ 1077 │ DIF radix-32 in-place stage             │
// │ bb_dif_r32_bitrev           │ 1180 │ DIF radix-32 + bit-reversal gather      │
// │ bb_dif_r16_bitrev           │ 1270 │ DIF radix-16 + bit-reversal gather      │
// │ bb_dif_r8_bitrev            │ 1368 │ DIF radix-8  + bit-reversal gather      │
// │ bb_dif_r4_bitrev            │ 1471 │ DIF radix-4  + bit-reversal gather      │
// │ bb_dif_r2_bitrev            │ 1536 │ DIF radix-2  + bit-reversal gather      │
// │ bb_dif_shared_bitrev        │ 1652 │ Shared-memory DIF + bitrev (small n)    │
// ├─────────────────────────────┼──────┼─────────────────────────────────────────┤
// │ bb_transpose                │ 1578 │ Matrix transpose                        │
// │ bb_ntt_twiddle_transpose    │ 1602 │ Twiddle multiply + transpose            │
// ├─────────────────────────────┼──────┼─────────────────────────────────────────┤
// │ poseidon2_hash_leaves       │ 1838 │ Poseidon2 sponge over leaf rows         │
// │ poseidon2_merkle_compress   │ 1870 │ Poseidon2 2-to-1 Merkle compression     │
// │ poseidon2_pow_grind         │ 1913 │ Parallel PoW nonce search               │
// └─────────────────────────────┴──────┴─────────────────────────────────────────┘

// Simple read-write bandwidth test: data[i] += 1 for every element.
kernel void bb_bandwidth_test(
    device uint* data [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    data[gid] = data[gid] + 1u;
}

kernel void bb_buf_copy(
    device const uint* src [[buffer(0)]],
    device uint* dst       [[buffer(1)]],
    uint gid               [[thread_position_in_grid]]
) {
    dst[gid] = src[gid];
}

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
//
// Uses only 32-bit mulhi to avoid 64-bit arithmetic. Key insight:
// since MU * P ≡ 1 (mod 2^32), the low 32 bits of x and u are always
// equal, so (x - u) >> 32 = x_hi - u_hi (no borrow possible).
Bb bb_mul(Bb a, Bb b) {
    uint x_lo = a.v * b.v;
    uint x_hi = mulhi(a.v, b.v);
    uint t = x_lo * BB_MONTY_MU;
    uint u_hi = mulhi(t, BB_P);
    uint hi = x_hi - u_hi;
    return Bb{u_hi > x_hi ? hi + BB_P : hi};
}

Bb bb_neg(Bb a) { return Bb{a.v == 0 ? 0u : BB_P - a.v}; }

uint bit_reverse_n(uint v, uint bits) {
    uint rev = 0;
    for (uint i = 0; i < bits; i++) {
        rev = (rev << 1) | (v & 1);
        v >>= 1;
    }
    return rev;
}

// ── Bit-reverse gather (out-of-place, batched, row-major) ─────────────
// dst[row * W + col] = src[bitrev(row) * W + col]
// Writes are sequential/coalesced; reads are random (but from GPU-cached memory).
kernel void bb_bitrev_gather(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& log_height      [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= height || col >= width) return;

    uint rev = bit_reverse_n(row, log_height);
    dst[row * width + col] = src[rev * width + col];
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
    constant uint& do_bitrev       [[buffer(5)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Column-major layout: sdata[col * tg_size + row].
    // Consecutive threads access consecutive addresses → zero bank conflicts
    // for butterfly stages with stride < 32 (intra-SIMD-group).
    threadgroup Bb sdata[8192];

    uint base_row = tgid * tg_size;
    uint global_row = base_row + tid;
    uint gm_row = global_row * width;
    uint tile_w = 8192 / tg_size;

    uint sm_load_tid = do_bitrev ? bit_reverse_n(tid, log_block) : tid;

    for (uint col_base = 0; col_base < width; col_base += tile_w) {
        uint cw = min(tile_w, width - col_base);

        for (uint c = 0; c < cw; c++) {
            sdata[c * tg_size + sm_load_tid] = data[gm_row + col_base + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stage = 0; stage < log_block; stage++) {
            uint stride = 1u << stage;

            if ((tid & stride) == 0) {
                uint partner = tid | stride;
                uint local_idx = tid & (stride - 1);
                uint tw_idx = local_idx * (height >> (stage + 1));

                if (tw_idx == 0) {
                    for (uint c = 0; c < cw; c++) {
                        uint off = c * tg_size;
                        Bb a = sdata[off + tid];
                        Bb b = sdata[off + partner];
                        sdata[off + tid] = bb_add(a, b);
                        sdata[off + partner] = bb_sub(a, b);
                    }
                } else {
                    Bb w = twiddles[tw_idx];
                    for (uint c = 0; c < cw; c++) {
                        uint off = c * tg_size;
                        Bb a = sdata[off + tid];
                        Bb b = sdata[off + partner];
                        Bb wb = bb_mul(w, b);
                        sdata[off + tid] = bb_add(a, wb);
                        sdata[off + partner] = bb_sub(a, wb);
                    }
                }
            }

            if (stride >= 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint c = 0; c < cw; c++) {
            data[gm_row + col_base + c] = sdata[c * tg_size + tid];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Four-step FFT: gather-DFT-scatter with separate I/O buffers ──────
// Like bb_ntt_shared_mem but reads from `input` and writes to `output`
// using configurable row strides.  Always bit-reverses on load.
//
// load_stride = 1  → contiguous: load_row = tgid * block + tid
// load_stride > 1  → gather:     load_row = tgid + tid * load_stride
// store_stride = 1 → contiguous: store_row = tgid * block + tid
// store_stride > 1 → scatter:    store_row = tgid + tid * store_stride
kernel void bb_ntt_shared_mem_gs(
    device const Bb* input         [[buffer(0)]],
    device Bb* output              [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& log_block       [[buffer(5)]],
    constant uint& load_stride     [[buffer(6)]],
    constant uint& store_stride    [[buffer(7)]],
    constant uint& apply_twiddle   [[buffer(8)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup Bb sdata[8192];

    uint load_row  = (load_stride  == 1) ? (tgid * tg_size + tid)
                                         : (tgid + tid * load_stride);
    uint store_row = (store_stride == 1) ? (tgid * tg_size + tid)
                                         : (tgid + tid * store_stride);

    uint gm_load  = load_row  * width;
    uint gm_store = store_row * width;
    uint tile_w = 8192 / tg_size;

    uint sm_load_tid = bit_reverse_n(tid, log_block);

    Bb tw_factor;
    bool has_twiddle = false;
    if (apply_twiddle) {
        uint tw_raw = tgid * tid;
        if (tw_raw != 0) {
            has_twiddle = true;
            uint half_height = height >> 1;
            if (tw_raw >= half_height) {
                tw_factor = bb_neg(twiddles[tw_raw - half_height]);
            } else {
                tw_factor = twiddles[tw_raw];
            }
        }
    }

    for (uint col_base = 0; col_base < width; col_base += tile_w) {
        uint cw = min(tile_w, width - col_base);

        for (uint c = 0; c < cw; c++) {
            Bb val = input[gm_load + col_base + c];
            if (has_twiddle) val = bb_mul(tw_factor, val);
            sdata[c * tg_size + sm_load_tid] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stage = 0; stage < log_block; stage++) {
            uint stride = 1u << stage;

            if ((tid & stride) == 0) {
                uint partner = tid | stride;
                uint local_idx = tid & (stride - 1);
                uint tw_idx = local_idx * (height >> (stage + 1));

                if (tw_idx == 0) {
                    for (uint c = 0; c < cw; c++) {
                        uint off = c * tg_size;
                        Bb a = sdata[off + tid];
                        Bb b = sdata[off + partner];
                        sdata[off + tid] = bb_add(a, b);
                        sdata[off + partner] = bb_sub(a, b);
                    }
                } else {
                    Bb w = twiddles[tw_idx];
                    for (uint c = 0; c < cw; c++) {
                        uint off = c * tg_size;
                        Bb a = sdata[off + tid];
                        Bb b = sdata[off + partner];
                        Bb wb = bb_mul(w, b);
                        sdata[off + tid] = bb_add(a, wb);
                        sdata[off + partner] = bb_sub(a, wb);
                    }
                }
            }

            if (stride >= 32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        for (uint c = 0; c < cw; c++) {
            output[gm_store + col_base + c] = sdata[c * tg_size + tid];
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

// ── Stockham radix-4 DIT NTT (no bit-reversal needed) ────────────────
// Processes 4*tg_size elements per block using Stockham autosort.
// Reads in natural order, outputs in natural order.
// Uses column-major shared memory layout: sdata[col * block_size + row]
// for sequential (coalesced) access within SIMD groups.
//
// Each thread handles 4 elements per pass.  tg_size threads →
// block_size = 4 * tg_size elements.  log_block / 2 radix-4 passes,
// plus one radix-2 pass if log_block is odd.
//
// Twiddle index: omega_N^(k * j * height / (4 * stride)),
// where height is the FULL NTT size (works for both complete NTTs
// and sub-NTTs in the four-step framework).
constant uint BB_MONTY_ONE = 0x0ffffffeu; // 1 in Montgomery form

kernel void bb_ntt_stockham(
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

    uint block_size = 1u << log_block;
    uint quarter    = block_size >> 2;   // = tg_size
    uint tile_w     = min(8u, 8192 / block_size);
    uint base_row   = tgid * block_size;
    uint n_passes   = log_block >> 1;

    // omega_4 = omega_N^(N/4) — the "imaginary unit" for radix-4 butterflies.
    Bb omega_4 = twiddles[height >> 2];

    for (uint col_base = 0; col_base < width; col_base += tile_w) {
        uint cw = min(tile_w, width - col_base);

        // ═══ Pass 0: read from global memory, no external twiddle ═══
        // Read positions: tid, tid+quarter, tid+2*quarter, tid+3*quarter
        // (each offset by base_row in global, natural order).
        for (uint c = 0; c < cw; c++) {
            uint sm = c * block_size;
            uint gm = col_base + c;

            Bb r0 = data[(base_row + tid              ) * width + gm];
            Bb r1 = data[(base_row + tid +     quarter) * width + gm];
            Bb r2 = data[(base_row + tid + 2 * quarter) * width + gm];
            Bb r3 = data[(base_row + tid + 3 * quarter) * width + gm];

            // Radix-4 DIT butterfly (no twiddle for pass 0)
            Bb t0 = bb_add(r0, r2);
            Bb t1 = bb_sub(r0, r2);
            Bb t2 = bb_add(r1, r3);
            Bb t3 = bb_mul(omega_4, bb_sub(r1, r3));

            // Stockham write (stride=1): wr = tid * 4 + j
            uint wr = tid << 2;
            sdata[sm + wr    ] = bb_add(t0, t2);
            sdata[sm + wr + 1] = bb_add(t1, t3);
            sdata[sm + wr + 2] = bb_sub(t0, t2);
            sdata[sm + wr + 3] = bb_sub(t1, t3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══ Intermediate radix-4 passes ═══
        // Two barriers per pass: one to ensure all reads complete before any
        // writes (Stockham read/write addresses overlap in the same buffer),
        // one after writes to make the results visible for the next pass.
        uint stride = 4;
        for (uint pass = 1; pass < n_passes; pass++) {
            uint pos = tid & (stride - 1);
            uint grp = tid / stride;
            uint tw_step = height / (stride << 2);

            Bb tw1, tw2, tw3;
            if (pos == 0) {
                tw1 = Bb{BB_MONTY_ONE};
                tw2 = Bb{BB_MONTY_ONE};
                tw3 = Bb{BB_MONTY_ONE};
            } else {
                tw1 = twiddles[pos * tw_step];
                tw2 = bb_mul(tw1, tw1);
                tw3 = bb_mul(tw2, tw1);
            }

            uint wr = grp * (stride << 2) + pos;

            for (uint c = 0; c < cw; c++) {
                uint sm = c * block_size;
                Bb r0 = sdata[sm + tid              ];
                Bb r1 = sdata[sm + tid +     quarter];
                Bb r2 = sdata[sm + tid + 2 * quarter];
                Bb r3 = sdata[sm + tid + 3 * quarter];

                r1 = bb_mul(tw1, r1);
                r2 = bb_mul(tw2, r2);
                r3 = bb_mul(tw3, r3);

                Bb t0 = bb_add(r0, r2);
                Bb t1 = bb_sub(r0, r2);
                Bb t2 = bb_add(r1, r3);
                Bb t3 = bb_mul(omega_4, bb_sub(r1, r3));

                threadgroup_barrier(mem_flags::mem_threadgroup);

                sdata[sm + wr              ] = bb_add(t0, t2);
                sdata[sm + wr +     stride ] = bb_add(t1, t3);
                sdata[sm + wr + 2 * stride ] = bb_sub(t0, t2);
                sdata[sm + wr + 3 * stride ] = bb_sub(t1, t3);

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            stride <<= 2;
        }

        // ═══ Final radix-2 pass (if log_block is odd) ═══
        // N/2 butterfly pairs but only N/4 threads → each thread handles 2 pairs.
        // Pair addresses are non-overlapping so no intermediate barrier needed.
        if (log_block & 1u) {
            uint half_block = block_size >> 1;
            uint tw_step = height / block_size;

            for (uint c = 0; c < cw; c++) {
                uint sm = c * block_size;
                for (uint pair = 0; pair < 2; pair++) {
                    uint k = tid + pair * quarter;
                    Bb r0 = sdata[sm + k             ];
                    Bb r1 = sdata[sm + k + half_block];

                    uint tw_idx = k * tw_step;
                    if (tw_idx != 0) {
                        r1 = bb_mul(twiddles[tw_idx], r1);
                    }
                    sdata[sm + k             ] = bb_add(r0, r1);
                    sdata[sm + k + half_block] = bb_sub(r0, r1);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ═══ Store tile back to global memory ═══
        for (uint j = 0; j < 4; j++) {
            uint elem = tid + j * quarter;
            for (uint c = 0; c < cw; c++) {
                data[(base_row + elem) * width + col_base + c] =
                    sdata[c * block_size + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Stockham radix-4 with gather/scatter I/O (for four-step FFT) ────
// Like bb_ntt_stockham but reads from `input` and writes to `output`
// using configurable row strides, enabling the four-step FFT decomposition.
//
// load_stride = 1  → contiguous: row(i) = tgid * block_size + i
// load_stride > 1  → gather:     row(i) = tgid + i * load_stride
// store_stride = 1 → contiguous: row(i) = tgid * block_size + i
// store_stride > 1 → scatter:    row(i) = tgid + i * store_stride
//
// When apply_twiddle != 0, each loaded element is multiplied by
// ω_N^{tgid * i} where i is the element's position within the block.
// This fuses the inter-block twiddle of the Cooley-Tukey four-step.
kernel void bb_ntt_stockham_gs(
    device const Bb* input          [[buffer(0)]],
    device Bb* output               [[buffer(1)]],
    device const Bb* twiddles       [[buffer(2)]],
    constant uint& height           [[buffer(3)]],
    constant uint& width            [[buffer(4)]],
    constant uint& log_block        [[buffer(5)]],
    constant uint& load_stride      [[buffer(6)]],
    constant uint& store_stride     [[buffer(7)]],
    constant uint& apply_twiddle    [[buffer(8)]],
    uint tid   [[thread_index_in_threadgroup]],
    uint tgid  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup Bb sdata[8192];

    uint block_size = 1u << log_block;
    uint quarter    = block_size >> 2;   // = tg_size
    uint tile_w     = min(8u, 8192 / block_size);
    uint n_passes   = log_block >> 1;
    uint half_height = height >> 1;

    Bb omega_4 = twiddles[height >> 2];

    // Precompute twiddle factors for the four elements this thread loads.
    Bb tw_f[4];
    bool has_twiddle[4] = {false, false, false, false};
    if (apply_twiddle) {
        for (uint j = 0; j < 4; j++) {
            uint elem = tid + j * quarter;
            uint tw_raw = tgid * elem;
            if (tw_raw != 0) {
                has_twiddle[j] = true;
                if (tw_raw >= half_height) {
                    tw_f[j] = bb_neg(twiddles[tw_raw - half_height]);
                } else {
                    tw_f[j] = twiddles[tw_raw];
                }
            }
        }
    }

    for (uint col_base = 0; col_base < width; col_base += tile_w) {
        uint cw = min(tile_w, width - col_base);

        // ═══ Pass 0: gather from input, optional twiddle, first R4 butterfly ═══
        for (uint c = 0; c < cw; c++) {
            uint sm = c * block_size;
            uint gm = col_base + c;

            Bb r0, r1, r2, r3;
            if (load_stride == 1) {
                uint base_row = tgid * block_size;
                r0 = input[(base_row + tid              ) * width + gm];
                r1 = input[(base_row + tid +     quarter) * width + gm];
                r2 = input[(base_row + tid + 2 * quarter) * width + gm];
                r3 = input[(base_row + tid + 3 * quarter) * width + gm];
            } else {
                r0 = input[(tgid + (tid              ) * load_stride) * width + gm];
                r1 = input[(tgid + (tid +     quarter) * load_stride) * width + gm];
                r2 = input[(tgid + (tid + 2 * quarter) * load_stride) * width + gm];
                r3 = input[(tgid + (tid + 3 * quarter) * load_stride) * width + gm];
            }

            if (apply_twiddle) {
                if (has_twiddle[0]) r0 = bb_mul(tw_f[0], r0);
                if (has_twiddle[1]) r1 = bb_mul(tw_f[1], r1);
                if (has_twiddle[2]) r2 = bb_mul(tw_f[2], r2);
                if (has_twiddle[3]) r3 = bb_mul(tw_f[3], r3);
            }

            Bb t0 = bb_add(r0, r2);
            Bb t1 = bb_sub(r0, r2);
            Bb t2 = bb_add(r1, r3);
            Bb t3 = bb_mul(omega_4, bb_sub(r1, r3));

            uint wr = tid << 2;
            sdata[sm + wr    ] = bb_add(t0, t2);
            sdata[sm + wr + 1] = bb_add(t1, t3);
            sdata[sm + wr + 2] = bb_sub(t0, t2);
            sdata[sm + wr + 3] = bb_sub(t1, t3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══ Intermediate radix-4 passes ═══
        uint stride = 4;
        for (uint pass = 1; pass < n_passes; pass++) {
            uint pos = tid & (stride - 1);
            uint grp = tid / stride;
            uint tw_step = height / (stride << 2);

            Bb tw1, tw2, tw3;
            if (pos == 0) {
                tw1 = Bb{BB_MONTY_ONE};
                tw2 = Bb{BB_MONTY_ONE};
                tw3 = Bb{BB_MONTY_ONE};
            } else {
                tw1 = twiddles[pos * tw_step];
                tw2 = bb_mul(tw1, tw1);
                tw3 = bb_mul(tw2, tw1);
            }

            uint wr = grp * (stride << 2) + pos;

            for (uint c = 0; c < cw; c++) {
                uint sm = c * block_size;
                Bb r0_v = sdata[sm + tid              ];
                Bb r1_v = sdata[sm + tid +     quarter];
                Bb r2_v = sdata[sm + tid + 2 * quarter];
                Bb r3_v = sdata[sm + tid + 3 * quarter];

                r1_v = bb_mul(tw1, r1_v);
                r2_v = bb_mul(tw2, r2_v);
                r3_v = bb_mul(tw3, r3_v);

                Bb t0_v = bb_add(r0_v, r2_v);
                Bb t1_v = bb_sub(r0_v, r2_v);
                Bb t2_v = bb_add(r1_v, r3_v);
                Bb t3_v = bb_mul(omega_4, bb_sub(r1_v, r3_v));

                threadgroup_barrier(mem_flags::mem_threadgroup);

                sdata[sm + wr              ] = bb_add(t0_v, t2_v);
                sdata[sm + wr +     stride ] = bb_add(t1_v, t3_v);
                sdata[sm + wr + 2 * stride ] = bb_sub(t0_v, t2_v);
                sdata[sm + wr + 3 * stride ] = bb_sub(t1_v, t3_v);

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            stride <<= 2;
        }

        // ═══ Final radix-2 pass (if log_block is odd) ═══
        if (log_block & 1u) {
            uint half_block = block_size >> 1;
            uint tw_step_f = height / block_size;

            for (uint c = 0; c < cw; c++) {
                uint sm = c * block_size;
                for (uint pair = 0; pair < 2; pair++) {
                    uint k = tid + pair * quarter;
                    Bb r0_v = sdata[sm + k             ];
                    Bb r1_v = sdata[sm + k + half_block];

                    uint tw_idx = k * tw_step_f;
                    if (tw_idx != 0) {
                        r1_v = bb_mul(twiddles[tw_idx], r1_v);
                    }
                    sdata[sm + k             ] = bb_add(r0_v, r1_v);
                    sdata[sm + k + half_block] = bb_sub(r0_v, r1_v);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ═══ Store tile to output with scatter stride ═══
        for (uint j = 0; j < 4; j++) {
            uint elem = tid + j * quarter;
            uint store_row = (store_stride == 1) ? (tgid * block_size + elem)
                                                  : (tgid + elem * store_stride);
            for (uint c = 0; c < cw; c++) {
                output[store_row * width + col_base + c] =
                    sdata[c * block_size + elem];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Stockham global radix-2: out-of-place, one stage ────────────────
// Reads from src, writes to dst.  Both reads and writes are coalesced.
// p = 2^k where k is the current completed-stage count.
// Thread tid (0..N/2-1) handles one butterfly:
//   a = src[tid],  b = src[tid + N/2]
//   dst[group*2p + pos]     = a + w*b
//   dst[group*2p + pos + p] = a - w*b
// where pos = tid % p, group = tid / p, w = ω_N^(pos * N/(2p)).
kernel void bb_stockham_global_r2(
    device const Bb* src       [[buffer(0)]],
    device Bb* dst             [[buffer(1)]],
    device const Bb* twiddles  [[buffer(2)]],
    constant uint& height      [[buffer(3)]],
    constant uint& width       [[buffer(4)]],
    constant uint& p           [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint tid = gid.y;
    uint half_n = height >> 1;
    if (col >= width || tid >= half_n) return;

    uint pos   = tid % p;
    uint group = tid / p;

    Bb a = src[tid * width + col];
    Bb b = src[(tid + half_n) * width + col];

    uint tw_idx = pos * (half_n / p);
    if (tw_idx != 0) {
        b = bb_mul(twiddles[tw_idx], b);
    }

    uint dst_a = group * 2 * p + pos;
    uint dst_b = dst_a + p;
    dst[dst_a * width + col] = bb_add(a, b);
    dst[dst_b * width + col] = bb_sub(a, b);
}

// ═══════════════════════════════════════════════════════════════════════
// DIF (Decimation in Frequency) kernels — Gentleman-Sande butterflies.
// Process stages from LARGE stride to SMALL stride.
// Input is in NATURAL order, output is in BIT-REVERSED order.
// Eliminates the need for a separate bit-reversal pass.
// ═══════════════════════════════════════════════════════════════════════
//
// DIF butterfly:  a' = a + b,  b' = (a - b) * w
//
// For DIF stage s (0 = largest stride, log_n-1 = smallest):
//   stride = height >> (s + 1)
//   twiddle for position k within the stride: omega_N^(k << s) = twiddles[k << s]
//
// DIF R8: fuses 3 consecutive DIF stages (dif_stage, dif_stage+1, dif_stage+2).
// Parameterized by `dif_stage` = the first (outermost) DIF stage number.
//
// The 8 elements per unit are at stride h = height >> (dif_stage + 3):
//   e[i] = data[(base + i*h) * width + col]  for i=0..7
// where base = group * 8h + j,  j = unit_id % h,  group = unit_id / h.

kernel void bb_dif_r8(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& dif_stage       [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 3;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 3);
    uint block_r8 = h << 3;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r8 + j;

    Bb e0 = data[(base          ) * width + col];
    Bb e1 = data[(base +     h  ) * width + col];
    Bb e2 = data[(base + 2 * h  ) * width + col];
    Bb e3 = data[(base + 3 * h  ) * width + col];
    Bb e4 = data[(base + 4 * h  ) * width + col];
    Bb e5 = data[(base + 5 * h  ) * width + col];
    Bb e6 = data[(base + 6 * h  ) * width + col];
    Bb e7 = data[(base + 7 * h  ) * width + col];

    uint step_s = 1u << dif_stage;
    uint quarter_n = height >> 2;

    // ── DIF stage dif_stage: stride = 4h, pairs (0,4)(1,5)(2,6)(3,7) ──
    {
        uint tw0_idx = j * step_s;
        uint tw1_idx = tw0_idx + (h * step_s);  // = tw0 + height/8
        uint tw2_idx = tw0_idx + quarter_n;      // = tw0 + height/4
        uint tw3_idx = tw1_idx + quarter_n;      // = tw0 + 3*height/8

        Bb s0 = bb_add(e0, e4); Bb d0 = bb_sub(e0, e4);
        Bb s1 = bb_add(e1, e5); Bb d1 = bb_sub(e1, e5);
        Bb s2 = bb_add(e2, e6); Bb d2 = bb_sub(e2, e6);
        Bb s3 = bb_add(e3, e7); Bb d3 = bb_sub(e3, e7);

        e0 = s0; e1 = s1; e2 = s2; e3 = s3;
        e4 = (tw0_idx == 0) ? d0 : bb_mul(twiddles[tw0_idx], d0);
        e5 = bb_mul(twiddles[tw1_idx], d1);
        e6 = bb_mul(twiddles[tw2_idx], d2);
        e7 = bb_mul(twiddles[tw3_idx], d3);
    }

    // ── DIF stage dif_stage+1: stride = 2h, pairs (0,2)(1,3)(4,6)(5,7) ──
    {
        uint step_s1 = step_s << 1;
        uint tw4_idx = j * step_s1;
        uint tw5_idx = tw4_idx + (h * step_s1); // = tw4 + height/4

        Bb s0 = bb_add(e0, e2); Bb d0 = bb_sub(e0, e2);
        Bb s1 = bb_add(e1, e3); Bb d1 = bb_sub(e1, e3);
        Bb s4 = bb_add(e4, e6); Bb d4 = bb_sub(e4, e6);
        Bb s5 = bb_add(e5, e7); Bb d5 = bb_sub(e5, e7);

        e0 = s0; e1 = s1; e4 = s4; e5 = s5;
        e2 = (tw4_idx == 0) ? d0 : bb_mul(twiddles[tw4_idx], d0);
        e3 = bb_mul(twiddles[tw5_idx], d1);
        e6 = (tw4_idx == 0) ? d4 : bb_mul(twiddles[tw4_idx], d4);
        e7 = bb_mul(twiddles[tw5_idx], d5);
    }

    // ── DIF stage dif_stage+2: stride = h, pairs (0,1)(2,3)(4,5)(6,7) ──
    {
        uint step_s2 = step_s << 2;
        uint tw6_idx = j * step_s2;

        Bb s0 = bb_add(e0, e1); Bb d0 = bb_sub(e0, e1);
        Bb s2 = bb_add(e2, e3); Bb d2 = bb_sub(e2, e3);
        Bb s4 = bb_add(e4, e5); Bb d4 = bb_sub(e4, e5);
        Bb s6 = bb_add(e6, e7); Bb d6 = bb_sub(e6, e7);

        e0 = s0; e2 = s2; e4 = s4; e6 = s6;
        if (tw6_idx == 0) {
            e1 = d0; e3 = d2; e5 = d4; e7 = d6;
        } else {
            Bb w = twiddles[tw6_idx];
            e1 = bb_mul(w, d0); e3 = bb_mul(w, d2);
            e5 = bb_mul(w, d4); e7 = bb_mul(w, d6);
        }
    }

    data[(base          ) * width + col] = e0;
    data[(base +     h  ) * width + col] = e1;
    data[(base + 2 * h  ) * width + col] = e2;
    data[(base + 3 * h  ) * width + col] = e3;
    data[(base + 4 * h  ) * width + col] = e4;
    data[(base + 5 * h  ) * width + col] = e5;
    data[(base + 6 * h  ) * width + col] = e6;
    data[(base + 7 * h  ) * width + col] = e7;
}

// DIF R4: fuses 2 consecutive DIF stages.
kernel void bb_dif_r4(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& dif_stage       [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 2;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 2);
    uint block_r4 = h << 2;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r4 + j;

    Bb e0 = data[(base          ) * width + col];
    Bb e1 = data[(base +     h  ) * width + col];
    Bb e2 = data[(base + 2 * h  ) * width + col];
    Bb e3 = data[(base + 3 * h  ) * width + col];

    uint step_s = 1u << dif_stage;

    // DIF stage dif_stage: stride = 2h, pairs (0,2)(1,3)
    {
        uint tw0_idx = j * step_s;
        uint tw1_idx = tw0_idx + (h * step_s);

        Bb s0 = bb_add(e0, e2); Bb d0 = bb_sub(e0, e2);
        Bb s1 = bb_add(e1, e3); Bb d1 = bb_sub(e1, e3);

        e0 = s0; e1 = s1;
        e2 = (tw0_idx == 0) ? d0 : bb_mul(twiddles[tw0_idx], d0);
        e3 = bb_mul(twiddles[tw1_idx], d1);
    }

    // DIF stage dif_stage+1: stride = h, pairs (0,1)(2,3)
    {
        uint step_s1 = step_s << 1;
        uint tw2_idx = j * step_s1;

        Bb s0 = bb_add(e0, e1); Bb d0 = bb_sub(e0, e1);
        Bb s2 = bb_add(e2, e3); Bb d2 = bb_sub(e2, e3);

        e0 = s0; e2 = s2;
        if (tw2_idx == 0) {
            e1 = d0; e3 = d2;
        } else {
            Bb w = twiddles[tw2_idx];
            e1 = bb_mul(w, d0); e3 = bb_mul(w, d2);
        }
    }

    data[(base          ) * width + col] = e0;
    data[(base +     h  ) * width + col] = e1;
    data[(base + 2 * h  ) * width + col] = e2;
    data[(base + 3 * h  ) * width + col] = e3;
}

// DIF R2: single DIF stage.
kernel void bb_dif_r2(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& dif_stage       [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint butterfly_id = gid.y;
    uint num_butterflies = height >> 1;
    if (butterfly_id >= num_butterflies || col >= width) return;

    uint stride = height >> (dif_stage + 1);
    uint block_size = stride << 1;
    uint block_idx = butterfly_id / stride;
    uint k = butterfly_id % stride;
    uint i = block_idx * block_size + k;
    uint j = i + stride;

    uint idx_a = i * width + col;
    uint idx_b = j * width + col;
    Bb a = data[idx_a];
    Bb b = data[idx_b];

    Bb sum = bb_add(a, b);
    Bb diff = bb_sub(a, b);

    uint tw_idx = k * (1u << dif_stage);
    data[idx_a] = sum;
    data[idx_b] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
}

// DIF R16: fuses 4 consecutive DIF stages (dif_stage..dif_stage+3).
// Each thread processes 16 elements.  h = height >> (dif_stage + 4).
kernel void bb_dif_r16(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& dif_stage       [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 4;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 4);
    uint block_r16 = h << 4;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r16 + j;

    Bb e[16];
    for (uint i = 0; i < 16; i++)
        e[i] = data[(base + i * h) * width + col];

    uint step_s = 1u << dif_stage;
    uint sixteenth_n = height >> 4; // h * step_s = height/16

    // ── DIF stage dif_stage: stride = 8h, pairs (k, k+8) for k=0..7 ──
    {
        for (uint k = 0; k < 8; k++) {
            uint tw_idx = j * step_s + k * sixteenth_n;
            Bb sum = bb_add(e[k], e[k + 8]);
            Bb diff = bb_sub(e[k], e[k + 8]);
            e[k] = sum;
            e[k + 8] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    // ── DIF stage dif_stage+1: stride = 4h ──
    {
        uint step_s1 = step_s << 1;
        uint eighth_n = height >> 3;
        for (uint hh = 0; hh < 2; hh++) {
            uint off = hh * 8;
            for (uint k = 0; k < 4; k++) {
                uint tw_idx = j * step_s1 + k * eighth_n;
                Bb sum = bb_add(e[off + k], e[off + k + 4]);
                Bb diff = bb_sub(e[off + k], e[off + k + 4]);
                e[off + k] = sum;
                e[off + k + 4] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // ── DIF stage dif_stage+2: stride = 2h ──
    {
        uint step_s2 = step_s << 2;
        uint quarter_n = height >> 2;
        for (uint q = 0; q < 4; q++) {
            uint off = q * 4;
            for (uint k = 0; k < 2; k++) {
                uint tw_idx = j * step_s2 + k * quarter_n;
                Bb sum = bb_add(e[off + k], e[off + k + 2]);
                Bb diff = bb_sub(e[off + k], e[off + k + 2]);
                e[off + k] = sum;
                e[off + k + 2] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // ── DIF stage dif_stage+3: stride = h ──
    {
        uint step_s3 = step_s << 3;
        for (uint p = 0; p < 8; p++) {
            uint off = p * 2;
            uint tw_idx = j * step_s3;
            Bb sum = bb_add(e[off], e[off + 1]);
            Bb diff = bb_sub(e[off], e[off + 1]);
            e[off] = sum;
            e[off + 1] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    for (uint i = 0; i < 16; i++)
        data[(base + i * h) * width + col] = e[i];
}

// DIF R16 out-of-place: reads from src, writes to dst at the same indices.
// Used for the first dispatch to copy data from zero-copy input to managed.
kernel void bb_dif_r16_oop(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 4;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 4);
    uint block_r16 = h << 4;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r16 + j;

    Bb e[16];
    for (uint i = 0; i < 16; i++)
        e[i] = src[(base + i * h) * width + col];

    uint step_s = 1u << dif_stage;
    uint sixteenth_n = height >> 4;

    {
        for (uint k = 0; k < 8; k++) {
            uint tw_idx = j * step_s + k * sixteenth_n;
            Bb sum = bb_add(e[k], e[k + 8]);
            Bb diff = bb_sub(e[k], e[k + 8]);
            e[k] = sum;
            e[k + 8] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    {
        uint step_s1 = step_s << 1;
        uint eighth_n = height >> 3;
        for (uint hh = 0; hh < 2; hh++) {
            uint off = hh * 8;
            for (uint k = 0; k < 4; k++) {
                uint tw_idx = j * step_s1 + k * eighth_n;
                Bb sum = bb_add(e[off + k], e[off + k + 4]);
                Bb diff = bb_sub(e[off + k], e[off + k + 4]);
                e[off + k] = sum;
                e[off + k + 4] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    {
        uint step_s2 = step_s << 2;
        uint quarter_n = height >> 2;
        for (uint q = 0; q < 4; q++) {
            uint off = q * 4;
            for (uint k = 0; k < 2; k++) {
                uint tw_idx = j * step_s2 + k * quarter_n;
                Bb sum = bb_add(e[off + k], e[off + k + 2]);
                Bb diff = bb_sub(e[off + k], e[off + k + 2]);
                e[off + k] = sum;
                e[off + k + 2] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    {
        uint step_s3 = step_s << 3;
        for (uint p = 0; p < 8; p++) {
            uint off = p * 2;
            uint tw_idx = j * step_s3;
            Bb sum = bb_add(e[off], e[off + 1]);
            Bb diff = bb_sub(e[off], e[off + 1]);
            e[off] = sum;
            e[off + 1] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    for (uint i = 0; i < 16; i++)
        dst[(base + i * h) * width + col] = e[i];
}

// DIF R32: fuses 5 consecutive DIF stages (dif_stage..dif_stage+4).
// Each thread processes 32 elements.  h = height >> (dif_stage + 5).
kernel void bb_dif_r32(
    device Bb* data                [[buffer(0)]],
    device const Bb* twiddles      [[buffer(1)]],
    constant uint& height          [[buffer(2)]],
    constant uint& width           [[buffer(3)]],
    constant uint& dif_stage       [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 5;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 5);
    uint block_r32 = h << 5;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r32 + j;

    Bb e[32];
    for (uint i = 0; i < 32; i++)
        e[i] = data[(base + i * h) * width + col];

    uint step_s = 1u << dif_stage;

    // Stage 0: stride = 16h, pairs (k, k+16)
    {
        uint spacing = height >> 5;
        for (uint k = 0; k < 16; k++) {
            uint tw_idx = j * step_s + k * spacing;
            Bb sum = bb_add(e[k], e[k + 16]);
            Bb diff = bb_sub(e[k], e[k + 16]);
            e[k] = sum;
            e[k + 16] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    // Stage 1: stride = 8h, pairs within 2 halves of 16
    {
        uint step_s1 = step_s << 1;
        uint spacing = height >> 4;
        for (uint hh = 0; hh < 2; hh++) {
            uint off = hh * 16;
            for (uint k = 0; k < 8; k++) {
                uint tw_idx = j * step_s1 + k * spacing;
                Bb sum = bb_add(e[off + k], e[off + k + 8]);
                Bb diff = bb_sub(e[off + k], e[off + k + 8]);
                e[off + k] = sum;
                e[off + k + 8] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // Stage 2: stride = 4h, pairs within 4 quarters of 8
    {
        uint step_s2 = step_s << 2;
        uint spacing = height >> 3;
        for (uint q = 0; q < 4; q++) {
            uint off = q * 8;
            for (uint k = 0; k < 4; k++) {
                uint tw_idx = j * step_s2 + k * spacing;
                Bb sum = bb_add(e[off + k], e[off + k + 4]);
                Bb diff = bb_sub(e[off + k], e[off + k + 4]);
                e[off + k] = sum;
                e[off + k + 4] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // Stage 3: stride = 2h, pairs within 8 octets of 4
    {
        uint step_s3 = step_s << 3;
        uint spacing = height >> 2;
        for (uint o = 0; o < 8; o++) {
            uint off = o * 4;
            for (uint k = 0; k < 2; k++) {
                uint tw_idx = j * step_s3 + k * spacing;
                Bb sum = bb_add(e[off + k], e[off + k + 2]);
                Bb diff = bb_sub(e[off + k], e[off + k + 2]);
                e[off + k] = sum;
                e[off + k + 2] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // Stage 4: stride = h, 16 pairs
    {
        uint step_s4 = step_s << 4;
        uint tw_idx = j * step_s4;
        for (uint p = 0; p < 16; p++) {
            uint off = p * 2;
            Bb sum = bb_add(e[off], e[off + 1]);
            Bb diff = bb_sub(e[off], e[off + 1]);
            e[off] = sum;
            e[off + 1] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    for (uint i = 0; i < 32; i++)
        data[(base + i * h) * width + col] = e[i];
}

// DIF R32 + fused bit-reversal (out-of-place).
kernel void bb_dif_r32_bitrev(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    constant uint& log_n           [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 5;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 5);
    uint block_r32 = h << 5;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r32 + j;

    Bb e[32];
    for (uint i = 0; i < 32; i++)
        e[i] = src[(base + i * h) * width + col];

    uint step_s = 1u << dif_stage;

    { uint spacing = height >> 5;
      for (uint k = 0; k < 16; k++) {
          uint tw_idx = j * step_s + k * spacing;
          Bb sum = bb_add(e[k], e[k + 16]);
          Bb diff = bb_sub(e[k], e[k + 16]);
          e[k] = sum;
          e[k + 16] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
      }
    }
    { uint step_s1 = step_s << 1; uint spacing = height >> 4;
      for (uint hh = 0; hh < 2; hh++) {
          uint off = hh * 16;
          for (uint k = 0; k < 8; k++) {
              uint tw_idx = j * step_s1 + k * spacing;
              Bb sum = bb_add(e[off + k], e[off + k + 8]);
              Bb diff = bb_sub(e[off + k], e[off + k + 8]);
              e[off + k] = sum;
              e[off + k + 8] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
          }
      }
    }
    { uint step_s2 = step_s << 2; uint spacing = height >> 3;
      for (uint q = 0; q < 4; q++) {
          uint off = q * 8;
          for (uint k = 0; k < 4; k++) {
              uint tw_idx = j * step_s2 + k * spacing;
              Bb sum = bb_add(e[off + k], e[off + k + 4]);
              Bb diff = bb_sub(e[off + k], e[off + k + 4]);
              e[off + k] = sum;
              e[off + k + 4] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
          }
      }
    }
    { uint step_s3 = step_s << 3; uint spacing = height >> 2;
      for (uint o = 0; o < 8; o++) {
          uint off = o * 4;
          for (uint k = 0; k < 2; k++) {
              uint tw_idx = j * step_s3 + k * spacing;
              Bb sum = bb_add(e[off + k], e[off + k + 2]);
              Bb diff = bb_sub(e[off + k], e[off + k + 2]);
              e[off + k] = sum;
              e[off + k + 2] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
          }
      }
    }
    { uint step_s4 = step_s << 4;
      uint tw_idx = j * step_s4;
      for (uint p = 0; p < 16; p++) {
          uint off = p * 2;
          Bb sum = bb_add(e[off], e[off + 1]);
          Bb diff = bb_sub(e[off], e[off + 1]);
          e[off] = sum;
          e[off + 1] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
      }
    }

    for (uint i = 0; i < 32; i++) {
        uint rev_row = bit_reverse_n(base + i * h, log_n);
        dst[rev_row * width + col] = e[i];
    }
}

// DIF R16 + fused bit-reversal (out-of-place).
kernel void bb_dif_r16_bitrev(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    constant uint& log_n           [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 4;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 4);
    uint block_r16 = h << 4;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r16 + j;

    Bb e[16];
    for (uint i = 0; i < 16; i++)
        e[i] = src[(base + i * h) * width + col];

    uint step_s = 1u << dif_stage;
    uint sixteenth_n = height >> 4;

    // ── DIF stage dif_stage: stride = 8h ──
    {
        for (uint k = 0; k < 8; k++) {
            uint tw_idx = j * step_s + k * sixteenth_n;
            Bb sum = bb_add(e[k], e[k + 8]);
            Bb diff = bb_sub(e[k], e[k + 8]);
            e[k] = sum;
            e[k + 8] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    // ── DIF stage dif_stage+1: stride = 4h ──
    {
        uint step_s1 = step_s << 1;
        uint eighth_n = height >> 3;
        for (uint hh = 0; hh < 2; hh++) {
            uint off = hh * 8;
            for (uint k = 0; k < 4; k++) {
                uint tw_idx = j * step_s1 + k * eighth_n;
                Bb sum = bb_add(e[off + k], e[off + k + 4]);
                Bb diff = bb_sub(e[off + k], e[off + k + 4]);
                e[off + k] = sum;
                e[off + k + 4] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // ── DIF stage dif_stage+2: stride = 2h ──
    {
        uint step_s2 = step_s << 2;
        uint quarter_n = height >> 2;
        for (uint q = 0; q < 4; q++) {
            uint off = q * 4;
            for (uint k = 0; k < 2; k++) {
                uint tw_idx = j * step_s2 + k * quarter_n;
                Bb sum = bb_add(e[off + k], e[off + k + 2]);
                Bb diff = bb_sub(e[off + k], e[off + k + 2]);
                e[off + k] = sum;
                e[off + k + 2] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
            }
        }
    }

    // ── DIF stage dif_stage+3: stride = h ──
    {
        uint step_s3 = step_s << 3;
        for (uint p = 0; p < 8; p++) {
            uint off = p * 2;
            uint tw_idx = j * step_s3;
            Bb sum = bb_add(e[off], e[off + 1]);
            Bb diff = bb_sub(e[off], e[off + 1]);
            e[off] = sum;
            e[off + 1] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }
    }

    uint rows[16];
    for (uint i = 0; i < 16; i++)
        rows[i] = base + i * h;

    for (uint i = 0; i < 16; i++) {
        uint rev_row = bit_reverse_n(rows[i], log_n);
        dst[rev_row * width + col] = e[i];
    }
}

// DIF R8 + fused bit-reversal: OUT-OF-PLACE.
// Same as bb_dif_r8 but reads from `src` and writes to `dst` at
// bit-reversed row addresses.  This is the LAST dispatch in the DIF
// pipeline, producing natural-order output without a separate bitrev pass.
kernel void bb_dif_r8_bitrev(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    constant uint& log_n           [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 3;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 3);
    uint block_r8 = h << 3;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r8 + j;

    Bb e0 = src[(base          ) * width + col];
    Bb e1 = src[(base +     h  ) * width + col];
    Bb e2 = src[(base + 2 * h  ) * width + col];
    Bb e3 = src[(base + 3 * h  ) * width + col];
    Bb e4 = src[(base + 4 * h  ) * width + col];
    Bb e5 = src[(base + 5 * h  ) * width + col];
    Bb e6 = src[(base + 6 * h  ) * width + col];
    Bb e7 = src[(base + 7 * h  ) * width + col];

    uint step_s = 1u << dif_stage;
    uint quarter_n = height >> 2;

    // DIF stage dif_stage: stride = 4h
    {
        uint tw0_idx = j * step_s;
        uint tw1_idx = tw0_idx + (h * step_s);
        uint tw2_idx = tw0_idx + quarter_n;
        uint tw3_idx = tw1_idx + quarter_n;

        Bb s0 = bb_add(e0, e4); Bb d0 = bb_sub(e0, e4);
        Bb s1 = bb_add(e1, e5); Bb d1 = bb_sub(e1, e5);
        Bb s2 = bb_add(e2, e6); Bb d2 = bb_sub(e2, e6);
        Bb s3 = bb_add(e3, e7); Bb d3 = bb_sub(e3, e7);

        e0 = s0; e1 = s1; e2 = s2; e3 = s3;
        e4 = (tw0_idx == 0) ? d0 : bb_mul(twiddles[tw0_idx], d0);
        e5 = bb_mul(twiddles[tw1_idx], d1);
        e6 = bb_mul(twiddles[tw2_idx], d2);
        e7 = bb_mul(twiddles[tw3_idx], d3);
    }

    // DIF stage dif_stage+1: stride = 2h
    {
        uint step_s1 = step_s << 1;
        uint tw4_idx = j * step_s1;
        uint tw5_idx = tw4_idx + (h * step_s1);

        Bb s0 = bb_add(e0, e2); Bb d0 = bb_sub(e0, e2);
        Bb s1 = bb_add(e1, e3); Bb d1 = bb_sub(e1, e3);
        Bb s4 = bb_add(e4, e6); Bb d4 = bb_sub(e4, e6);
        Bb s5 = bb_add(e5, e7); Bb d5 = bb_sub(e5, e7);

        e0 = s0; e1 = s1; e4 = s4; e5 = s5;
        e2 = (tw4_idx == 0) ? d0 : bb_mul(twiddles[tw4_idx], d0);
        e3 = bb_mul(twiddles[tw5_idx], d1);
        e6 = (tw4_idx == 0) ? d4 : bb_mul(twiddles[tw4_idx], d4);
        e7 = bb_mul(twiddles[tw5_idx], d5);
    }

    // DIF stage dif_stage+2: stride = h
    {
        uint step_s2 = step_s << 2;
        uint tw6_idx = j * step_s2;

        Bb s0 = bb_add(e0, e1); Bb d0 = bb_sub(e0, e1);
        Bb s2 = bb_add(e2, e3); Bb d2 = bb_sub(e2, e3);
        Bb s4 = bb_add(e4, e5); Bb d4 = bb_sub(e4, e5);
        Bb s6 = bb_add(e6, e7); Bb d6 = bb_sub(e6, e7);

        e0 = s0; e2 = s2; e4 = s4; e6 = s6;
        if (tw6_idx == 0) {
            e1 = d0; e3 = d2; e5 = d4; e7 = d6;
        } else {
            Bb w = twiddles[tw6_idx];
            e1 = bb_mul(w, d0); e3 = bb_mul(w, d2);
            e5 = bb_mul(w, d4); e7 = bb_mul(w, d6);
        }
    }

    // Write to bit-reversed row addresses in dst.
    uint rows[8] = {
        base, base + h, base + 2*h, base + 3*h,
        base + 4*h, base + 5*h, base + 6*h, base + 7*h
    };
    Bb vals[8] = {e0, e1, e2, e3, e4, e5, e6, e7};
    for (uint i = 0; i < 8; i++) {
        uint rev_row = bit_reverse_n(rows[i], log_n);
        dst[rev_row * width + col] = vals[i];
    }
}

// DIF R4 + fused bit-reversal (out-of-place).
kernel void bb_dif_r4_bitrev(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    constant uint& log_n           [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint unit_id = gid.y;
    uint num_units = height >> 2;
    if (unit_id >= num_units || col >= width) return;

    uint h = height >> (dif_stage + 2);
    uint block_r4 = h << 2;
    uint group = unit_id / h;
    uint j = unit_id % h;
    uint base = group * block_r4 + j;

    Bb e0 = src[(base          ) * width + col];
    Bb e1 = src[(base +     h  ) * width + col];
    Bb e2 = src[(base + 2 * h  ) * width + col];
    Bb e3 = src[(base + 3 * h  ) * width + col];

    uint step_s = 1u << dif_stage;

    {
        uint tw0_idx = j * step_s;
        uint tw1_idx = tw0_idx + (h * step_s);

        Bb s0 = bb_add(e0, e2); Bb d0 = bb_sub(e0, e2);
        Bb s1 = bb_add(e1, e3); Bb d1 = bb_sub(e1, e3);

        e0 = s0; e1 = s1;
        e2 = (tw0_idx == 0) ? d0 : bb_mul(twiddles[tw0_idx], d0);
        e3 = bb_mul(twiddles[tw1_idx], d1);
    }

    {
        uint step_s1 = step_s << 1;
        uint tw2_idx = j * step_s1;

        Bb s0 = bb_add(e0, e1); Bb d0 = bb_sub(e0, e1);
        Bb s2 = bb_add(e2, e3); Bb d2 = bb_sub(e2, e3);

        e0 = s0; e2 = s2;
        if (tw2_idx == 0) {
            e1 = d0; e3 = d2;
        } else {
            Bb w = twiddles[tw2_idx];
            e1 = bb_mul(w, d0); e3 = bb_mul(w, d2);
        }
    }

    uint rows[4] = { base, base + h, base + 2*h, base + 3*h };
    Bb vals[4] = { e0, e1, e2, e3 };
    for (uint i = 0; i < 4; i++) {
        uint rev_row = bit_reverse_n(rows[i], log_n);
        dst[rev_row * width + col] = vals[i];
    }
}

// DIF R2 + fused bit-reversal (out-of-place).
kernel void bb_dif_r2_bitrev(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& height          [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    constant uint& dif_stage       [[buffer(5)]],
    constant uint& log_n           [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint butterfly_id = gid.y;
    uint num_butterflies = height >> 1;
    if (butterfly_id >= num_butterflies || col >= width) return;

    uint stride = height >> (dif_stage + 1);
    uint block_size = stride << 1;
    uint block_idx = butterfly_id / stride;
    uint k = butterfly_id % stride;
    uint i = block_idx * block_size + k;
    uint j_pos = i + stride;

    Bb a = src[i * width + col];
    Bb b = src[j_pos * width + col];

    Bb sum = bb_add(a, b);
    Bb diff = bb_sub(a, b);

    uint tw_idx = k * (1u << dif_stage);
    Bb result_a = sum;
    Bb result_b = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);

    uint rev_i = bit_reverse_n(i, log_n);
    uint rev_j = bit_reverse_n(j_pos, log_n);
    dst[rev_i * width + col] = result_a;
    dst[rev_j * width + col] = result_b;
}

// ── Four-step FFT: pure transpose (no twiddle) ──────────────────────
// Out-of-place.  Interprets src as (src_rows × src_cols) blocks and
// writes the transpose (src_cols × src_rows) to dst.
// 2D grid: gid.x = column, gid.y = row (0..N-1).
kernel void bb_transpose(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    constant uint& src_rows        [[buffer(2)]],
    constant uint& src_cols        [[buffer(3)]],
    constant uint& width           [[buffer(4)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint n = src_rows * src_cols;
    if (col >= width || row >= n) return;

    uint r = row / src_cols;
    uint c = row % src_cols;
    uint dst_row = c * src_rows + r;
    dst[dst_row * width + col] = src[row * width + col];
}

// ── Four-step FFT: twiddle multiply + matrix transpose ───────────────
// Out-of-place.  Reads from src in (N2 blocks × N1 rows) layout,
// applies cross twiddle omega_N^{block * pos}, and writes to dst in
// (N1 blocks × N2 rows) layout.
// 2D grid: gid.x = column, gid.y = row (0..N-1).
kernel void bb_ntt_twiddle_transpose(
    device const Bb* src           [[buffer(0)]],
    device Bb* dst                 [[buffer(1)]],
    device const Bb* twiddles      [[buffer(2)]],
    constant uint& n1              [[buffer(3)]],
    constant uint& n2              [[buffer(4)]],
    constant uint& width           [[buffer(5)]],
    constant uint& half_n          [[buffer(6)]],
    uint2 gid                      [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    uint n = n1 * n2;
    if (col >= width || row >= n) return;

    uint blk = row / n1;   // n2-index (0..N2-1)
    uint pos = row % n1;   // k1-index (0..N1-1)

    Bb val = src[row * width + col];

    uint tw_raw = blk * pos;
    if (tw_raw != 0) {
        bool neg = tw_raw >= half_n;
        Bb tw = twiddles[neg ? tw_raw - half_n : tw_raw];
        if (neg) tw = bb_neg(tw);
        val = bb_mul(val, tw);
    }

    uint dst_row = pos * n2 + blk;
    dst[dst_row * width + col] = val;
}

// ── Shared-memory DIF with fused bit-reversal ─────────────────────────
// Processes the last `log_block` DIF stages entirely in threadgroup shared
// memory, then writes to `dst` at the full bit-reversed row index.
//
// Dispatch: one threadgroup per (column, block).
//   gpos.x = column,  gpos.y = block index
//   Threadgroup size = block_size / 2  (512 for log_block=10).
//
// Inputs:
//   src        – data with global DIF stages 0..start_stage-1 already done
//   dst        – output buffer (natural order after bit-reversal)
//   twiddles   – precomputed twiddle table
//   height     – number of rows (must be power of 2)
//   width      – number of columns
//   start_stage – first DIF stage processed here (= log_n - log_block)
//   log_n      – log2(height)
//   log_block  – number of stages in shared memory (≤ 10)

kernel void bb_dif_shared_bitrev(
    device const Bb* src       [[buffer(0)]],
    device Bb* dst             [[buffer(1)]],
    device const Bb* twiddles  [[buffer(2)]],
    constant uint& height      [[buffer(3)]],
    constant uint& width       [[buffer(4)]],
    constant uint& start_stage [[buffer(5)]],
    constant uint& log_n       [[buffer(6)]],
    constant uint& log_block   [[buffer(7)]],
    uint2 gpos                 [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]]
) {
    uint col = gpos.x;
    uint block_idx = gpos.y;
    if (col >= width) return;

    uint block_size = 1u << log_block;
    uint half_block = block_size >> 1;
    uint global_base = block_idx << log_block;

    threadgroup Bb sdata[1024];

    sdata[lid]              = src[(global_base + lid) * width + col];
    sdata[lid + half_block] = src[(global_base + lid + half_block) * width + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ls = 0; ls < log_block; ls++) {
        uint s = start_stage + ls;
        uint log_stride = (log_block - 1) - ls;
        uint local_stride = 1u << log_stride;

        uint grp = lid >> log_stride;
        uint k   = lid & (local_stride - 1);
        uint idx_a = (grp << (log_stride + 1)) + k;
        uint idx_b = idx_a + local_stride;

        Bb a = sdata[idx_a];
        Bb b = sdata[idx_b];
        Bb sum  = bb_add(a, b);
        Bb diff = bb_sub(a, b);

        uint tw_idx = k << s;
        sdata[idx_a] = sum;
        sdata[idx_b] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);

        if (ls + 1 < log_block) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint log_upper = log_n - log_block;
    uint num_blocks = height >> log_block;
    uint rev_block = bit_reverse_n(block_idx, log_upper);

    for (uint t = 0; t < 2; t++) {
        uint local_i = lid + t * half_block;
        uint rev_i   = bit_reverse_n(local_i, log_block);
        uint dst_row = rev_i * num_blocks + rev_block;
        dst[dst_row * width + col] = sdata[local_i];
    }
}

// Large shared-memory DIF + bit-reversal (4 elements per thread).
// Handles up to 4096 elements (log_block ≤ 12) with 1024 threads.
// Same interface as bb_dif_shared_bitrev but 4× larger blocks.
kernel void bb_dif_shared_bitrev_lg(
    device const Bb* src       [[buffer(0)]],
    device Bb* dst             [[buffer(1)]],
    device const Bb* twiddles  [[buffer(2)]],
    constant uint& height      [[buffer(3)]],
    constant uint& width       [[buffer(4)]],
    constant uint& start_stage [[buffer(5)]],
    constant uint& log_n       [[buffer(6)]],
    constant uint& log_block   [[buffer(7)]],
    uint2 gpos                 [[threadgroup_position_in_grid]],
    uint lid                   [[thread_index_in_threadgroup]]
) {
    uint col = gpos.x;
    uint block_idx = gpos.y;
    if (col >= width) return;

    uint block_size = 1u << log_block;
    uint quarter    = block_size >> 2;
    uint half_block = block_size >> 1;
    uint global_base = block_idx << log_block;

    threadgroup Bb sdata[4096];

    for (uint j = 0; j < 4; j++) {
        uint elem = lid + j * quarter;
        sdata[elem] = src[(global_base + elem) * width + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ls = 0; ls < log_block; ls++) {
        uint s = start_stage + ls;
        uint log_stride = (log_block - 1) - ls;
        uint local_stride = 1u << log_stride;

        for (uint b = 0; b < 2; b++) {
            uint butterfly = lid + b * quarter;
            uint grp = butterfly >> log_stride;
            uint k   = butterfly & (local_stride - 1);
            uint idx_a = (grp << (log_stride + 1)) + k;
            uint idx_b = idx_a + local_stride;

            Bb a = sdata[idx_a];
            Bb bv = sdata[idx_b];
            Bb sum  = bb_add(a, bv);
            Bb diff = bb_sub(a, bv);

            uint tw_idx = k << s;
            sdata[idx_a] = sum;
            sdata[idx_b] = (tw_idx == 0) ? diff : bb_mul(twiddles[tw_idx], diff);
        }

        if (ls + 1 < log_block) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint log_upper = log_n - log_block;
    uint num_blocks = height >> log_block;
    uint rev_block = bit_reverse_n(block_idx, log_upper);

    for (uint j = 0; j < 4; j++) {
        uint local_i = lid + j * quarter;
        uint rev_i   = bit_reverse_n(local_i, log_block);
        uint dst_row = rev_i * num_blocks + rev_block;
        dst[dst_row * width + col] = sdata[local_i];
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Poseidon2 BabyBear width-16 permutation + Merkle hashing kernels
// ═══════════════════════════════════════════════════════════════════════
//
// Constants buffer layout (all values in Montgomery form):
//   [  0.. 64) : external_initial_rc   (4 rounds × 16)
//   [ 64.. 77) : internal_rc           (13 partial-round constants)
//   [ 77..141) : external_terminal_rc  (4 rounds × 16)
//   [141]      : INV_256  = (2^8)^{-1} mod p   in Montgomery form
//   [142]      : INV_2_27 = (2^{27})^{-1} mod p in Montgomery form
//   Total: 143 Bb values

Bb bb_double(Bb a) { return bb_add(a, a); }

Bb bb_halve(Bb a) {
    uint v = a.v;
    return Bb{(v & 1u) ? ((v >> 1) + ((BB_P >> 1) + 1)) : (v >> 1)};
}

// ── Transpose + zero-pad ─────────────────────────────────────────────
// Transposes an [in_rows × in_cols] matrix of multi-word elements into
// an [out_rows × in_rows] matrix, zero-padding rows beyond in_cols.
// `elem_size` is the number of Bb words per logical element (1 for base
// field, 4 for quartic extension field).
// Each thread handles one logical element of the output.
kernel void bb_transpose_pad(
    device const Bb* src       [[buffer(0)]],
    device Bb* dst             [[buffer(1)]],
    constant uint& in_rows     [[buffer(2)]],
    constant uint& in_cols     [[buffer(3)]],
    constant uint& out_rows    [[buffer(4)]],
    constant uint& elem_size   [[buffer(5)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint out_col = gid.x;
    uint out_row = gid.y;
    if (out_col >= in_rows || out_row >= out_rows) return;

    uint dst_base = (out_row * in_rows + out_col) * elem_size;

    if (out_row < in_cols) {
        uint src_base = (out_col * in_cols + out_row) * elem_size;
        for (uint d = 0; d < elem_size; d++) {
            dst[dst_base + d] = src[src_base + d];
        }
    } else {
        for (uint d = 0; d < elem_size; d++) {
            dst[dst_base + d] = Bb{0};
        }
    }
}

Bb bb_sbox(Bb x) {
    Bb x2 = bb_mul(x, x);
    Bb x3 = bb_mul(x2, x);
    Bb x4 = bb_mul(x2, x2);
    return bb_mul(x4, x3);     // x^7
}

// 4×4 MDS matrix: [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
void apply_mat4(thread Bb* x) {
    Bb t01   = bb_add(x[0], x[1]);
    Bb t23   = bb_add(x[2], x[3]);
    Bb t0123 = bb_add(t01, t23);
    Bb t01123 = bb_add(t0123, x[1]);
    Bb t01233 = bb_add(t0123, x[3]);
    Bb new3 = bb_add(t01233, bb_double(x[0]));
    Bb new1 = bb_add(t01123, bb_double(x[2]));
    x[0] = bb_add(t01123, t01);
    x[2] = bb_add(t01233, t23);
    x[1] = new1;
    x[3] = new3;
}

// External linear layer for width 16:
// 1. apply_mat4 to each 4-element chunk
// 2. sums[k] = sum of state[k], state[k+4], state[k+8], state[k+12]
// 3. state[i] += sums[i % 4]
void mds_light_16(thread Bb* s) {
    apply_mat4(s);
    apply_mat4(s + 4);
    apply_mat4(s + 8);
    apply_mat4(s + 12);

    Bb sums[4];
    for (uint k = 0; k < 4; k++) {
        sums[k] = bb_add(bb_add(s[k], s[k + 4]), bb_add(s[k + 8], s[k + 12]));
    }
    for (uint i = 0; i < 16; i++) {
        s[i] = bb_add(s[i], sums[i & 3]);
    }
}

// Internal linear layer for BabyBear width-16.
// state[0] is already updated by the caller.
// full_sum = part_sum + state[0] (passed in).
void internal_layer_16(thread Bb* s, Bb full_sum, Bb inv256, Bb inv2_27) {
    s[1]  = bb_add(s[1], full_sum);
    s[2]  = bb_add(bb_double(s[2]), full_sum);
    s[3]  = bb_add(bb_halve(s[3]), full_sum);
    s[4]  = bb_add(full_sum, bb_add(bb_double(s[4]), s[4]));
    s[5]  = bb_add(full_sum, bb_double(bb_double(s[5])));
    s[6]  = bb_sub(full_sum, bb_halve(s[6]));
    s[7]  = bb_sub(full_sum, bb_add(bb_double(s[7]), s[7]));
    s[8]  = bb_sub(full_sum, bb_double(bb_double(s[8])));
    s[9]  = bb_add(bb_mul(s[9], inv256), full_sum);
    s[10] = bb_add(bb_halve(bb_halve(s[10])), full_sum);
    s[11] = bb_add(bb_halve(bb_halve(bb_halve(s[11]))), full_sum);
    s[12] = bb_add(bb_mul(s[12], inv2_27), full_sum);
    s[13] = bb_sub(full_sum, bb_mul(s[13], inv256));
    s[14] = bb_sub(full_sum, bb_halve(bb_halve(bb_halve(bb_halve(s[14])))));
    s[15] = bb_sub(full_sum, bb_mul(s[15], inv2_27));
}

// Full Poseidon2 permutation: width=16, R_F=8 (4+4), R_P=13, S-box=x^7.
void poseidon2_permute_16(thread Bb* state, constant Bb* rc) {
    Bb inv256  = rc[141];
    Bb inv2_27 = rc[142];

    // Initial external MDS
    mds_light_16(state);

    // 4 initial full rounds
    constant Bb* ext_init = rc;
    for (uint r = 0; r < 4; r++) {
        for (uint i = 0; i < 16; i++) {
            state[i] = bb_sbox(bb_add(state[i], ext_init[r * 16 + i]));
        }
        mds_light_16(state);
    }

    // 13 partial (internal) rounds
    constant Bb* int_rc = rc + 64;
    for (uint r = 0; r < 13; r++) {
        state[0] = bb_sbox(bb_add(state[0], int_rc[r]));

        Bb part_sum = state[1];
        for (uint i = 2; i < 16; i++) part_sum = bb_add(part_sum, state[i]);
        Bb full_sum = bb_add(part_sum, state[0]);
        state[0] = bb_sub(part_sum, state[0]);

        internal_layer_16(state, full_sum, inv256, inv2_27);
    }

    // 4 terminal full rounds
    constant Bb* ext_term = rc + 77;
    for (uint r = 0; r < 4; r++) {
        for (uint i = 0; i < 16; i++) {
            state[i] = bb_sbox(bb_add(state[i], ext_term[r * 16 + i]));
        }
        mds_light_16(state);
    }
}

// ── Leaf hashing: PaddingFreeSponge<Perm, 16, 8, 8> ──────────────────
// Each thread hashes one row (leaf) of the DFT output matrix.
// Absorbs 8 elements at a time (overwrite mode), permutes, squeezes 8.
kernel void poseidon2_hash_leaves(
    device const Bb* data        [[buffer(0)]],
    device Bb* digests           [[buffer(1)]],
    constant Bb* rc              [[buffer(2)]],
    constant uint& num_leaves    [[buffer(3)]],
    constant uint& leaf_width    [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= num_leaves) return;

    Bb state[16] = {};
    uint row_start = gid * leaf_width;

    for (uint absorbed = 0; absorbed < leaf_width; absorbed += 8) {
        uint remaining = min(8u, leaf_width - absorbed);
        for (uint i = 0; i < remaining; i++) {
            state[i] = data[row_start + absorbed + i];
        }
        if (remaining < 8) {
            for (uint i = remaining; i < 8; i++) state[i] = Bb{0};
        }
        poseidon2_permute_16(state, rc);
    }

    uint out_start = gid * 8;
    for (uint i = 0; i < 8; i++) {
        digests[out_start + i] = state[i];
    }
}

// ── Merkle compression: TruncatedPermutation<Perm, 2, 8, 16> ─────────
// Each thread compresses two 8-element child digests into one parent.
kernel void poseidon2_merkle_compress(
    device const Bb* children    [[buffer(0)]],
    device Bb* parents           [[buffer(1)]],
    constant Bb* rc              [[buffer(2)]],
    constant uint& num_pairs     [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= num_pairs) return;

    Bb state[16];
    uint left_start  = (2 * gid) * 8;
    uint right_start = (2 * gid + 1) * 8;

    for (uint i = 0; i < 8; i++) state[i]     = children[left_start + i];
    for (uint i = 0; i < 8; i++) state[8 + i]  = children[right_start + i];

    poseidon2_permute_16(state, rc);

    uint out_start = gid * 8;
    for (uint i = 0; i < 8; i++) parents[out_start + i] = state[i];
}

// ── Fused leaf hash + first compress ──────────────────────────────────
// Each thread hashes two adjacent leaves, writes both leaf digests
// to the leaf buffer (needed for query answering), AND compresses
// the pair directly, writing the parent to the compress buffer.
// Eliminates the global-memory read of the leaf buffer for the first
// compression level.
// Thread gid handles leaves 2*gid and 2*gid+1.
kernel void poseidon2_hash_and_compress(
    device const Bb* data        [[buffer(0)]],
    device Bb* leaf_digests      [[buffer(1)]],
    device Bb* parents           [[buffer(2)]],
    constant Bb* rc              [[buffer(3)]],
    constant uint& num_pairs     [[buffer(4)]],
    constant uint& leaf_width    [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= num_pairs) return;

    // Hash left leaf (2*gid)
    Bb left[16] = {};
    uint left_start = (2 * gid) * leaf_width;
    for (uint absorbed = 0; absorbed < leaf_width; absorbed += 8) {
        uint remaining = min(8u, leaf_width - absorbed);
        for (uint i = 0; i < remaining; i++) {
            left[i] = data[left_start + absorbed + i];
        }
        if (remaining < 8) {
            for (uint i = remaining; i < 8; i++) left[i] = Bb{0};
        }
        poseidon2_permute_16(left, rc);
    }
    // Write left leaf digest
    uint left_out = (2 * gid) * 8;
    for (uint i = 0; i < 8; i++) leaf_digests[left_out + i] = left[i];

    // Hash right leaf (2*gid + 1)
    Bb right[16] = {};
    uint right_start = (2 * gid + 1) * leaf_width;
    for (uint absorbed = 0; absorbed < leaf_width; absorbed += 8) {
        uint remaining = min(8u, leaf_width - absorbed);
        for (uint i = 0; i < remaining; i++) {
            right[i] = data[right_start + absorbed + i];
        }
        if (remaining < 8) {
            for (uint i = remaining; i < 8; i++) right[i] = Bb{0};
        }
        poseidon2_permute_16(right, rc);
    }
    // Write right leaf digest
    uint right_out = (2 * gid + 1) * 8;
    for (uint i = 0; i < 8; i++) leaf_digests[right_out + i] = right[i];

    // Compress: state = [left_digest || right_digest], permute, truncate
    Bb state[16];
    for (uint i = 0; i < 8; i++) state[i]     = left[i];
    for (uint i = 0; i < 8; i++) state[8 + i]  = right[i];
    poseidon2_permute_16(state, rc);

    uint out_start = gid * 8;
    for (uint i = 0; i < 8; i++) parents[out_start + i] = state[i];
}

// ── GPU Proof-of-Work (PoW) grinding ──────────────────────────────────
// Parallel brute-force search for a valid PoW witness.
// Each thread tries one candidate nonce, applies Poseidon2 permutation,
// and checks whether the output satisfies the PoW condition.

Bb bb_from_canonical(uint x, Bb r_squared) {
    return bb_mul(Bb{x}, r_squared);
}

uint bb_to_canonical(Bb x) {
    uint t = x.v * BB_MONTY_MU;
    uint u_hi = mulhi(t, BB_P);
    return u_hi == 0u ? 0u : BB_P - u_hi;
}

kernel void poseidon2_pow_grind(
    constant uint*        base_state    [[buffer(0)]],
    constant Bb*          rc            [[buffer(1)]],
    constant uint&        witness_idx   [[buffer(2)]],
    constant uint&        pow_bits      [[buffer(3)]],
    constant uint&        r_squared     [[buffer(4)]],
    device atomic_uint*   result        [[buffer(5)]],
    device atomic_uint*   found         [[buffer(6)]],
    constant uint&        nonce_offset  [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(found, memory_order_relaxed)) return;

    uint nonce = gid + nonce_offset;
    if (nonce >= BB_P) return;

    Bb state[16];
    for (uint i = 0; i < 16; i++) state[i] = Bb{base_state[i]};

    Bb r2 = Bb{r_squared};
    state[witness_idx] = bb_from_canonical(nonce, r2);

    poseidon2_permute_16(state, rc);

    uint canonical = bb_to_canonical(state[7]);
    uint mask = (1u << pow_bits) - 1u;

    if ((canonical & mask) == 0u) {
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                found, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(result, nonce, memory_order_relaxed);
        }
    }
}
