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
// One thread per ROW. Each thread processes ALL columns (loops over width).
// Shared memory layout: row-major sdata[row * width + col].
// Twiddle factor depends only on row position, so it is loaded once and
// reused across all columns — the GPU analogue of Plonky3's SIMD packing.
//
// block_size = threads_per_threadgroup (= 1 << log_block).
// Shared memory usage: block_size * width * 4 bytes ≤ 32 KB.
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
    // 32 KB = 8192 uint32s — enough for block_size * width elements.
    threadgroup Bb sdata[8192];

    uint base_row = tgid * tg_size;
    uint global_row = base_row + tid;

    // Load all columns for this thread's row into shared memory.
    uint sm_base = tid * width;
    uint gm_base = global_row * width;
    for (uint c = 0; c < width; c++) {
        sdata[sm_base + c] = data[gm_base + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = 0; stage < log_block; stage++) {
        uint stride = 1u << stage;

        if ((tid & stride) == 0) {
            uint partner = tid | stride;
            uint local_idx = tid & (stride - 1);
            uint tw_idx = local_idx * (height >> (stage + 1));

            uint a_base = tid * width;
            uint b_base = partner * width;

            if (tw_idx == 0) {
                for (uint c = 0; c < width; c++) {
                    Bb a = sdata[a_base + c];
                    Bb b = sdata[b_base + c];
                    sdata[a_base + c] = bb_add(a, b);
                    sdata[b_base + c] = bb_sub(a, b);
                }
            } else {
                Bb w = twiddles[tw_idx];
                for (uint c = 0; c < width; c++) {
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

    // Store all columns back to global memory.
    for (uint c = 0; c < width; c++) {
        data[gm_base + c] = sdata[sm_base + c];
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
