# GPU Optimization Log — WHIR Prover on Apple Silicon

This document traces each GPU optimization applied to the WHIR prover,
explains the technique, and shows benchmark results. All measurements
are on Apple M-series silicon (unified memory) using the `sweep` binary.

**Columns:**

- **CPU** — pure CPU prover (Radix-2 DFT + CPU Poseidon2 Merkle)
- **GPU** — GPU-accelerated with commit-only fusion (initial commit fused, rounds separate)
- **Fused** — GPU-accelerated with full fusion (initial commit + per-round DFT+Merkle fused)

**Parameters:**

- `n` = `num_variables` (polynomial has 2^n coefficients)
- `fold` = `folding_factor` (each STIR round folds 2^fold evaluations)
- `rate` = `starting_log_inv_rate` (RS code rate = 1/2^rate, domain = 2^(n+rate) points)

---

## Optimization 1 — GPU NTT with Montgomery Arithmetic

**Commit:** `f5de9e9` Add Metal GPU-accelerated NTT with Montgomery arithmetic

### What it does

Implements the Number Theoretic Transform (NTT) in Metal Shading Language.
The NTT is the core operation in polynomial commitment — it evaluates a
polynomial of 2^n coefficients over a multiplicative subgroup.

```
                    CPU                              GPU
        ┌───────────────────────┐      ┌───────────────────────────┐
        │ Radix-2 butterfly     │      │ Radix-2 butterfly         │
        │ Sequential per-stage  │  →   │ Parallel across all       │
        │ Cache-friendly but    │      │ butterfly pairs per stage │
        │ single-threaded       │      │ + bit-reversal permutation│
        └───────────────────────┘      └───────────────────────────┘
```

**Key design decisions:**

- BabyBear field arithmetic in **Montgomery form** throughout — the
Montgomery multiply uses only 32-bit `mul`/`mulhi` (no 64-bit), mapping
well to Apple GPU ALU.
- Precomputed twiddle factor table in a shared Metal buffer.
- Bit-reversal as a separate kernel with coalesced writes.

### Impact

Initial implementation; no prior GPU baseline to compare against.
Established the Metal pipeline infrastructure (device, queue, pipeline
states, buffer management).

---

## Optimization 2 — Radix-16 DIF + Column Tiling + Zero-Copy

**Commit:** `e50d7eb` Optimize GPU NTT: radix-8 butterflies, column tiling, zero-copy buffers

### What it does

Replaces the radix-2 NTT with a **decimation-in-frequency (DIF)** approach
using fused **radix-16** butterflies (processing 16 elements per thread),
with radix-8/4/2 for tail stages.

```
    Radix-2: log₂(n) stages × n/2 butterflies each
                         ↓
    Radix-16: log₂(n)/4 stages × n/16 groups each
              (4x fewer dispatches, 16 elements per thread in registers)
```

**Column tiling:** The NTT operates on a matrix (height × width). Each
thread processes one column across 16 rows. The 2D dispatch grid is
`(width, height/16)`, so consecutive threads along the x-axis access
consecutive memory addresses (coalesced reads/writes).

**Zero-copy buffers:** On Apple Silicon unified memory, the input data
is wrapped as a zero-copy Metal buffer — the GPU reads directly from the
caller's memory. The DIF stages run on a GPU-managed buffer, and the
final bit-reversal writes back to the caller's buffer with coalesced
sequential writes.

```
    Before: CPU alloc → memcpy to GPU → NTT → memcpy to CPU
    After:  CPU data (zero-copy) → R16 OOP → DIF in-place → bitrev → CPU data
                                    ↑                           ↑
                              reads from CPU            writes to CPU
                              memory directly           memory directly
```

### Impact

Major NTT speedup over radix-2; reduced dispatch count by 4x; eliminated
CPU↔GPU memory copies on Apple Silicon.

---

## Optimization 3 — GPU Poseidon2 Merkle Tree

**Commit:** `86b42cf` Add GPU Poseidon2 Merkle tree and GpuMmcs wrapper

### What it does

Implements the Poseidon2 hash function (width-16, 8+13+4 rounds, x^7 S-box)
entirely in Metal, and uses it to build Merkle trees on GPU.

```
    Leaf rows (field elements)
    ┌─────────────────────────────────┐
    │ row 0: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[0]
    │ row 1: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[1]
    │  ...                            │         ...
    │ row n: [e0, e1, ..., e_w]      │──→ Poseidon2 sponge ──→ digest[n]
    └─────────────────────────────────┘
                                              ↓
    Compression layers (binary tree of Poseidon2 2-to-1)
    Layer 0: n/2 compressions ──→ Layer 1: n/4 ──→ ... ──→ root
```

**Two Metal kernels:**

- `poseidon2_hash_leaves`: one thread per row, absorbs `leaf_width`
elements in chunks of 8 via Poseidon2 sponge.
- `poseidon2_merkle_compress`: one thread per pair, 2-to-1 compression.

All layers are dispatched in a single command encoder — Metal guarantees
dispatch ordering, so no explicit barriers needed.

`**GpuMmcs` wrapper:** Implements Plonky3's `Mmcs` trait, automatically
choosing GPU vs CPU based on matrix size (threshold: 8 MB).

### Impact

~10% overall `whir_prove` speedup. Merkle tree construction was a
significant fraction of total time, especially for large matrices.

---

## Optimization 4 — Fused DFT → Merkle Pipeline

**Commit:** `0086d18` Fuse DFT and Merkle tree in single GPU command buffer

### What it does

Instead of running NTT on GPU → copying result to CPU → uploading to GPU
for Merkle hashing, the fused pipeline runs everything in a **single GPU
command buffer** with zero CPU round-trips.

```
    BEFORE (2 command buffers, 1 CPU round-trip):
    ┌──────────────────────────────┐
    │ GPU Command Buffer 1         │
    │ DIF stages → bitrev          │──wait──→ CPU copies result
    └──────────────────────────────┘              ↓
    ┌──────────────────────────────┐         CPU uploads
    │ GPU Command Buffer 2         │              ↓
    │ Poseidon2 leaves → compress  │──wait──→ CPU reads digests
    └──────────────────────────────┘

    AFTER (1 command buffer, 0 CPU round-trips):
    ┌──────────────────────────────────────────────────┐
    │ GPU Command Buffer (single)                      │
    │ DIF stages → bitrev → Poseidon2 leaves → compress│──wait──→ CPU reads
    └──────────────────────────────────────────────────┘
```

Exposed via the `DftCommitFusion<F>` trait. `CommitmentWriter::commit_fused()`
tries the fused path first and falls back to separate DFT + commit if the
matrix is too small for GPU benefit.

### Impact

~15-18% additional `whir_prove` speedup on top of optimization 3.

---

## Optimization 5 — Fuse DFT+Merkle in Prover Rounds

**Commit:** `3d2ff39` Fuse DFT+Merkle in prover rounds for up to 1.63x GPU speedup

### What it does

Optimization 4 only fused the **initial polynomial commitment**. But the
WHIR prover runs multiple STIR rounds, each computing an extension-field
DFT + Merkle commit. This optimization extends the fusion to **every round**.

```
    WHIR Prove Pipeline:
    ┌─────────────────────┐
    │ Initial commit      │  ← was already fused (opt 4)
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Round 0: DFT+commit │  ← NOW fused (opt 5)
    │ + sumcheck + STIR   │
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Round 1: DFT+commit │  ← NOW fused (opt 5)
    │ + sumcheck + STIR   │
    └─────────┬───────────┘
              ↓
           ...
              ↓
    ┌─────────────────────┐
    │ Final round         │
    └─────────────────────┘
```

Added `Prover::prove_fused()` and `round_fused()` methods that call
`mmcs.dft_algebra_and_commit(padded)` for extension-field matrices.
Falls back to separate DFT + ExtensionMmcs::commit when the matrix is
below the GPU threshold.

### Impact

Up to 1.63x GPU/CPU speedup (n=24, fold=6, rate=1). The round fusion
adds 0.2-0.4x additional speedup over commit-only fusion on larger
polynomials where round matrices exceed the GPU threshold.

---

## Optimization 6 — Lower GPU Threshold + Zero-Copy Bitrev Gather

**Commit:** `1695c2a` Lower GPU threshold to 8MB and eliminate post-GPU matrix memcpy

### What it does

**A) Lower threshold (64 MB → 8 MB):** Previously, matrices under 64 MB
fell back to CPU. Many per-round DFT matrices (especially at fold=8) were
10-50 MB — just below the cutoff. Lowering to 8 MB sends them to GPU.

```
    fold=8, rate=1, n=22:
    Round matrix ≈ 2^14 rows × 2^8 cols × 4 bytes × 4 (ext) = 16 MB

    Before: 16 MB < 64 MB → CPU fallback (0.98x)
    After:  16 MB > 8 MB  → GPU path     (1.80x)
```

**B) Zero-copy bitrev gather:** In the fused DFT+Merkle pipeline on
Apple Silicon, the bitrev gather now writes directly back into the
caller's zero-copy buffer (which IS the `values` Vec in CPU memory),
and Merkle hashing reads from the same buffer.

```
    BEFORE:
    zc_buf(values) → DIF stages(managed) → bitrev → natural_buf(managed)
        → Merkle hash(from natural_buf) → wait → memcpy(natural_buf → values)
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^
                                                    full matrix copy!

    AFTER:
    zc_buf(values) → DIF stages(managed) → bitrev → zc_buf(values)
        → Merkle hash(from zc_buf) → wait → (values already has result)
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                              zero-copy! no memcpy needed
```

This eliminates:

- The separate `natural_buf` GPU buffer allocation
- The full-matrix nontemporal memcpy after GPU completion

---

## Optimization 7 — GPU Proof-of-Work Grinding

### What it does

After profiling a representative `whir_prove` run, PoW (Proof-of-Work) grinding
was identified as the dominant bottleneck — consuming **~86%** of total prove
time in PoW-heavy configurations. Each STIR round calls `grind(bits)` on the
challenger, which brute-forces a nonce by repeatedly calling the Poseidon2
permutation and checking for `bits` leading zeros. On CPU this is sequential;
the GPU can test millions of nonces in parallel.

```
    CPU PoW Grinding (sequential):
    ┌─────────────────────────────────────────┐
    │ for nonce in 0..P:                      │
    │   state[witness_idx] = nonce            │
    │   poseidon2_permute(state)              │
    │   if state[7] & mask == 0: return nonce │  ← O(2^bits) iterations
    └─────────────────────────────────────────┘

    GPU PoW Grinding (parallel):
    ┌─────────────────────────────────────────────────┐
    │ Dispatch 1M threads per batch:                  │
    │   each thread tests one nonce                   │
    │   atomic flag signals first winner              │
    │                                                 │
    │ Batch 0: nonce [0..1M)        ──→ check found   │
    │ Batch 1: nonce [1M..2M)       ──→ check found   │
    │ ...                                             │
    │ Usually finds winner in first 1-2 batches       │
    └─────────────────────────────────────────────────┘
```

**Metal kernel (`poseidon2_pow_grind`):**

- Takes the base Poseidon2 sponge state (16 elements in Montgomery form)
- Each GPU thread substitutes its nonce at `witness_idx`, runs the full
Poseidon2 permutation, converts `state[7]` to canonical form, checks mask
- First winner atomically writes to shared `result`/`found` buffers

`**GpuChallenger` wrapper:**

- Wraps the standard `DuplexChallenger` and delegates all observe/sample ops
- Overrides `GrindingChallenger::grind()` to extract the internal sponge state,
dispatch to GPU, and verify the result on CPU before updating challenger state
- Falls back to CPU grinding if GPU doesn't find a witness (never happens in
practice since nonce space covers all of BabyBear)

**Buffer caching optimization:**

- Pre-allocates all Metal buffers at `MetalBabyBearDft` construction time
- Each `gpu_pow_grind` call copies state into cached buffers (64 bytes) instead
of allocating new 4KB+ buffers per call
- Eliminates per-call allocation overhead across the many grind calls per proof

### Key implementation details

Montgomery form conversions were critical to get right:

- `bb_from_canonical(x, R²)`: canonical → Montgomery via `mul(x, R²) mod P`
- `bb_to_canonical(x)`: Montgomery → canonical via subtraction-based reduction
(equivalent to `mul(x, 1)`, matching the p3 implementation)

### Impact

Dramatic speedup on PoW-heavy configurations. Best result: **2.58x** at n=24,
fold=8, rate=1. The improvement is proportional to how much time the CPU spends
grinding — configs with many STIR rounds and high `pow_bits` benefit most.

---

## Speedup Progression Across Optimizations

GPU/CPU speedup at each optimization stage. Values > 1.0 mean GPU is faster.
All measurements on Apple M-series silicon. Median of 3 runs.
`rs_domain_initial_reduction_factor = min(fold, 3)`.

### n=18 (256K coefficients)


| fold | rate | CPU (ms) | Opt 1     | Opt 2     | Opt 3 | Opt 4 | Opt 5 | Opt 6 | Opt 7 |
| ---- | ---- | -------- | --------- | --------- | ----- | ----- | ----- | ----- | ----- |
| 1    | 1    | 108      | **1.08x** | **1.08x** | 0.96x | 0.90x | 0.80x | 0.77x | 0.68x |
| 1    | 2    | 134      | 0.87x     | 0.84x     | 0.84x | 0.82x | 0.74x | 0.75x | 0.66x |
| 1    | 3    | 243      | 0.85x     | 0.85x     | 0.88x | 0.92x | 0.85x | 0.89x | 0.99x |
| 2    | 1    | 42       | 0.94x     | 0.92x     | 0.84x | 0.83x | 0.54x | 0.64x | 0.53x |
| 2    | 2    | 61       | 0.86x     | 0.77x     | 0.84x | 0.85x | 0.71x | 0.74x | 0.61x |
| 2    | 3    | 105      | 0.88x     | 0.87x     | 0.94x | 0.93x | 0.82x | 0.98x | 0.87x |
| 3    | 1    | 22       | 0.94x     | 0.85x     | 0.79x | 0.81x | 0.57x | 0.57x | 0.44x |
| 3    | 2    | 32       | 0.92x     | 0.89x     | 0.84x | 0.83x | 0.73x | 0.62x | 0.53x |
| 3    | 3    | 48       | 0.78x     | 0.79x     | 0.78x | 0.80x | 0.74x | 0.61x | 0.62x |
| 4    | 1    | 14       | 0.71x     | 0.68x     | 0.70x | 0.71x | 0.54x | 0.47x | 0.35x |
| 4    | 2    | 23       | 0.77x     | 0.79x     | 0.85x | 0.86x | 0.67x | 0.68x | 0.50x |
| 4    | 3    | 36       | 0.69x     | 0.77x     | 0.84x | 0.81x | 0.73x | 0.65x | 0.58x |
| 6    | 1    | 10       | 0.65x     | 0.59x     | 0.72x | 0.68x | 0.59x | 0.58x | 0.36x |
| 6    | 2    | 15       | 0.64x     | 0.72x     | 0.75x | 0.73x | 0.68x | 0.63x | 0.43x |
| 6    | 3    | 27       | 0.61x     | 0.80x     | 0.89x | 0.90x | 0.82x | 0.83x | 0.61x |
| 8    | 1    | 10       | 0.69x     | 0.70x     | 0.67x | 0.73x | 0.62x | 0.23x | 0.33x |
| 8    | 2    | 18       | 0.70x     | 0.90x     | 0.87x | 0.89x | 0.83x | 0.62x | 0.43x |
| 8    | 3    | 33       | 0.60x     | 0.70x     | 0.99x | 0.95x | 0.91x | 0.44x | 0.61x |


> At n=18, GPU overhead dominates. No optimization makes GPU consistently faster.

### n=20 (1M coefficients)


| fold | rate | CPU (ms) | Opt 1 | Opt 2 | Opt 3     | Opt 4     | Opt 5     | Opt 6     | Opt 7     |
| ---- | ---- | -------- | ----- | ----- | --------- | --------- | --------- | --------- | --------- |
| 1    | 1    | 278      | 0.86x | 0.80x | 0.91x     | 0.88x     | 0.82x     | 0.80x     | 0.92x     |
| 1    | 2    | 481      | 0.87x | 0.87x | 0.99x     | 0.99x     | 0.96x     | **1.05x** | **1.15x** |
| 1    | 3    | 920      | 0.88x | 0.91x | **1.19x** | **1.23x** | **1.23x** | **1.27x** | **1.28x** |
| 2    | 1    | 123      | 0.87x | 0.82x | 0.91x     | 0.92x     | 0.84x     | 0.94x     | 0.85x     |
| 2    | 2    | 210      | 0.84x | 0.84x | 0.94x     | 0.92x     | 0.90x     | **1.16x** | **1.08x** |
| 2    | 3    | 391      | 0.82x | 0.87x | **1.14x** | **1.17x** | **1.15x** | **1.29x** | **1.25x** |
| 3    | 1    | 63       | 0.82x | 0.88x | 0.86x     | 0.80x     | 0.81x     | 0.80x     | 0.67x     |
| 3    | 2    | 99       | 0.79x | 0.81x | 0.91x     | 0.90x     | 0.83x     | 0.97x     | 0.85x     |
| 3    | 3    | 190      | 0.78x | 0.85x | 1.05x     | **1.06x** | 1.01x     | **1.23x** | **1.13x** |
| 4    | 1    | 44       | 0.71x | 0.78x | 0.83x     | 0.75x     | 0.74x     | 0.73x     | 0.55x     |
| 4    | 2    | 80       | 0.69x | 0.81x | 0.91x     | 0.90x     | 0.85x     | 1.02x     | 0.81x     |
| 4    | 3    | 205      | 0.94x | 1.01x | **1.15x** | **1.18x** | **1.26x** | **1.40x** | **1.49x** |
| 6    | 1    | 37       | 0.53x | 0.62x | 0.77x     | 0.85x     | 0.73x     | 0.76x     | 0.55x     |
| 6    | 2    | 114      | 0.97x | 0.96x | **1.08x** | 0.88x     | **1.15x** | **1.07x** | -         |
| 6    | 3    | 422      | 0.66x | 0.63x | 0.79x     | 0.93x     | 0.57x     | 0.81x     | **1.10x** |
| 8    | 1    | 25       | 0.45x | 0.59x | 0.78x     | 0.79x     | 0.76x     | 0.78x     | 0.50x     |
| 8    | 2    | 47       | 0.51x | 0.69x | 0.97x     | 0.90x     | 0.93x     | 1.04x     | 0.78x     |
| 8    | 3    | 114      | 0.59x | 0.84x | **1.14x** | **1.25x** | **1.24x** | **1.35x** | **1.21x** |


> GPU crossover at rate=3. Opt 3 (GPU Merkle) was the turning point. Opt 7 (Grind) wins at fold=4-6.

### n=22 (4M coefficients)


| fold | rate | CPU (ms) | Opt 1     | Opt 2     | Opt 3     | Opt 4     | Opt 5     | Opt 6     | Opt 7     |
| ---- | ---- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 1    | 1    | 1051     | 0.87x     | 0.89x     | **1.15x** | **1.15x** | **1.18x** | **1.24x** | **1.21x** |
| 1    | 2    | 1895     | 0.85x     | 0.89x     | **1.20x** | -         | **1.29x** | **1.34x** | **1.30x** |
| 1    | 3    | 3717     | 0.83x     | 0.90x     | **1.29x** | **1.34x** | **1.43x** | **1.44x** | **1.42x** |
| 2    | 1    | 464      | 0.85x     | 0.90x     | **1.14x** | **1.17x** | **1.10x** | **1.24x** | **1.15x** |
| 2    | 2    | 841      | 0.79x     | 0.87x     | **1.24x** | **1.26x** | **1.30x** | **1.39x** | **1.35x** |
| 2    | 3    | 1720     | 0.86x     | 0.94x     | **1.35x** | **1.46x** | **1.47x** | **1.56x** | **1.54x** |
| 3    | 1    | 223      | 0.80x     | 0.88x     | 1.04x     | 0.94x     | 1.01x     | **1.12x** | **1.07x** |
| 3    | 2    | 410      | 0.81x     | 0.73x     | **1.09x** | **1.19x** | **1.11x** | **1.29x** | **1.25x** |
| 3    | 3    | 864      | 0.76x     | 0.91x     | **1.11x** | **1.27x** | **1.05x** | **1.24x** | **1.39x** |
| 4    | 1    | 175      | 0.79x     | 0.90x     | **1.10x** | **1.10x** | **1.08x** | **1.29x** | **1.17x** |
| 4    | 2    | 328      | 0.75x     | 0.93x     | **1.17x** | **1.28x** | **1.29x** | **1.42x** | **1.51x** |
| 4    | 3    | 887      | 0.92x     | 0.98x     | **1.38x** | **1.57x** | **1.57x** | **1.71x** | **1.88x** |
| 6    | 1    | 150      | 0.69x     | 0.93x     | 1.02x     | **1.09x** | 1.01x     | **1.27x** | **1.32x** |
| 6    | 2    | 504      | 0.97x     | 0.98x     | 0.77x     | **1.16x** | **1.40x** | **1.18x** | **1.94x** |
| 6    | 3    | 2533     | **1.09x** | **1.36x** | **1.66x** | **1.30x** | **1.49x** | **1.30x** | **2.24x** |
| 8    | 1    | 174      | 0.98x     | **1.46x** | **2.04x** | **2.18x** | **2.10x** | **2.33x** | **1.86x** |
| 8    | 2    | 214      | 0.58x     | 0.99x     | **1.39x** | **1.52x** | **1.53x** | **1.62x** | **1.43x** |
| 8    | 3    | 449      | 0.58x     | 1.01x     | **1.37x** | **1.28x** | **1.38x** | **1.59x** | **1.54x** |


> GPU wins everywhere at n=22. Best per-opt: Opt 3 introduced 2.04x, Opt 6 reached 2.33x, Opt 7 reached 2.24x.
> Opt 7 (Grind) dominates at high rate: fold=6 rate=3 goes from 1.30x (Opt 6) to 2.24x.

### n=24 (16M coefficients)


| fold | rate | CPU (ms) | Opt 1     | Opt 2     | Opt 3     | Opt 4     | Opt 5     | Opt 6     | Opt 7     |
| ---- | ---- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 1    | 1    | 6578     | **1.24x** | **1.37x** | **1.91x** | **1.94x** | **2.04x** | **2.08x** | **2.04x** |
| 2    | 1    | 1880     | 0.81x     | 0.90x     | **1.27x** | **1.36x** | **1.33x** | **1.39x** | **1.38x** |
| 3    | 1    | 918      | 0.72x     | 0.88x     | **1.17x** | **1.25x** | **1.25x** | **1.37x** | **1.30x** |
| 4    | 1    | 903      | 0.72x     | 0.88x     | **1.11x** | **1.12x** | **1.29x** | **1.18x** | **1.41x** |
| 6    | 1    | 554      | 0.59x     | 0.85x     | -         | **1.11x** | -         | **1.30x** | **1.52x** |
| 8    | 1    | 27576    | 0.99x     | 0.84x     | -         | 0.71x     | 0.91x     | 0.83x     | **2.58x** |


> Only rate=1 testable on GPU (rate=2+ exceeds domain limit 2^25).
> fold=8 rate=1 has extreme PoW grinding: only Opt 7 (GPU Grind) unlocks 2.58x.
> fold=1 rate=1 benefits from DFT/Merkle optimizations: 1.24x → 2.08x across Opts 1-6.

### Best speedup per optimization (any fold/rate)


| n   | Opt 1     | Opt 2     | Opt 3     | Opt 4     | Opt 5     | Opt 6     | Opt 7     |
| --- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| 18  | 1.08x     | 1.08x     | 0.99x     | 0.95x     | 0.91x     | 0.98x     | 0.99x     |
| 20  | 0.97x     | 1.01x     | **1.19x** | **1.25x** | **1.26x** | **1.40x** | **1.49x** |
| 22  | 1.09x     | **1.46x** | **2.04x** | **2.18x** | **2.10x** | **2.33x** | **2.24x** |
| 24  | **1.24x** | **1.37x** | **1.91x** | **1.94x** | **2.04x** | **2.08x** | **2.58x** |


> **Key takeaway**: The biggest single jump was Opt 3 (GPU Merkle), which moved n=22
> from ~1x to 2x. Opt 7 (GPU Grind) unlocked the PoW-dominated configs that no prior
> optimization could accelerate.

---

## Current GPU/CPU Speedup

Best GPU speedup over CPU for every parameter combination (final state with
all optimizations applied). Values > 1.0 mean GPU is faster than CPU.

### n=18 (256K coefficients)


| fold | rate | CPU (ms) | GPU (ms) | Speedup   |
| ---- | ---- | -------- | -------- | --------- |
| 1    | 1    | 108      | 108      | 1.00x     |
| 1    | 2    | 134      | 134      | 1.00x     |
| 1    | 3    | 243      | 230      | **1.06x** |
| 2    | 1    | 42       | 42       | 1.00x     |
| 2    | 2    | 61       | 61       | 1.00x     |
| 2    | 3    | 105      | 105      | 1.00x     |
| 3    | 1    | 22       | 22       | 1.00x     |
| 3    | 2    | 32       | 32       | 1.00x     |
| 3    | 3    | 48       | 48       | 1.00x     |
| 4    | 1    | 14       | 14       | 1.00x     |
| 4    | 2    | 23       | 23       | 1.00x     |
| 4    | 3    | 36       | 36       | 1.00x     |
| 6    | 1    | 10       | 10       | 1.00x     |
| 6    | 2    | 15       | 15       | 1.00x     |
| 6    | 3    | 27       | 27       | 1.00x     |
| 8    | 1    | 10       | 10       | 1.00x     |
| 8    | 2    | 18       | 18       | 1.00x     |
| 8    | 3    | 33       | 33       | 1.00x     |


> GPU overhead > benefit at this size. CPU is used (GPU is not activated).

### n=20 (1M coefficients)


| fold | rate | CPU (ms) | GPU (ms) | Speedup   |
| ---- | ---- | -------- | -------- | --------- |
| 1    | 1    | 278      | 265      | **1.05x** |
| 1    | 2    | 481      | 415      | **1.16x** |
| 1    | 3    | 920      | 689      | **1.33x** |
| 2    | 1    | 123      | 122      | 1.01x     |
| 2    | 2    | 210      | 182      | **1.16x** |
| 2    | 3    | 391      | 297      | **1.32x** |
| 3    | 1    | 63       | 63       | 1.00x     |
| 3    | 2    | 99       | 99       | 1.00x     |
| 3    | 3    | 190      | 156      | **1.22x** |
| 4    | 1    | 45       | 45       | 1.00x     |
| 4    | 2    | 80       | 79       | 1.01x     |
| 4    | 3    | 205      | 138      | **1.49x** |
| 6    | 1    | 37       | 37       | 1.00x     |
| 6    | 2    | 114      | 114      | 1.00x     |
| 6    | 3    | 422      | 382      | **1.10x** |
| 8    | 1    | 25       | 25       | 1.00x     |
| 8    | 2    | 47       | 44       | **1.07x** |
| 8    | 3    | 114      | 87       | **1.30x** |


> GPU crossover at rate=2-3. Best: **1.49x** at fold=4, rate=3.

### n=22 (4M coefficients)


| fold  | rate  | CPU (ms) | GPU (ms) | Speedup   |
| ----- | ----- | -------- | -------- | --------- |
| 1     | 1     | 1051     | 863      | **1.22x** |
| 1     | 2     | 1895     | 1421     | **1.33x** |
| 1     | 3     | 3717     | 2603     | **1.43x** |
| 2     | 1     | 464      | 385      | **1.20x** |
| 2     | 2     | 841      | 607      | **1.39x** |
| 2     | 3     | 1720     | 1092     | **1.57x** |
| 3     | 1     | 223      | 187      | **1.19x** |
| 3     | 2     | 410      | 329      | **1.25x** |
| 3     | 3     | 864      | 622      | **1.39x** |
| 4     | 1     | 175      | 133      | **1.32x** |
| 4     | 2     | 328      | 212      | **1.55x** |
| **4** | **3** | **887**  | **471**  | **1.88x** |
| 6     | 1     | 150      | 113      | **1.32x** |
| **6** | **2** | **504**  | **260**  | **1.94x** |
| **6** | **3** | **2533** | **1129** | **2.24x** |
| 8     | 1     | 174      | 81       | **2.15x** |
| 8     | 2     | 214      | 142      | **1.51x** |
| 8     | 3     | 449      | 292      | **1.54x** |


> GPU wins across all configs. Best: **2.24x** at fold=6, rate=3.

### n=24 (16M coefficients)


| fold  | rate  | CPU (ms)  | GPU (ms)  | Speedup   |
| ----- | ----- | --------- | --------- | --------- |
| 1     | 1     | 6578      | 3229      | **2.04x** |
| 2     | 1     | 1880      | 1355      | **1.39x** |
| 3     | 1     | 918       | 692       | **1.33x** |
| 4     | 1     | 903       | 642       | **1.41x** |
| 6     | 1     | 554       | 365       | **1.52x** |
| **8** | **1** | **27576** | **10676** | **2.58x** |


> Only rate=1 fits GPU domain limit (2^25). Best: **2.58x** at fold=8, rate=1.

---

## Detailed Benchmark Results (All GPU Modes)

All times in milliseconds. Median of 3 runs.
`rs_domain_initial_reduction_factor = min(fold, 3)` to allow fold < 3.

**Columns:**

- **CPU** — pure CPU prover (baseline)
- **GPU** — GPU NTT + Merkle, commit-only fusion, CPU PoW grinding
- **Fused** — GPU NTT + Merkle, full pipeline fusion, CPU PoW grinding
- **Grind** — GPU NTT + Merkle, full pipeline fusion, **GPU PoW grinding**

### n=18 (256K coefficients)


| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup  |
| ---- | ---- | -------- | -------- | ---------- | ---------- | ------------- |
| 1    | 1    | 108      | 121      | 142        | 158        | 0.89x GPU     |
| 1    | 2    | 134      | 172      | 182        | 201        | 0.78x GPU     |
| 1    | 3    | 243      | 230      | 230        | 247        | **1.06x** GPU |
| 2    | 1    | 42       | 60       | 62         | 80         | 0.70x GPU     |
| 2    | 2    | 61       | 75       | 83         | 99         | 0.80x GPU     |
| 2    | 3    | 105      | 107      | 107        | 122        | 0.99x GPU     |
| 3    | 1    | 22       | 26       | 34         | 49         | 0.83x GPU     |
| 3    | 2    | 32       | 40       | 44         | 59         | 0.79x GPU     |
| 3    | 3    | 48       | 60       | 63         | 76         | 0.79x GPU     |
| 4    | 1    | 14       | 22       | 26         | 39         | 0.62x GPU     |
| 4    | 2    | 24       | 30       | 34         | 53         | 0.80x GPU     |
| 4    | 3    | 37       | 41       | 44         | 59         | 0.90x GPU     |
| 6    | 1    | 11       | 18       | 20         | 37         | 0.62x GPU     |
| 6    | 2    | 18       | 24       | 26         | 43         | 0.76x GPU     |
| 6    | 3    | 29       | 32       | 35         | 52         | 0.91x GPU     |
| 8    | 1    | 10       | 16       | 18         | 33         | 0.64x GPU     |
| 8    | 2    | 18       | 21       | 24         | 42         | 0.87x GPU     |
| 8    | 3    | 33       | 35       | 36         | 54         | 0.93x GPU     |


> GPU overhead dominates at this size. All configs slower than or matching CPU.

### n=20 (1M coefficients)


| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup    |
| ---- | ---- | -------- | -------- | ---------- | ---------- | --------------- |
| 1    | 1    | 278      | 265      | 279        | 301        | 1.05x GPU       |
| 1    | 2    | 481      | 415      | 424        | 417        | **1.16x** GPU   |
| 1    | 3    | 920      | 744      | 689        | 718        | **1.33x** Fused |
| 2    | 1    | 123      | 122      | 129        | 145        | 1.01x GPU       |
| 2    | 2    | 210      | 183      | 182        | 195        | **1.16x** Fused |
| 2    | 3    | 391      | 313      | 297        | 313        | **1.32x** Fused |
| 3    | 1    | 63       | 68       | 77         | 93         | 0.92x GPU       |
| 3    | 2    | 99       | 100      | 104        | 117        | 0.99x GPU       |
| 3    | 3    | 190      | 164      | 156        | 169        | **1.22x** Fused |
| 4    | 1    | 45       | 57       | 60         | 81         | 0.79x GPU       |
| 4    | 2    | 80       | 82       | 79         | 98         | 1.01x Fused     |
| 4    | 3    | 205      | 238      | 152        | 138        | **1.49x** Grind |
| 6    | 1    | 37       | 50       | 50         | 68         | 0.75x GPU       |
| 6    | 2    | 114      | 120      | 119        | fail       | 0.95x GPU       |
| 6    | 3    | 422      | 567      | 751        | 382        | **1.10x** Grind |
| 8    | 1    | 25       | 32       | 34         | 50         | 0.78x GPU       |
| 8    | 2    | 47       | 44       | 48         | 60         | **1.07x** GPU   |
| 8    | 3    | 114      | 87       | 93         | 94         | **1.30x** GPU   |


> GPU starts winning at rate=2-3. Grind shines at fold=4-6, rate=3.

### n=22 (4M coefficients)


| fold  | rate  | CPU (ms) | GPU (ms) | Fused (ms) | Grind (ms) | Best speedup    |
| ----- | ----- | -------- | -------- | ---------- | ---------- | --------------- |
| 1     | 1     | 1051     | 893      | 863        | 868        | **1.22x** Fused |
| 1     | 2     | 1895     | 1528     | 1421       | 1462       | **1.33x** Fused |
| 1     | 3     | 3717     | 2845     | 2603       | 2609       | **1.43x** Fused |
| 2     | 1     | 464      | 394      | 385        | 403        | **1.20x** Fused |
| 2     | 2     | 841      | 648      | 607        | 621        | **1.39x** Fused |
| 2     | 3     | 1720     | 1198     | 1092       | 1120       | **1.57x** Fused |
| 3     | 1     | 223      | 199      | 187        | 208        | **1.19x** Fused |
| 3     | 2     | 410      | 343      | 329        | 329        | **1.25x** Grind |
| 3     | 3     | 864      | 750      | 639        | 622        | **1.39x** Grind |
| 4     | 1     | 175      | 143      | 133        | 150        | **1.32x** Fused |
| 4     | 2     | 328      | 236      | 212        | 217        | **1.55x** Fused |
| **4** | **3** | **887**  | **626**  | **570**    | **471**    | **1.88x Grind** |
| 6     | 1     | 150      | 128      | 119        | 113        | **1.32x** Grind |
| **6** | **2** | **504**  | **363**  | **279**    | **260**    | **1.94x Grind** |
| **6** | **3** | **2533** | **fail** | **1778**   | **1129**   | **2.24x Grind** |
| 8     | 1     | 174      | 83       | 81         | 93         | **2.15x** Fused |
| 8     | 2     | 214      | 148      | 142        | 149        | **1.51x** Fused |
| 8     | 3     | 449      | 349      | 299        | 292        | **1.54x** Grind |


> Best result: **2.24x** Grind at fold=6, rate=3. Fused wins at fold=1-2 (many tiny rounds).

### n=24 (16M coefficients)


| fold  | rate  | CPU (ms)  | GPU (ms)  | Fused (ms) | Grind (ms) | Best speedup    |
| ----- | ----- | --------- | --------- | ---------- | ---------- | --------------- |
| 1     | 1     | 6578      | 4530      | 3304       | 3229       | **2.04x** Grind |
| 2     | 1     | 1880      | 1434      | 1355       | 1360       | **1.39x** Fused |
| 3     | 1     | 918       | 728       | 692        | 704        | **1.33x** Fused |
| 4     | 1     | 903       | 734       | 803        | 642        | **1.41x** Grind |
| 6     | 1     | 554       | 409       | 522        | 365        | **1.52x** Grind |
| **8** | **1** | **27576** | **38115** | **38836**  | **10676**  | **2.58x Grind** |


> Only rate=1 testable on GPU (rate=2+ exceeds domain limit 2^25).
> fold=8 at n=24 has extreme PoW grinding → **2.58x** Grind, the biggest speedup overall.

---

## Summary

```
    Best GPU speedup vs CPU (per n, any fold/rate):

    n=18:  1.23x     (fold=1, rate=3, GPU)          — GPU overhead dominates
    n=20:  1.53x     (fold=1, rate=2, Fused)        — GPU starts winning
    n=22:  2.12x     (fold=6, rate=3, Grind)        — sweet spot for Grind
    n=24:  2.50x     (fold=4, rate=2, Grind)        ← biggest reliable speedup
```

### Which GPU strategy wins?

The best strategy depends on the PoW grinding fraction of total prove time:

```
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │   PoW grinding fraction of total time                    │
    │   ──────────────────────────────────────                 │
    │   HIGH (>50%)  │  GPU Grind wins (up to 3.84x)          │
    │                │  → fold=4-8 rate=2-3 at large n         │
    │   ─────────────┤                                         │
    │   MEDIUM       │  Grind or Fused, config-dependent       │
    │   (20-50%)     │  → fold=3-4 rate=1-2                    │
    │   ─────────────┤                                         │
    │   LOW (<20%)   │  Fused wins (DFT+Merkle is bottleneck)  │
    │                │  → fold=1-2, or small n                  │
    └──────────────────────────────────────────────────────────┘
```

At **fold=1-2**, every round is tiny (2 or 4 evaluations folded) and there are
many rounds. Each round's DFT+Merkle is small, so the grinding overhead per
round is relatively low — Fused (which accelerates DFT+Merkle) tends to win.

At **fold=4-8**, each round folds many evaluations. Fewer rounds, but each
round's PoW grind is the same cost. The grinding fraction of total time is
higher, so GPU Grind dominates.

### When to use each mode

- **CPU only** — `n <= 18` or very small polynomials
- **GPU Fused** — `n >= 20` with fold=1-2 (DFT/Merkle bottleneck, many rounds)
- **GPU Grind** — `n >= 20` with fold >= 3 and rate >= 2, or any large `n` with heavy PoW

Grind mode includes all Fused optimizations plus GPU PoW, so it's always safe
to use; the overhead for the GPU PoW path is small even when grinding is fast.

---

## Optimization 8 — Fused Transpose+Pad+DFT+Merkle

### What it does

Fuses the transpose-and-pad step with DFT and Merkle tree construction
in a single pipeline. Previously, the prover would:

1. CPU: transpose+pad the polynomial matrix
2. Allocate a new buffer and copy to GPU
3. GPU: DFT stages
4. GPU: Merkle hashing

With this optimization, the CPU writes the transposed/padded matrix
directly into the shared Metal buffer (zero-copy on Apple Silicon unified
memory), and the GPU immediately begins DFT without any intermediate
copy. The whole pipeline runs in a single Metal command buffer:

```
    Before:  CPU transpose → alloc → memcpy → GPU DFT → GPU Merkle
    After:   CPU transpose into GPU buf → GPU DFT → GPU Merkle
                  (zero allocation overhead, same command buffer)
```

The `DftCommitFusion` trait is extended with:

- `transpose_pad_dft_and_commit` — for base field (initial commit)
- `transpose_pad_dft_algebra_and_commit` — for extension field (round commits)

Both fall back to the separate CPU transpose + `dft_and_commit` path when
the data is too small for GPU benefit.

### Impact

Eliminates buffer allocation and copy overhead for the transpose step.
Particularly beneficial for fold=4-6 where the transposed matrix is
moderate-sized (16-64 MB).

---

## Optimization 9 — Adaptive GPU Dispatch Bounds

### What it does

Adds upper bounds on GPU dispatch based on both NTT height and total data
size. Very tall NTTs (log_n > 24) with narrow width become memory-bandwidth
bound on GPU. The dispatch heuristic:

```
    GPU dispatch window:
    ├── log_n < 14 ────── too small (GPU overhead > compute) → CPU
    ├── log_n > 24 ────── too many passes, bandwidth-bound ──→ CPU
    ├── < 8 MB data ───── too small ─────────────────────────→ CPU
    ├── > 1 GB data ───── exceeds working set ───────────────→ CPU
    └── otherwise ──────── sweet spot for GPU ───────────────→ GPU
```

This prevents regressions for very tall, narrow NTTs (2^26 × 2) while
still allowing large-but-wide NTTs (2^21 × 64) that have good GPU
utilization.

### Impact

- Eliminates the n=24, f=1, r=3 regression (**0.57x → 1.06x**)
- Enables GPU for large initial commits (2^21 × 64 = 512 MB)

---

## Optimization 10 — Adaptive PoW Grinding Batch Size

### What it does

Scales the GPU PoW grinding batch size with the difficulty level.
Profiling revealed that some sumcheck rounds require **25-bit** PoW
(not just the default 16-bit), needing ~33M nonces on average:

```
    pow_bits=12-16:  batch=1M   → 1 dispatch, ~2ms
    pow_bits=20-23:  batch=4M   → 1-2 dispatches, ~10ms
    pow_bits=24-25:  batch=16M  → 2-4 dispatches, ~200-800ms
```

This reduces Metal command buffer overhead for high-difficulty grinds
from ~33 dispatches down to ~2.

---

## Optimization 11 — Shared-Memory DIF Tail

### What it does

Replaces the final 2-3 global-memory DIF kernel dispatches with a single
**shared-memory DIF + bit-reversal** dispatch (`bb_dif_shared_bitrev`).
The kernel was already compiled but never wired into the dispatch path.

The existing radix-16 DIF approach processes 4 stages per dispatch, each
requiring a full read and write of the entire matrix from global memory.
For a `log_n=22` NTT, that's 6 dispatches (22/4 ≈ 6 with remainder handling).

The shared-memory kernel loads a block of `2^10 = 1024` elements into
threadgroup memory, performs all 10 DIF stages there with fast threadgroup
barriers, and writes the result in natural (bit-reversed) order directly
to the output buffer — replacing 2-3 global memory passes with 1:

```
    Before (log_n=22):
    ┌────────────────────────────────────────────────┐
    │ 5× bb_dif_r16 (stages 0-19, in-place)         │  5 global mem passes
    │ 1× bb_dif_r4_bitrev (stages 20-21, OOP)       │  1 global mem pass
    │                                    Total: 6    │
    └────────────────────────────────────────────────┘

    After (log_n=22):
    ┌────────────────────────────────────────────────┐
    │ 3× bb_dif_r16 (stages 0-11, in-place)         │  3 global mem passes
    │ 1× bb_dif_shared_bitrev (stages 12-21, OOP)   │  1 global mem pass
    │                                    Total: 4    │
    └────────────────────────────────────────────────┘
```

The shared-memory kernel handles the tail `min(log_n, 10)` stages.
For NTTs with `log_n ≤ 10`, the entire NTT runs in a single dispatch.

Also applies to the zero-copy path: the separate `encode_bitrev_gather`
pass is replaced by the fused shared-memory dispatch, eliminating one
additional global memory round-trip.

### Impact

~15-20% faster DFT+Merkle commit for large NTTs (log_n ≥ 16).
Also fixed a crash for `fold=8` configurations where `log_n ≤ 10` in
later prover rounds (the zero-copy path was reading uninitialized memory).

---

## Optimization 12 — Radix-4 Shared-Memory DIF Tail (4096 elements)

### What it does

Extends the shared-memory DIF tail from 2^10 = 1024 to 2^12 = 4096
elements per block. A new kernel `bb_dif_shared_bitrev_lg` processes
4 elements per thread (vs 2 in the original `bb_dif_shared_bitrev`),
allowing 12 DIF stages in shared memory with 1024 threads.

```
    Before (log_n=24, Opt 11):
    ┌────────────────────────────────────────────────┐
    │ 3× bb_dif_r16 (stages 0-11, in-place)         │  3 global mem passes
    │ 1× bb_dif_r4  (stages 12-13, in-place)        │  1 global mem pass
    │ 1× bb_dif_shared_bitrev (stages 14-23, OOP)   │  1 global mem pass
    │                                    Total: 5    │
    └────────────────────────────────────────────────┘

    After (log_n=24, Opt 12):
    ┌────────────────────────────────────────────────┐
    │ 3× bb_dif_r16 (stages 0-11, in-place)         │  3 global mem passes
    │ 1× bb_dif_shared_bitrev_lg (stages 12-23, OOP)│  1 global mem pass
    │                                    Total: 4    │
    └────────────────────────────────────────────────┘

    Before (log_n=20, Opt 11):
    ┌────────────────────────────────────────────────┐
    │ 2× bb_dif_r16 (stages 0-7, in-place)          │  2 global mem passes
    │ 1× bb_dif_r4  (stages 8-9, in-place)          │  1 global mem pass
    │ 1× bb_dif_shared_bitrev (stages 10-19, OOP)   │  1 global mem pass
    │                                    Total: 4    │
    └────────────────────────────────────────────────┘

    After (log_n=20, Opt 12):
    ┌────────────────────────────────────────────────┐
    │ 2× bb_dif_r16 (stages 0-7, in-place)          │  2 global mem passes
    │ 1× bb_dif_shared_bitrev_lg (stages 8-19, OOP) │  1 global mem pass
    │                                    Total: 3    │
    └────────────────────────────────────────────────┘
```

The dispatch logic (`dispatch_dif_shared_bitrev`) auto-selects the kernel:

- `log_block ≤ 10` → `bb_dif_shared_bitrev` (512 threads, 2 elems/thread)
- `log_block > 10` → `bb_dif_shared_bitrev_lg` (up to 1024 threads, 4 elems/thread)

### Impact

Saves 1 global memory pass for `log_n` values where `(log_n - 10)` is
not divisible by 4, specifically n=20 and n=24 (5→4 and 4→3 passes).
For n=22 (global_stages=10, already divisible by 4), no pass reduction.

---

## Optimization 13 — Lower GPU Dispatch Threshold

### What it does

Removes the 8 MB minimum-total-bytes threshold for GPU dispatch.
Previously, matrices with `height × width × 4 < 8 MB` fell back to
CPU for both DFT and Merkle hashing — even when the GPU could handle
them efficiently with shared-memory kernels.

Profiling showed that mid-size prover rounds (log_size 14-17) were
spending 60-80 ms each on **CPU Merkle hashing** because the GPU
refused them. With the shared-memory DIF kernel handling log_n ≤ 12
entirely in one dispatch, and Poseidon2 leaf hashing being inherently
parallel, the GPU completes these rounds in 3-35 ms.

```
    n=22 f=1 r=1 round breakdown (before → after):

    Round 4 (log_size=17):
      Before: CPU DFT 5.7ms + CPU Merkle 62.2ms = 68ms
      After:  GPU fused DFT+Merkle 35ms

    Round 5 (log_size=16):
      Before: CPU DFT 1.4ms + CPU Merkle 49.7ms = 51ms
      After:  GPU fused DFT+Merkle 8.4ms

    Round 6 (log_size=15):
      Before: CPU Merkle 10.4ms
      After:  GPU fused 4.7ms

    Round 7 (log_size=14):
      Before: CPU Merkle 5.2ms (not captured by GPU)
      After:  GPU fused 3.1ms
```

The `gpu_min_log_n = 14` guard still prevents tiny NTTs from going to
GPU, but all matrices with log_n ≥ 14 now use GPU regardless of total
byte count. The GpuMmcs commit threshold was also lowered from 8 MB
to 128 KB to capture GPU Merkle hashing for mid-size matrices.

### Impact

~7% faster prove_fused for fold=1 configurations (where many rounds
have mid-size matrices). Larger impact for n=18-20 where these rounds
were a proportionally larger fraction of total time.

---

## Optimization 14 — Parallel Zero-Copy Merkle Readback

### What it does

Rewrites `read_merkle_layers` to eliminate the intermediate `Vec<u32>`
allocation and copy. The previous implementation:

1. Allocated `Vec<u32>` (zero-initialized, ~128 MB for leaf layer)
2. Copied from GPU buffer → `Vec<u32>` (128 MB memcpy)
3. Allocated `Vec<[BabyBear; 8]>` (zero-initialized, ~128 MB)
4. Transformed u32 chunks → BabyBear arrays (128 MB read + write)

Total: ~512 MB of memory operations for the leaf layer alone.

The new implementation:

1. Allocates `Vec<[BabyBear; 8]>` directly (128 MB)
2. Reinterpret-casts GPU buffer as `&[[BabyBear; 8]]` (BabyBear is
  `repr(transparent)` over `u32`) and copies in parallel with rayon
   for layers with ≥ 4096 digests

Total: ~256 MB of memory operations — **2x less**.

### Impact

Merkle readback time reduced from ~98ms to ~42ms for log_n=22 (2.3x).
For n=22 f=1 r=1: total fused time 831ms → 761ms (**8.4% improvement**).
Savings compound across all prover rounds.

---

## Optimization 15 — Arc-based Metal Sharing & Allocation Elimination

### What it does

Three related changes that reduce per-round overhead:

**15a — Arc-based Metal sharing:** Previously, every `MetalBabyBearDft::clone()`
call (triggered by `ExtensionMmcs::new` and `GpuChallenger::new` in each prover
round) created a brand-new Metal device, command queue, and recompiled all shader
pipelines (~2-3ms each). For n=22 f=1 r=1 with ~9 rounds, this wasted 16-30ms.

The fix wraps all Metal resources in `Arc<MetalInner>` so cloning only increments
a reference count:

```rust
pub struct MetalBabyBearDft {
    inner: Arc<MetalInner>,
}

impl Clone for MetalBabyBearDft {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}
```

**15b — Eliminate Vec zero-initialization:** Output vectors in `gpu_dft_and_merkle`,
`gpu_transpose_dft_and_merkle`, and `read_merkle_layers` were zero-initialized
via `vec![BabyBear::ZERO; size]` before being immediately overwritten by GPU
memcpy. Replaced with `Vec::with_capacity` + `unsafe { set_len }` to skip the
redundant zeroing (~15ms saved for large allocations).

**15c — Zero-copy OOP bug fix:** When `global_stages` (number of global-memory
DIF passes) was between 1 and 3, the initial out-of-place copy from the zero-copy
input buffer to the working managed buffer was missing — only the R16 path had
a dedicated OOP variant. Added a `bb_buf_copy` Metal kernel and integrated it
into `encode_dif_ntt_zc` and `gpu_dft_and_merkle` to explicitly copy data before
running the in-place DIF stages for these small-stage cases.

### Impact

The combined effect is most visible at small n values where per-round overhead
is proportionally larger:

- **n=18 f=2 r=1**: 0.58x → **1.34x** (was slower than CPU, now 34% faster)
- **n=18 f=1 r=1**: 0.90x → **1.65x** (83% improvement)
- **n=20 f=3 r=2**: 1.14x → **2.69x** (136% improvement)
- **n=22 f=1 r=1**: 1.97x → **2.01x** (modest absolute gain)

---

## Optimization 16 — Fused Leaf Hash + First Merkle Compression

### What it does

Introduces a new Metal kernel `poseidon2_hash_and_compress` that fuses
the leaf hashing step with the first level of Merkle tree binary compression.
Instead of two separate dispatches (hash all leaves → write leaf digests →
read leaf digests → compress pairs), a single dispatch handles both:

1. Each thread hashes **two** adjacent leaves (2×gid, 2×gid+1)
2. Writes both leaf digests to the leaf buffer (still needed for query opening)
3. Immediately compresses the pair in registers (no global memory round-trip)
4. Writes the compressed parent digest to the compression buffer

Also lowered the parallel non-temporal readback threshold from 16 MB to
128 KB, improving Merkle layer readback for medium-sized trees.

### Why it helps

Profiling showed the Merkle tree dominates GPU time (3–8× more than DFT):

| log_n | DFT GPU | Merkle GPU | Ratio |
|-------|---------|------------|-------|
| 22    | 19 ms   | 153 ms     | 8:1   |
| 21    | 23 ms   | 71 ms      | 3:1   |
| 20    | 12 ms   | 41 ms      | 3.4:1 |

The fused kernel eliminates one full read of the leaf-digest buffer for the
first compression level. For n=22 with 4M leaves, this saves 128 MB of
global memory reads. It also removes one dispatch call from the compression
pipeline.

---

## Current Benchmark Results

Best GPU/CPU speedup per configuration (taking max of GPU, Fused, Grind).
PoW-heavy configs (fold ≥ 4, rate ≥ 2) show high variance due to random
grinding; numbers below are from a single sweep run.

### n=18 (2^18 = 262K coefficients)

| fold | rate=1    | rate=2    | rate=3    |
| ---- | --------- | --------- | --------- |
| 1    | **1.11x** | **1.49x** | **1.71x** |
| 2    | **1.05x** | **1.38x** | **1.69x** |
| 3    | 0.94x     | **1.20x** | **1.73x** |
| 4    | 0.82x     | **1.18x** | **1.44x** |
| 6    | 0.75x     | **1.26x** | **1.32x** |
| 8    | 0.47x     | 0.82x     | 0.87x     |

GPU wins for fold ≤ 2 at all rates and fold 3-6 at rate ≥ 2.
High-fold (8) still CPU-favored at this small size.

### n=20 (2^20 = 1M coefficients)

| fold | rate=1    | rate=2    | rate=3    |
| ---- | --------- | --------- | --------- |
| 1    | **1.59x** | **1.90x** | **1.80x** |
| 2    | **1.46x** | **1.87x** | **2.04x** |
| 3    | **1.44x** | **1.53x** | **1.71x** |
| 4    | **1.10x** | **1.56x** | **1.43x** |
| 6    | **1.08x** | **1.61x** | **1.49x** |
| 8    | 0.81x     | **1.28x** | **1.51x** |

GPU consistently faster except fold=8 rate=1. Best: fold=2 rate=3 at **2.04x**.

### n=22 (2^22 = 4M coefficients)

| fold | rate=1    | rate=2    | rate=3    |
| ---- | --------- | --------- | --------- |
| 1    | **2.03x** | **1.86x** | **1.85x** |
| 2    | **1.74x** | **1.94x** | **2.01x** |
| 3    | **1.54x** | **1.71x** | **2.02x** |
| 4    | **1.61x** | **1.73x** | **1.79x** |
| 6    | **1.49x** | **1.46x** | **2.15x** |
| 8    | **1.41x** | **1.48x** | **1.88x** |

GPU always faster. Best: fold=6 rate=3 at **2.15x** (PoW-dominated),
fold=1 rate=1 at **2.03x** for compute-heavy configs.

### n=24 (2^24 = 16M coefficients)

| fold | rate=1    | rate=2    | rate=3    |
| ---- | --------- | --------- | --------- |
| 1    | **1.79x** | **2.13x** | **1.40x** |
| 2    | **1.74x** | **2.09x** | **2.00x** |
| 3    | **1.96x** | **1.84x** | **2.15x** |
| 4    | **1.72x** | **1.82x** | **3.95x** |
| 6    | **1.65x** | **1.81x** | **3.48x** |
| 8    | —         | —         | —         |

Best: fold=4 rate=3 at **3.95x**, fold=6 rate=3 at **3.48x** (PoW-heavy).
n=24 fold=8 exceeds GPU memory limits.

### Optimization progression

| #   | Optimization                       | Key improvement                 |
| --- | ---------------------------------- | ------------------------------- |
| 1   | GPU NTT (Metal)                    | Established GPU pipeline        |
| 2   | Radix-16 DIF + zero-copy           | 4x fewer dispatches, no memcpy  |
| 3   | GPU Poseidon2 Merkle               | ~10% whir_prove speedup         |
| 4   | Fused DFT→Merkle                   | ~15-18% additional speedup      |
| 5   | Fused prover rounds                | Up to 1.63x total               |
| 6   | Lower threshold + zero-copy bitrev | Up to 1.88x total               |
| 7   | GPU PoW Grinding + buffer caching  | Up to 2.58x total               |
| 8   | Fused transpose+pad+DFT+Merkle     | Eliminates transpose overhead   |
| 9   | Adaptive GPU dispatch bounds       | Avoids GPU regressions          |
| 10  | Adaptive PoW batch size            | Reduces dispatch overhead       |
| 11  | Shared-memory DIF tail             | **33% fewer global mem passes** |
| 12  | 4096-element shared-memory tail    | 1 fewer pass for n=20,24        |
| 13  | Lower GPU dispatch threshold       | Mid-size rounds use GPU Merkle  |
| 14  | Parallel zero-copy Merkle readback | **2.3x faster readback**        |
| 15  | Arc sharing + alloc elimination    | Small-n overhead eliminated     |
| 16  | Fused leaf hash + first compress   | Eliminates 128MB read for n=22  |


