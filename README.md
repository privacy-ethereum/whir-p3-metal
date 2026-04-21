# whir-p3

A version of https://github.com/WizardOfMenlo/whir/ which uses the Plonky3 library.

## GPU acceleration (Apple Silicon / Metal)

Enable with `--features gpu-metal`. Accelerates the `whir_prove` pipeline by
offloading NTT (Number Theoretic Transform) and Merkle tree construction to the
GPU via Metal compute shaders.

### Architecture

The GPU pipeline lives in two files:

- **`shaders/babybear_ntt.metal`** — Metal Shading Language kernels for BabyBear
  field arithmetic, DIF-NTT butterfly stages, bit-reversal, and Poseidon2 hashing.
- **`src/gpu_dft.rs`** — Rust orchestration: Metal pipeline setup, buffer
  management, kernel dispatch, and the `GpuMmcs` wrapper that plugs into
  Plonky3's `Mmcs` trait.

### Key optimizations

**1. Radix-16 DIF NTT with zero-copy I/O**

NTT uses decimation-in-frequency (DIF) with fused radix-16/8/4/2 butterfly
kernels. On Apple Silicon's unified memory, input data is wrapped as a
zero-copy Metal buffer (no CPU→GPU copy). DIF stages run in-place on a
GPU-managed buffer, and the final bit-reversal gather writes results back to
the zero-copy buffer with coalesced sequential writes.

**2. GPU Poseidon2 Merkle tree**

Merkle tree construction uses Poseidon2 (width-16, 8+13+4 rounds, x^7 S-box)
implemented entirely in Metal. Leaf hashing and all compression layers are
dispatched in a single command buffer — Apple GPU guarantees dispatch ordering
within a compute command encoder, so no explicit barriers are needed.

**3. Fused DFT → Merkle pipeline (`DftCommitFusion` trait)**

Instead of running DFT on GPU, copying the result back to CPU, and
re-uploading for Merkle hashing, the fused path runs DFT + bit-reversal +
Poseidon2 leaf hashing + all Merkle compression layers in a **single GPU
command buffer** with zero CPU round-trips between stages.

This is exposed via the `DftCommitFusion<F>` trait, which `GpuMmcs`
implements. The fusion is used in two places:

1. **Initial commit** — `CommitmentWriter::commit_fused()` fuses the base-field
   DFT + Merkle for the initial polynomial commitment.
2. **Per-round commits** — `Prover::prove_fused()` fuses the extension-field
   DFT (`dft_algebra_batch`) + Merkle for every STIR round that has a large
   enough matrix. Falls back to separate DFT + commit when the matrix is
   below the GPU threshold (64 MB).

The fused pipeline for a single commit:

```
CPU input (zero-copy) → R16 OOP → DIF stages (managed) → bitrev gather (managed→managed)
    → Poseidon2 leaf hash → compress layers → [single wait] → memcpy result to CPU
```

vs the non-fused pipeline:

```
CPU input (zero-copy) → DIF stages → bitrev → CPU    [GPU command buffer 1, wait]
CPU → upload to GPU → leaf hash → compress layers     [GPU command buffer 2, wait]
```

**4. Montgomery arithmetic in Metal**

BabyBear field operations use Montgomery form throughout. The Montgomery
multiply uses only 32-bit `mul`/`mulhi` instructions (no 64-bit arithmetic),
mapping efficiently to Apple GPU's ALU.

### Benchmarks

All benchmarks on Apple M-series silicon (unified memory). Parameters:
- `n` = `num_variables` (polynomial has 2^n coefficients)
- `fold` = `folding_factor` (each STIR round folds 2^fold evaluations)
- `rate` = `starting_log_inv_rate` (RS code rate = 1/2^rate, domain = 2^(n+rate) points)
- **GPU** = fused initial commit only (rounds use separate DFT + commit)
- **Fused** = fused initial commit + fused per-round DFT+Merkle (`prove_fused`)
- Speedup = CPU time / GPU or Fused time

#### Parameter sweep — n=22 (4M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU speedup | Fused speedup |
|------|------|----------|----------|------------|-------------|---------------|
| 4 | 1 | 167 | 165 | 176 | 1.01x | 0.95x |
| 4 | 2 | 379 | 260 | 277 | **1.46x** | **1.37x** |
| 4 | 3 | 731 | 671 | 622 | 1.09x | **1.17x** |
| 6 | 1 | 142 | 133 | 146 | 1.07x | 0.98x |
| 6 | 2 | 421 | 379 | 363 | 1.11x | **1.16x** |
| 6 | 3 | 2723 | 1715 | 2380 | **1.59x** | 1.14x |
| 8 | 1 | 93 | 95 | 96 | 0.98x | 0.97x |
| 8 | 2 | 189 | 139 | 155 | **1.36x** | **1.22x** |
| 8 | 3 | 444 | 328 | **283** | 1.35x | **1.57x** |
| 10 | 1 | 125 | 116 | 114 | 1.08x | **1.10x** |

#### Parameter sweep — n=24 (16M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU speedup | Fused speedup |
|------|------|----------|----------|------------|-------------|---------------|
| **4** | **1** | **1142** | **939** | **727** | **1.22x** | **1.57x** |
| **6** | **1** | **704** | **484** | **432** | **1.45x** | **1.63x** |
| **10** | **1** | **491** | fail | **351** | - | **1.40x** |

#### Observations

- **Round fusion matters at n=24.** The fused round path (`prove_fused`)
  adds 0.2–0.4x additional speedup over commit-only fusion on larger
  polynomials. At n=24 fold=4 rate=1, fusion improves from 1.22x to 1.57x.
- **Best overall speedup: n=24, fold=6, rate=1 at 1.63x** with full fusion.
  The round DFT matrices at this config are large enough (>64 MB) for the
  GPU fusion to kick in during STIR rounds.
- **Round fusion wins when round matrices are large.** At n=22 fold=8 rate=3,
  the round matrix is large enough for GPU fusion, yielding 1.57x (fused)
  vs 1.35x (commit-only).
- **Round fusion can hurt on small round matrices.** At n=22 fold=6 rate=3,
  the fused path (1.14x) underperforms commit-only (1.59x) because the
  round DFTs are borderline size and the fusion overhead dominates.
- **GPU loses at very small data (<64 MB)** — the pipeline automatically
  falls back to CPU.
- **Folding factors ≥ 12 are untested on GPU** — they trigger Metal driver
  instability (kernel panics) on some Apple Silicon configurations.
- GPU domain size is capped at 2^25 to avoid Metal driver crashes.

#### Running benchmarks

```bash
# Full whir_prove benchmark (default config: n=24, fold=4, rate=1)
cargo bench --features gpu-metal --bench dft_gpu -- "whir_prove"

# Parameter sweep comparing CPU / GPU / GPU+fused rounds
# (writes results to sweep_results.txt)
cargo run --release --features gpu-metal --bin sweep
```
