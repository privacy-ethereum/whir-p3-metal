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
CPU input (zero-copy) → R16 OOP → DIF stages (managed) → bitrev gather (managed→zero-copy)
    → Poseidon2 leaf hash → compress layers → [single wait] → result already in CPU memory
```

The bitrev gather writes directly back into the zero-copy buffer (the
caller's `values` Vec), and Merkle hashing reads from the same buffer.
This eliminates both a separate GPU buffer allocation and the full-matrix
memcpy that was previously needed after GPU completion.

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
| 4 | 1 | 184 | 155 | 139 | **1.19x** | **1.32x** |
| 4 | 2 | 342 | 254 | 238 | **1.35x** | **1.44x** |
| 4 | 3 | 856 | 673 | 765 | **1.27x** | 1.12x |
| 6 | 1 | 154 | 142 | 174 | 1.08x | 0.89x |
| 6 | 2 | 414 | 498 | 346 | 0.83x | **1.20x** |
| **8** | **1** | **126** | **70** | **79** | **1.80x** | **1.59x** |
| 8 | 2 | 184 | 158 | 156 | **1.17x** | **1.18x** |
| 8 | 3 | 436 | 369 | 326 | 1.18x | **1.34x** |

#### Parameter sweep — n=24 (16M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Fused (ms) | GPU speedup | Fused speedup |
|------|------|----------|----------|------------|-------------|---------------|
| **4** | **1** | **1097** | **955** | **700** | **1.15x** | **1.57x** |
| **6** | **1** | **590** | **527** | **391** | **1.12x** | **1.51x** |

#### Observations

- **Best GPU speedup: n=22, fold=8, rate=1 at 1.80x.** Lowering the GPU
  dispatch threshold from 64 MB to 8 MB allowed per-round DFT matrices
  (previously just below the cutoff) to run on GPU, dramatically improving
  this config from break-even to nearly 2x.
- **Round fusion + low threshold: n=24, fold=4, rate=1 at 1.57x.** The
  fused round path (`prove_fused`) adds 0.2–0.4x over commit-only fusion
  on larger polynomials.
- **Zero-copy bitrev gather** eliminates the post-GPU full-matrix memcpy.
  On Apple Silicon unified memory, the bitrev gather writes directly into
  the caller's Vec and Merkle hashing reads from the same buffer.
- **GPU loses at very small data (<8 MB)** — the pipeline automatically
  falls back to CPU.
- **Folding factors ≥ 10 on GPU** are unstable on some Apple Silicon
  configurations (Metal driver crashes).
- GPU domain size is capped at 2^25 to avoid Metal driver crashes.

#### Running benchmarks

```bash
# Full whir_prove benchmark (default config: n=24, fold=4, rate=1)
cargo bench --features gpu-metal --bench dft_gpu -- "whir_prove"

# Parameter sweep comparing CPU / GPU / GPU+fused rounds
# (writes results to sweep_results.txt)
cargo run --release --features gpu-metal --bin sweep
```
