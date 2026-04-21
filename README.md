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

The biggest optimization: instead of running DFT on GPU, copying the result
back to CPU, and re-uploading for Merkle hashing, the fused path runs
DFT + bit-reversal + Poseidon2 leaf hashing + all Merkle compression layers
in a **single GPU command buffer** with zero CPU round-trips between stages.

This is exposed via the `DftCommitFusion<F>` trait, which `GpuMmcs`
implements. The committer's `commit_fused()` method tries the fused path
first and falls back to separate DFT + commit if the matrix is too small for
GPU benefit (threshold: 64 MB).

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
- Speedup = CPU time / GPU time

#### Parameter sweep — n=22 (4M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Speedup |
|------|------|----------|----------|---------|
| 4 | 1 | 182 | 201 | 0.91x |
| 4 | 2 | 385 | 284 | **1.35x** |
| 4 | 3 | 818 | 579 | **1.41x** |
| **6** | **1** | **265** | **142** | **1.87x** |
| **6** | **2** | **911** | **491** | **1.86x** |
| 8 | 1 | 125 | 98 | **1.28x** |
| 8 | 2 | 210 | 135 | **1.55x** |
| 8 | 3 | 405 | 327 | **1.24x** |
| 10 | 1 | 140 | 112 | **1.26x** |
| 10 | 3 | 1696 | 1500 | **1.13x** |

#### Parameter sweep — n=24 (16M coefficients)

| fold | rate | CPU (ms) | GPU (ms) | Speedup |
|------|------|----------|----------|---------|
| 4 | 1 | 982 | 921 | **1.07x** |
| 4 | 2 | 5034 | 3651 | **1.38x** |
| 6 | 1 | 579 | 488 | **1.19x** |
| 6 | 2 | 2221 | 2005 | **1.11x** |
| 6 | 3 | 11844 | 8539 | **1.39x** |
| 7 | 1 | 2438 | 2070 | **1.18x** |
| 8 | 1 | 40070 | 35668 | **1.12x** |

#### Observations

- **Best GPU speedup: fold=6 at rate 1-2** — consistently 1.8-1.9x for n=22.
  Fold=6 produces moderately large DFTs (2^16-2^18 rows) that fully saturate
  GPU parallelism while keeping the number of STIR rounds manageable.
- **Higher rates amplify GPU advantage** (larger domain → more GPU-friendly
  work), but rates above 3 make both CPU and GPU very slow.
- **GPU loses when data < 64 MB** — the fused pipeline skips GPU and falls
  back to CPU automatically (see threshold in `gpu_dft_and_merkle`).
- **Folding factors ≥ 12 are untested on GPU** — they trigger Metal driver
  instability (kernel panics) on some Apple Silicon configurations.
- The GPU pipeline is most beneficial when the initial commit DFT processes
  2^16 to 2^21 rows with total data between 64 MB and 256 MB.

#### Running benchmarks

```bash
# Full whir_prove benchmark (default config: n=24, fold=4, rate=1)
cargo bench --features gpu-metal --bench dft_gpu -- "whir_prove"

# Parameter sweep (writes results to sweep_results.txt)
cargo run --release --features gpu-metal --bin sweep
```
