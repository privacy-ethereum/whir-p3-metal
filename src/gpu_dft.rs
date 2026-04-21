//! GPU-accelerated DFT for BabyBear using **Metal compute only** (embedded MSL).
//!
//! Pipeline: Cooley–Tukey radix-2 DIT — bit-reverse permutation, then butterfly stages.
//! All GPU kernels operate on **row-major** matrices (`data[row * width + col]`),
//! processing every column in a single dispatch.  No CPU transpose is needed.

extern crate alloc;

use alloc::vec::Vec;
use std::collections::HashMap;
use std::mem::size_of;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use objc::rc::autoreleasepool;
use p3_baby_bear::BabyBear;
use rayon::prelude::*;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

use p3_commit::Mmcs;
use p3_field::BasedVectorSpace;

const SHADER_MSL: &str = include_str!("../shaders/babybear_ntt.metal");

/// Optional DFT+Merkle fusion for MMCS implementations.
/// Returns `Ok(commitment, tree)` on success, `Err(mat)` on failure
/// (returning the unconsumed matrix for fallback).
pub trait DftCommitFusion<F: Field>: Mmcs<F> {
    fn dft_and_commit(
        &self,
        mat: RowMajorMatrix<F>,
    ) -> Result<
        (Self::Commitment, Self::ProverData<RowMajorMatrix<F>>),
        RowMajorMatrix<F>,
    > {
        Err(mat)
    }

    fn dft_algebra_and_commit<EF>(
        &self,
        mat: RowMajorMatrix<EF>,
    ) -> Result<
        (
            Self::Commitment,
            Self::ProverData<p3_matrix::extension::FlatMatrixView<F, EF, RowMajorMatrix<EF>>>,
        ),
        RowMajorMatrix<EF>,
    >
    where
        EF: p3_field::ExtensionField<F> + BasedVectorSpace<F> + Clone + Send + Sync,
    {
        Err(mat)
    }

    /// Fused transpose + pad + DFT + commit for base field.
    /// Input is a flat slice representing an [in_rows × in_cols] matrix.
    /// Returns `None` if the implementation doesn't support this operation.
    fn transpose_pad_dft_and_commit(
        &self,
        _data: &[F],
        _in_rows: usize,
        _in_cols: usize,
        _padded_height: usize,
    ) -> Option<(Self::Commitment, Self::ProverData<RowMajorMatrix<F>>)> {
        None
    }

    /// Fused transpose + pad + DFT + commit for extension field.
    fn transpose_pad_dft_algebra_and_commit<EF>(
        &self,
        _data: &[EF],
        _in_rows: usize,
        _in_cols: usize,
        _padded_height: usize,
    ) -> Option<(
        Self::Commitment,
        Self::ProverData<p3_matrix::extension::FlatMatrixView<F, EF, RowMajorMatrix<EF>>>,
    )>
    where
        EF: p3_field::ExtensionField<F> + BasedVectorSpace<F> + Clone + Send + Sync,
    {
        None
    }
}

/// Build the Poseidon2 BabyBear width-16 constants buffer (143 Montgomery u32s)
/// by replicating the same RNG sampling as `Poseidon2::new_from_rng_128`.
/// Layout: [0..64) ext_initial_rc, [64..77) int_rc, [77..141) ext_terminal_rc,
///         [141] INV_256, [142] INV_2_27.
fn build_poseidon2_constants(seed: u64) -> Vec<u32> {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    let bb_to_u32 = |bb: BabyBear| -> u32 {
        unsafe { std::mem::transmute::<BabyBear, u32>(bb) }
    };

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut buf = Vec::with_capacity(143);

    // Replicate ExternalLayerConstants::new_from_rng(rounds_f=8, rng):
    // initial_external_constants: 4 × [BabyBear; 16]
    for _ in 0..4 {
        let round: [BabyBear; 16] = rng.random();
        for bb in round { buf.push(bb_to_u32(bb)); }
    }
    // terminal_external_constants: 4 × [BabyBear; 16]
    let term_start = buf.len();
    for _ in 0..4 {
        let round: [BabyBear; 16] = rng.random();
        for bb in round { buf.push(bb_to_u32(bb)); }
    }
    let terminal_constants: Vec<u32> = buf[term_start..].to_vec();
    buf.truncate(term_start); // Remove terminal, insert internal first

    // internal_constants: 13 BabyBear values
    for _ in 0..13 {
        let bb: BabyBear = rng.random();
        buf.push(bb_to_u32(bb));
    }

    // Now append terminal constants
    buf.extend_from_slice(&terminal_constants);

    // INV_256 = (2^8)^{-1} mod p, in Montgomery form
    buf.push(bb_to_u32(BabyBear::new(256).inverse()));
    // INV_2_27 = (2^{27})^{-1} mod p, in Montgomery form
    buf.push(bb_to_u32(BabyBear::new(1u32 << 27).inverse()));

    assert_eq!(buf.len(), 143);
    buf
}


/// Precompute twiddle factors in **Montgomery form** (raw `BabyBear` representation).
/// The GPU shader uses Montgomery multiplication, so twiddles must be in the same form.
fn precompute_twiddle_monty(log_n: u32) -> Vec<u32> {
    let omega = BabyBear::two_adic_generator(log_n as usize);
    let half_n = 1usize << (log_n - 1);
    let mut twiddles: Vec<BabyBear> = Vec::with_capacity(half_n);
    let mut acc = BabyBear::ONE;
    twiddles.push(acc);
    for _ in 1..half_n {
        acc *= omega;
        twiddles.push(acc);
    }
    // BabyBear = MontyField31<..> is #[repr(transparent)] over u32.
    babybear_vec_to_u32(twiddles)
}

/// Reinterpret `Vec<BabyBear>` as `Vec<u32>` (zero-cost, same layout).
fn babybear_vec_to_u32(v: Vec<BabyBear>) -> Vec<u32> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: BabyBear is #[repr(transparent)] over u32 — identical layout.
    unsafe { Vec::from_raw_parts(ptr.cast::<u32>(), len, cap) }
}


/// Memcpy that uses rayon for large transfers to increase DRAM bandwidth utilization.
unsafe fn fast_memcpy(dst: *mut u8, src: *const u8, len: usize) {
    const MIN_PARALLEL: usize = 16 * 1024 * 1024;
    if len < MIN_PARALLEL {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    let src_slice = unsafe { std::slice::from_raw_parts(src, len) };
    let n = rayon::current_num_threads().max(4);
    let chunk = (len + n - 1) / n;
    dst_slice.chunks_mut(chunk)
        .zip(src_slice.chunks(chunk))
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(d, s)| d.copy_from_slice(s));
}

/// Like `fast_memcpy` but uses ARM64 non-temporal loads (LDNP) to read
/// GPU-written memory without triggering CPU cache-line fills.  This avoids
/// the coherency stall that occurs when normal loads try to pull GPU-dirty
/// cache lines into the CPU L1/L2.
#[cfg(target_arch = "aarch64")]
unsafe fn fast_memcpy_from_gpu(dst: *mut u8, src: *const u8, len: usize) {
    const MIN_PARALLEL: usize = 16 * 1024 * 1024;
    if len < MIN_PARALLEL {
        nontemporal_copy(dst, src, len);
        return;
    }
    let n = rayon::current_num_threads().max(4);
    let chunk = (len + n - 1) / n;
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, len) };
    let src_slice = unsafe { std::slice::from_raw_parts(src, len) };
    dst_slice.chunks_mut(chunk)
        .zip(src_slice.chunks(chunk))
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(d, s)| unsafe {
            nontemporal_copy(d.as_mut_ptr(), s.as_ptr(), s.len());
        });
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn nontemporal_copy(dst: *mut u8, src: *const u8, len: usize) {
    let mut i = 0usize;
    while i + 64 <= len {
        core::arch::asm!(
            "ldnp q0, q1, [{src}]",
            "ldnp q2, q3, [{src}, #32]",
            "stnp q0, q1, [{dst}]",
            "stnp q2, q3, [{dst}, #32]",
            src = in(reg) src.add(i),
            dst = in(reg) dst.add(i),
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
        );
        i += 64;
    }
    if i < len {
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn fast_memcpy_from_gpu(dst: *mut u8, src: *const u8, len: usize) {
    fast_memcpy(dst, src, len);
}

/// Metal-backed DFT for BabyBear (macOS).
/// Falls back to CPU for small sizes or if the GPU fails.
pub struct MetalBabyBearDft {
    cpu: Radix2DFTSmallBatch<BabyBear>,
    gpu_min_log_n: u32,
    pub device: Device,
    queue: metal::CommandQueue,
    bitrev_ps: ComputePipelineState,
    bitrev_gather_ps: ComputePipelineState,
    dif_r16_oop_ps: ComputePipelineState,
    butterfly_ps: ComputePipelineState,
    butterfly_r4_ps: ComputePipelineState,
    butterfly_r8_ps: ComputePipelineState,
    shared_mem_ps: ComputePipelineState,
    shared_mem_gs_ps: ComputePipelineState,
    stockham_ps: ComputePipelineState,
    stockham_gs_ps: ComputePipelineState,
    stockham_global_r2_ps: ComputePipelineState,
    twiddle_transpose_ps: ComputePipelineState,
    dif_r32_ps: ComputePipelineState,
    dif_r16_ps: ComputePipelineState,
    dif_r8_ps: ComputePipelineState,
    dif_r4_ps: ComputePipelineState,
    dif_r2_ps: ComputePipelineState,
    dif_r32_bitrev_ps: ComputePipelineState,
    dif_r16_bitrev_ps: ComputePipelineState,
    dif_r8_bitrev_ps: ComputePipelineState,
    dif_r4_bitrev_ps: ComputePipelineState,
    dif_r2_bitrev_ps: ComputePipelineState,
    dif_shared_bitrev_ps: ComputePipelineState,
    /// Hard upper bound on the shared-memory block log from the device
    /// (max threads per threadgroup). The kernel tiles columns internally,
    /// so the actual `log_block` is just `min(this, log_n)`.
    max_log_shared_block: u32,
    /// Stockham radix-4 processes 4 elements per thread, so the block
    /// size is 4× larger than max threads:  log_block = max_log_shared_block + 2.
    max_log_stockham_block: u32,
    twiddle_bufs: Mutex<HashMap<u32, Buffer>>,
    /// Cached temporary buffer for four-step FFT (avoids per-call allocation).
    temp_buf_cache: Mutex<Option<Buffer>>,
    /// Cached data buffer (avoids per-call allocation).
    data_buf_cache: Mutex<Option<Buffer>>,
    #[allow(dead_code)]
    dft_result_cache: Arc<Mutex<Option<(usize, usize, Buffer)>>>,
    // Poseidon2 Merkle hashing pipeline states
    poseidon2_hash_leaves_ps: ComputePipelineState,
    poseidon2_compress_ps: ComputePipelineState,
    poseidon2_pow_grind_ps: ComputePipelineState,
    transpose_pad_ps: ComputePipelineState,
    /// Poseidon2 round constants buffer (143 u32 values in Montgomery form).
    poseidon2_rc_buf: Buffer,
    /// Pre-allocated buffers for PoW grinding (avoids per-call allocation).
    pow_bufs: Mutex<PowBuffers>,
}

struct PowBuffers {
    state_buf: Buffer,
    params_buf: Buffer,
    result_buf: Buffer,
    found_buf: Buffer,
    offset_buf: Buffer,
    r_squared: u32,
    tg_size: u64,
}

impl Clone for MetalBabyBearDft {
    fn clone(&self) -> Self {
        let mut new = Self::new_with_params(self.cpu.clone(), self.gpu_min_log_n, Self::DEFAULT_POSEIDON2_SEED);
        // Share the DFT result cache so cloned instances (DFT and GpuMmcs)
        // can communicate GPU buffer availability.
        new.dft_result_cache = Arc::clone(&self.dft_result_cache);
        new
    }
}

impl Default for MetalBabyBearDft {
    fn default() -> Self {
        Self::new_with_params(Radix2DFTSmallBatch::default(), Self::DEFAULT_GPU_MIN_LOG_N, Self::DEFAULT_POSEIDON2_SEED)
    }
}

impl MetalBabyBearDft {
    /// Minimum log_n for GPU dispatch. The 8MB total-data threshold (in
    /// try_gpu_dft_inplace) is the primary gate; this only guards against
    /// degenerate cases with very few NTT stages but huge width.
    const DEFAULT_GPU_MIN_LOG_N: u32 = 14;
    /// Maximum total bytes for GPU dispatch. Very large, narrow NTTs
    /// (e.g. 2^26 × 2) become memory-bandwidth bound. We gate on both
    /// log_n and total bytes to allow large-but-wide NTTs (2^21 × 64)
    /// while blocking tall-and-narrow ones.
    const GPU_MAX_LOG_N: u32 = 24;
    const GPU_MAX_TOTAL_BYTES: usize = 1024 * 1024 * 1024;
    /// Default Poseidon2 RNG seed (must match the seed used to construct the permutation).
    const DEFAULT_POSEIDON2_SEED: u64 = 1;

    pub fn new(max_fft_size: usize) -> Self {
        Self::new_with_params(
            Radix2DFTSmallBatch::new(max_fft_size),
            Self::DEFAULT_GPU_MIN_LOG_N,
            Self::DEFAULT_POSEIDON2_SEED,
        )
    }

    pub fn new_with_poseidon2_seed(max_fft_size: usize, poseidon2_seed: u64) -> Self {
        Self::new_with_params(
            Radix2DFTSmallBatch::new(max_fft_size),
            Self::DEFAULT_GPU_MIN_LOG_N,
            poseidon2_seed,
        )
    }

    fn new_with_params(cpu: Radix2DFTSmallBatch<BabyBear>, gpu_min_log_n: u32, poseidon2_seed: u64) -> Self {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_MSL, &opts)
            .unwrap_or_else(|e| panic!("Metal shader compile failed: {e}"));

        let make_ps = |name| {
            let f = library.get_function(name, None).unwrap_or_else(|_| {
                panic!("Metal function {name} not found");
            });
            device
                .new_compute_pipeline_state_with_function(&f)
                .unwrap_or_else(|e| panic!("Pipeline {name}: {e}"))
        };

        let bitrev_ps = make_ps("bb_ntt_bitrev");
        let bitrev_gather_ps = make_ps("bb_bitrev_gather");
        let dif_r16_oop_ps = make_ps("bb_dif_r16_oop");
        let butterfly_ps = make_ps("bb_ntt_butterfly");
        let butterfly_r4_ps = make_ps("bb_ntt_butterfly_r4");
        let butterfly_r8_ps = make_ps("bb_ntt_butterfly_r8");
        let shared_mem_ps = make_ps("bb_ntt_shared_mem");
        let shared_mem_gs_ps = make_ps("bb_ntt_shared_mem_gs");
        let stockham_ps = make_ps("bb_ntt_stockham");
        let stockham_gs_ps = make_ps("bb_ntt_stockham_gs");
        let stockham_global_r2_ps = make_ps("bb_stockham_global_r2");
        let twiddle_transpose_ps = make_ps("bb_ntt_twiddle_transpose");
        let dif_r32_ps = make_ps("bb_dif_r32");
        let dif_r16_ps = make_ps("bb_dif_r16");
        let dif_r8_ps = make_ps("bb_dif_r8");
        let dif_r4_ps = make_ps("bb_dif_r4");
        let dif_r2_ps = make_ps("bb_dif_r2");
        let dif_r32_bitrev_ps = make_ps("bb_dif_r32_bitrev");
        let dif_r16_bitrev_ps = make_ps("bb_dif_r16_bitrev");
        let dif_r8_bitrev_ps = make_ps("bb_dif_r8_bitrev");
        let dif_r4_bitrev_ps = make_ps("bb_dif_r4_bitrev");
        let dif_r2_bitrev_ps = make_ps("bb_dif_r2_bitrev");
        let dif_shared_bitrev_ps = make_ps("bb_dif_shared_bitrev");
        let poseidon2_hash_leaves_ps = make_ps("poseidon2_hash_leaves");
        let poseidon2_compress_ps = make_ps("poseidon2_merkle_compress");
        let poseidon2_pow_grind_ps = make_ps("poseidon2_pow_grind");
        let transpose_pad_ps = make_ps("bb_transpose_pad");

        let pow_tg_size = (poseidon2_pow_grind_ps.max_total_threads_per_threadgroup() as u64)
            .min(1024);
        let opts_shared = MTLResourceOptions::StorageModeShared;
        let r = (1u128 << 32) % (0x7800_0001u128);
        let r_squared = ((r * r) % (0x7800_0001u128)) as u32;
        let pow_bufs = PowBuffers {
            state_buf: device.new_buffer((16 * size_of::<u32>()) as u64, opts_shared),
            params_buf: device.new_buffer((4 * size_of::<u32>()) as u64, opts_shared),
            result_buf: device.new_buffer(size_of::<u32>() as u64, opts_shared),
            found_buf: device.new_buffer(size_of::<u32>() as u64, opts_shared),
            offset_buf: device.new_buffer(size_of::<u32>() as u64, opts_shared),
            r_squared,
            tg_size: pow_tg_size,
        };

        let p2_rc = build_poseidon2_constants(poseidon2_seed);
        let poseidon2_rc_buf = device.new_buffer_with_data(
            p2_rc.as_ptr().cast(),
            (p2_rc.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );

        let max_tg = shared_mem_ps.max_total_threads_per_threadgroup() as u32;
        let max_log_shared_block = max_tg.min(1024).ilog2();
        let max_stockham_tg = stockham_ps.max_total_threads_per_threadgroup() as u32;
        let max_log_stockham_block = max_stockham_tg.min(1024).ilog2() + 2;

        eprintln!("[Metal] max_tg={max_tg} max_log_shared={max_log_shared_block} stockham_tg={max_stockham_tg} max_log_stockham={max_log_stockham_block}");

        Self {
            cpu,
            gpu_min_log_n,
            device,
            queue,
            bitrev_ps,
            bitrev_gather_ps,
            dif_r16_oop_ps,
            butterfly_ps,
            butterfly_r4_ps,
            butterfly_r8_ps,
            shared_mem_ps,
            shared_mem_gs_ps,
            stockham_ps,
            stockham_gs_ps,
            stockham_global_r2_ps,
            twiddle_transpose_ps,
            dif_r32_ps,
            dif_r16_ps,
            dif_r8_ps,
            dif_r4_ps,
            dif_r2_ps,
            dif_r32_bitrev_ps,
            dif_r16_bitrev_ps,
            dif_r8_bitrev_ps,
            dif_r4_bitrev_ps,
            dif_r2_bitrev_ps,
            dif_shared_bitrev_ps,
            max_log_shared_block,
            max_log_stockham_block,
            twiddle_bufs: Mutex::new(HashMap::new()),
            temp_buf_cache: Mutex::new(None),
            data_buf_cache: Mutex::new(None),
            dft_result_cache: Arc::new(Mutex::new(None)),
            poseidon2_hash_leaves_ps,
            poseidon2_compress_ps,
            poseidon2_pow_grind_ps,
            transpose_pad_ps,
            poseidon2_rc_buf,
            pow_bufs: Mutex::new(pow_bufs),
        }
    }

    /// Brute-force search for a PoW witness on GPU using Poseidon2.
    ///
    /// Given the DuplexChallenger state (sponge + input buffer) and PoW difficulty,
    /// launches a GPU kernel that tries candidates 0..P in parallel. Each thread
    /// inserts a candidate at `witness_idx`, applies Poseidon2, and checks if
    /// the low `pow_bits` of state\[7\] (canonical) are all zero.
    ///
    /// Returns `Some(canonical_witness)` on success.
    pub fn gpu_pow_grind(
        &self,
        base_state_monty: &[u32; 16],
        witness_idx: u32,
        pow_bits: u32,
    ) -> Option<u32> {
        const P: u64 = 0x7800_0001;

        let bufs = self.pow_bufs.lock().expect("pow_bufs mutex");

        unsafe {
            std::ptr::copy_nonoverlapping(
                base_state_monty.as_ptr(),
                bufs.state_buf.contents() as *mut u32,
                16,
            );
            *(bufs.params_buf.contents() as *mut [u32; 4]) =
                [witness_idx, pow_bits, bufs.r_squared, 0];
            *(bufs.result_buf.contents() as *mut u32) = 0;
            *(bufs.found_buf.contents() as *mut u32) = 0;
        }

        // Scale batch size with difficulty to amortize dispatch overhead.
        // For high-bit PoW (25+ bits), use larger batches so we need fewer dispatches.
        let batch_size: u64 = if pow_bits >= 24 {
            1 << 24 // 16M nonces
        } else if pow_bits >= 20 {
            1 << 22 // 4M nonces
        } else {
            1 << 20 // 1M nonces
        };
        let tg_size = bufs.tg_size;

        let mut nonce_offset: u64 = 0;
        while nonce_offset < P {
            let this_batch = (P - nonce_offset).min(batch_size);

            unsafe { *(bufs.offset_buf.contents() as *mut u32) = nonce_offset as u32; }

            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.poseidon2_pow_grind_ps);
            enc.set_buffer(0, Some(&bufs.state_buf), 0);
            enc.set_buffer(1, Some(&self.poseidon2_rc_buf), 0);
            enc.set_buffer(2, Some(&bufs.params_buf), 0);
            enc.set_buffer(3, Some(&bufs.params_buf), 4);
            enc.set_buffer(4, Some(&bufs.params_buf), 8);
            enc.set_buffer(5, Some(&bufs.result_buf), 0);
            enc.set_buffer(6, Some(&bufs.found_buf), 0);
            enc.set_buffer(7, Some(&bufs.offset_buf), 0);

            enc.dispatch_threads(
                metal::MTLSize::new(this_batch, 1, 1),
                metal::MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let found_val = unsafe { *(bufs.found_buf.contents() as *const u32) };
            if found_val != 0 {
                let winner = unsafe { *(bufs.result_buf.contents() as *const u32) };
                return Some(winner);
            }

            nonce_offset += this_batch;
        }

        None
    }

    /// Transpose an [in_rows × in_cols] matrix of multi-word elements to
    /// [out_height × in_rows] on GPU. Rows beyond in_cols are zero-filled.
    /// `elem_size` is the number of BabyBear words per logical element
    /// (1 for base field, 4 for quartic extension).
    pub fn gpu_transpose_pad(
        &self,
        data: &[BabyBear],
        in_rows: usize,
        in_cols: usize,
        out_height: usize,
        elem_size: usize,
    ) -> RowMajorMatrix<BabyBear> {
        let out_width_words = in_rows * elem_size;
        let total_out = out_height * out_width_words;

        let opts = MTLResourceOptions::StorageModeShared;
        let src_buf = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            (data.len() * size_of::<u32>()) as u64,
            opts,
        );
        let dst_buf = self.device.new_buffer(
            (total_out * size_of::<u32>()) as u64,
            opts,
        );

        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.transpose_pad_ps);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&dst_buf), 0);
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        set_u32(&enc, 2, in_rows as u32);
        set_u32(&enc, 3, in_cols as u32);
        set_u32(&enc, 4, out_height as u32);
        set_u32(&enc, 5, elem_size as u32);

        let tg = self.transpose_pad_ps.max_total_threads_per_threadgroup() as u64;
        let tg_w = (in_rows as u64).min(16);
        let tg_h = (tg / tg_w).min(64);
        enc.dispatch_threads(
            metal::MTLSize::new(in_rows as u64, out_height as u64, 1),
            metal::MTLSize::new(tg_w, tg_h, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let mut out = vec![BabyBear::ZERO; total_out];
        unsafe {
            std::ptr::copy_nonoverlapping(
                dst_buf.contents() as *const BabyBear,
                out.as_mut_ptr(),
                total_out,
            );
        }
        RowMajorMatrix::new(out, out_width_words)
    }

    fn twiddle_buffer(&self, log_n: u32) -> Buffer {
        let mut map = self.twiddle_bufs.lock().expect("twiddle mutex");
        map.entry(log_n)
            .or_insert_with(|| {
                let tw = precompute_twiddle_monty(log_n);
                let bytes = (tw.len() * size_of::<u32>()) as u64;
                self.device.new_buffer_with_data(
                    tw.as_ptr().cast(),
                    bytes,
                    MTLResourceOptions::CPUCacheModeDefaultCache
                        | MTLResourceOptions::StorageModeShared,
                )
            })
            .clone()
    }

    /// Number of butterfly stages fused in the shared-memory kernel.
    /// Column tiling inside the kernel means this no longer depends on width.
    fn effective_log_block(&self, log_n: u32) -> u32 {
        self.max_log_shared_block.min(log_n)
    }

    fn use_four_step(&self, _log_n: u32) -> bool {
        // The four-step FFT has poor memory coalescing for narrow matrices
        // (width < ~128) because the gather pattern causes each SIMD thread
        // to access a different cache line. The radix-16 DIF approach is
        // better for the typical narrow-width NTTs in the WHIR prover.
        false
    }

    /// Acquire a buffer from the cache or allocate a new one. The buffer
    /// is guaranteed to be at least `bytes` long.
    fn acquire_buf(cache: &Mutex<Option<Buffer>>, device: &metal::DeviceRef, bytes: u64) -> Buffer {
        let opts =
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared;
        let mut slot = cache.lock().expect("buf mutex");
        match slot.take() {
            Some(b) if b.length() >= bytes => b,
            _ => device.new_buffer(bytes, opts),
        }
    }

    fn release_buf(cache: &Mutex<Option<Buffer>>, buf: Buffer) {
        *cache.lock().expect("buf mutex") = Some(buf);
    }

    /// Run a single Poseidon2 permutation on GPU and return the result (for testing).
    #[cfg(test)]
    fn gpu_poseidon2_permute(&self, input: &[BabyBear; 16]) -> [BabyBear; 16] {
        let opts = MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared;
        let data_buf = self.device.new_buffer_with_data(
            input.as_ptr().cast(),
            (16 * size_of::<u32>()) as u64,
            opts,
        );
        let out_buf = self.device.new_buffer((8 * size_of::<u32>()) as u64, opts);

        // We'll use hash_leaves with 1 leaf of width 16 to test the permutation.
        // PaddingFreeSponge: absorbs 8, permutes, absorbs next 8, permutes, squeezes 8.
        // But for a direct permutation test, we need the full 16-element state.
        // Let's use compress instead: left=input[0..8], right=input[8..16], output=perm(input)[0..8].
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.poseidon2_compress_ps);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&self.poseidon2_rc_buf), 0);
            let num_pairs = 1u32;
            enc.set_bytes(3, size_of::<u32>() as u64, (&num_pairs as *const u32).cast());
            enc.dispatch_threads(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });

        // Read only the first 8 elements (truncated permutation output)
        let mut result = [BabyBear::ZERO; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents() as *const BabyBear,
                result.as_mut_ptr(),
                8,
            );
        }
        result
    }

    /// Build a full Merkle tree on GPU from leaf data.
    /// Build the full Merkle tree on GPU in a single command buffer.
    /// Returns all digest layers (layer 0 = leaf hashes, last = root).
    pub fn gpu_merkle_tree(
        &self,
        data_buf: &metal::BufferRef,
        num_leaves: u32,
        leaf_width: u32,
    ) -> Vec<Vec<u32>> {
        let layers = autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let layers = self.encode_merkle_tree(&enc, data_buf, num_leaves, leaf_width);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            layers
        });

        layers
            .iter()
            .map(|buf| {
                let count = buf.length() as usize / size_of::<u32>();
                let mut v = vec![0u32; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.contents() as *const u32,
                        v.as_mut_ptr(),
                        count,
                    );
                }
                v
            })
            .collect()
    }

    /// Encode Poseidon2 Merkle tree dispatches into an existing encoder.
    /// Returns the pre-allocated layer buffers (caller must wait on the
    /// command buffer before reading them).
    fn encode_merkle_tree(
        &self,
        enc: &ComputeCommandEncoderRef,
        data_buf: &metal::BufferRef,
        num_leaves: u32,
        leaf_width: u32,
    ) -> Vec<Buffer> {
        let opts = MTLResourceOptions::CPUCacheModeDefaultCache
            | MTLResourceOptions::StorageModeShared;
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let digest_elems = 8u32;

        let mut layers: Vec<Buffer> = Vec::new();
        let leaf_buf = self.device.new_buffer(
            u64::from(num_leaves) * u64::from(digest_elems) * (size_of::<u32>() as u64),
            opts,
        );
        layers.push(leaf_buf);

        let mut current_count = num_leaves;
        while current_count > 1 {
            let pairs = current_count / 2;
            layers.push(self.device.new_buffer(
                u64::from(pairs) * u64::from(digest_elems) * (size_of::<u32>() as u64),
                opts,
            ));
            current_count = pairs;
        }

        // Leaf hashing
        let max_tg_leaves =
            self.poseidon2_hash_leaves_ps.max_total_threads_per_threadgroup() as u64;
        enc.set_compute_pipeline_state(&self.poseidon2_hash_leaves_ps);
        enc.set_buffer(0, Some(data_buf), 0);
        enc.set_buffer(1, Some(&layers[0]), 0);
        enc.set_buffer(2, Some(&self.poseidon2_rc_buf), 0);
        set_u32(enc, 3, num_leaves);
        set_u32(enc, 4, leaf_width);
        enc.dispatch_threads(
            MTLSize { width: num_leaves as u64, height: 1, depth: 1 },
            MTLSize { width: max_tg_leaves.min(256), height: 1, depth: 1 },
        );

        // Compression layers
        let max_tg_compress =
            self.poseidon2_compress_ps.max_total_threads_per_threadgroup() as u64;
        current_count = num_leaves;
        for layer_idx in 1..layers.len() {
            let pairs = current_count / 2;
            enc.set_compute_pipeline_state(&self.poseidon2_compress_ps);
            enc.set_buffer(0, Some(&layers[layer_idx - 1]), 0);
            enc.set_buffer(1, Some(&layers[layer_idx]), 0);
            enc.set_buffer(2, Some(&self.poseidon2_rc_buf), 0);
            set_u32(enc, 3, pairs);
            enc.dispatch_threads(
                MTLSize { width: pairs as u64, height: 1, depth: 1 },
                MTLSize { width: max_tg_compress.min(256), height: 1, depth: 1 },
            );
            current_count = pairs;
        }

        layers
    }

    /// Convert GPU layer buffers into the digest_layers / cap format.
    fn read_merkle_layers(layers: &[Buffer]) -> (
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        Vec<Vec<[BabyBear; 8]>>,
        Vec<usize>,
    ) {
        let digest_layers: Vec<Vec<[BabyBear; 8]>> = layers
            .iter()
            .map(|buf| {
                let count = buf.length() as usize / size_of::<u32>();
                let mut v = vec![0u32; count];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.contents() as *const u32,
                        v.as_mut_ptr(),
                        count,
                    );
                }
                v.chunks_exact(8)
                    .map(|chunk| {
                        let mut digest = [BabyBear::ZERO; 8];
                        for (i, &val) in chunk.iter().enumerate() {
                            digest[i] = unsafe { std::mem::transmute::<u32, BabyBear>(val) };
                        }
                        digest
                    })
                    .collect()
            })
            .collect();

        let num_layers = digest_layers.len();
        let arity_schedule = vec![2usize; num_layers.saturating_sub(1)];
        let root_digest = digest_layers.last().unwrap()[0];
        let cap = p3_symmetric::MerkleCap::new(vec![root_digest]);
        (cap, digest_layers, arity_schedule)
    }

    /// Fused DFT + Merkle tree: runs NTT and Poseidon2 hashing in a single
    /// GPU command buffer, avoiding the CPU round-trip between them.
    /// Returns `(cap, digest_layers, arity_schedule, output_values)`.
    pub fn gpu_dft_and_merkle(
        &self,
        values: &mut Vec<BabyBear>,
        height: usize,
        width: usize,
    ) -> Option<(
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        Vec<Vec<[BabyBear; 8]>>,
        Vec<usize>,
    )> {
        let log_n = log2_strict_usize(height) as u32;
        if log_n < self.gpu_min_log_n || log_n > Self::GPU_MAX_LOG_N {
            return None;
        }
        let total_bytes_val = height * width * size_of::<u32>();
        if total_bytes_val < 8 * 1024 * 1024 || total_bytes_val > Self::GPU_MAX_TOTAL_BYTES {
            return None;
        }

        let total_bytes = (height * width * size_of::<u32>()) as u64;
        let tw = self.twiddle_buffer(log_n);

        // Managed buffer for DIF stages (fast GPU memory).
        let dif_buf = Self::acquire_buf(&self.data_buf_cache, &self.device, total_bytes);

        // Try zero-copy: wrap values as a Metal buffer so the bitrev gather
        // writes the result directly into `values` (no post-GPU memcpy).
        let zc_input = self.try_zero_copy_buffer(values, total_bytes);

        // If zero-copy works, bitrev writes back to the zc buffer and
        // Merkle hashing reads from it — eliminating the separate
        // natural_buf allocation and the full matrix readback.
        let need_natural_buf = zc_input.is_none();
        let natural_buf = if need_natural_buf {
            Some(Self::acquire_buf(&self.temp_buf_cache, &self.device, total_bytes))
        } else {
            None
        };

        let use_four_step = self.use_four_step(log_n);

        let result = autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            if let Some(ref zc_buf) = zc_input {
                if use_four_step {
                    // Four-step path: zc→temp→zc (natural order), then Merkle
                    let temp = Self::acquire_buf(&self.temp_buf_cache, &self.device, total_bytes);
                    self.encode_four_step_ntt_oop(&enc, zc_buf, zc_buf, &temp, &tw,
                        log_n, height as u32, width as u32);
                    let merkle_layers = self.encode_merkle_tree(
                        &enc, zc_buf, height as u32, width as u32,
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    Self::release_buf(&self.temp_buf_cache, temp);
                    if cmd.status() == MTLCommandBufferStatus::Error {
                        return None;
                    }
                    Some(Self::read_merkle_layers(&merkle_layers))
                } else {
                    // DIF path: partial global stages on managed, then shared-
                    // memory tail + bitrev writes back to zc for Merkle.
                    let shared_lb = log_n.min(self.max_log_shared_block);
                    let global_stages = log_n - shared_lb;
                    if global_stages > 0 {
                        self.encode_dif_stages_inplace(&enc, zc_buf, &dif_buf, &tw,
                            global_stages, height as u32, width as u32);
                        self.dispatch_dif_shared_bitrev(
                            &enc, &dif_buf, zc_buf, &tw,
                            height as u32, width as u32, global_stages, log_n, shared_lb,
                        );
                    } else {
                        // Entire NTT fits in shared memory. Need OOP copy
                        // zc → dif_buf first (shared_bitrev can't work in-place).
                        self.encode_dif_stages_inplace(&enc, zc_buf, &dif_buf, &tw,
                            log_n, height as u32, width as u32);
                        self.encode_bitrev_gather(&enc, &dif_buf, zc_buf,
                            height as u32, width as u32, log_n);
                    }
                    let merkle_layers = self.encode_merkle_tree(
                        &enc, zc_buf, height as u32, width as u32,
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    if cmd.status() == MTLCommandBufferStatus::Error {
                        return None;
                    }
                    Some(Self::read_merkle_layers(&merkle_layers))
                }
            } else {
                let nat = natural_buf.as_ref().unwrap();
                unsafe {
                    fast_memcpy(
                        dif_buf.contents() as *mut u8,
                        values.as_ptr().cast::<u8>(),
                        total_bytes as usize,
                    );
                }
                if use_four_step {
                    // Four-step in-place on dif_buf, then Merkle
                    self.encode_four_step_ntt(&enc, &dif_buf, nat, &tw,
                        log_n, height as u32, width as u32);
                    let merkle_layers = self.encode_merkle_tree(
                        &enc, &dif_buf, height as u32, width as u32,
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    if cmd.status() == MTLCommandBufferStatus::Error {
                        return None;
                    }
                    unsafe {
                        fast_memcpy_from_gpu(
                            values.as_mut_ptr().cast::<u8>(),
                            dif_buf.contents() as *const u8,
                            total_bytes as usize,
                        );
                    }
                    Some(Self::read_merkle_layers(&merkle_layers))
                } else {
                    // DIF+fused bitrev: last DIF stage writes bitrev'd output
                    // directly to nat, eliminating a separate bitrev pass.
                    self.encode_dif_ntt(
                        &enc, &dif_buf, nat, &tw,
                        log_n, height as u32, width as u32,
                    );
                    let merkle_layers = self.encode_merkle_tree(
                        &enc, nat, height as u32, width as u32,
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    if cmd.status() == MTLCommandBufferStatus::Error {
                        return None;
                    }
                    unsafe {
                        fast_memcpy_from_gpu(
                            values.as_mut_ptr().cast::<u8>(),
                            nat.contents() as *const u8,
                            total_bytes as usize,
                        );
                    }
                    Some(Self::read_merkle_layers(&merkle_layers))
                }
            }
        });

        Self::release_buf(&self.data_buf_cache, dif_buf);
        if let Some(buf) = natural_buf {
            Self::release_buf(&self.temp_buf_cache, buf);
        }
        result
    }

    /// Fused GPU transpose+pad → DFT → Merkle without CPU round-trip.
    /// Input is raw polynomial data (untransposed), output is the transposed
    /// DFT result in `out_values` plus Merkle digest layers.
    pub fn gpu_transpose_dft_and_merkle(
        &self,
        data: &[BabyBear],
        in_rows: usize,
        in_cols: usize,
        out_height: usize,
        elem_size: usize,
    ) -> Option<(
        Vec<BabyBear>,
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        Vec<Vec<[BabyBear; 8]>>,
        Vec<usize>,
    )> {
        let out_width = in_rows * elem_size;
        let total_out = out_height * out_width;
        let log_n = log2_strict_usize(out_height) as u32;

        if log_n < self.gpu_min_log_n || log_n > Self::GPU_MAX_LOG_N {
            return None;
        }
        let total_bytes_val = total_out * size_of::<u32>();
        if total_bytes_val < 8 * 1024 * 1024 || total_bytes_val > Self::GPU_MAX_TOTAL_BYTES {
            return None;
        }

        let total_bytes = total_bytes_val as u64;
        let tw = self.twiddle_buffer(log_n);

        let dif_buf = Self::acquire_buf(&self.data_buf_cache, &self.device, total_bytes);
        let natural_buf = Self::acquire_buf(&self.temp_buf_cache, &self.device, total_bytes);

        // Transpose+pad directly into the GPU buffer (shared memory on Apple Silicon).
        // This avoids allocating a separate src buffer and a GPU transpose kernel.
        {
            let dst_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    dif_buf.contents() as *mut BabyBear, total_out,
                )
            };
            dst_slice.fill(BabyBear::ZERO);
            let src_slice = data;
            dst_slice.par_chunks_mut(out_width).take(in_cols).enumerate().for_each(|(out_row, row)| {
                for out_col in 0..in_rows {
                    let src_base = (out_col * in_cols + out_row) * elem_size;
                    let dst_offset = out_col * elem_size;
                    for d in 0..elem_size {
                        row[dst_offset + d] = src_slice[src_base + d];
                    }
                }
            });
        }

        let use_four_step = self.use_four_step(log_n);

        let result = autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            if use_four_step {
                // Four-step: dif_buf → natural_buf (temp) → dif_buf (natural order)
                self.encode_four_step_ntt(&enc, &dif_buf, &natural_buf, &tw,
                    log_n, out_height as u32, out_width as u32);
                let merkle_layers = self.encode_merkle_tree(
                    &enc, &dif_buf, out_height as u32, out_width as u32,
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                if cmd.status() == MTLCommandBufferStatus::Error {
                    return None;
                }
                let mut out_values = vec![BabyBear::ZERO; total_out];
                unsafe {
                    fast_memcpy_from_gpu(
                        out_values.as_mut_ptr().cast::<u8>(),
                        dif_buf.contents() as *const u8,
                        total_bytes as usize,
                    );
                }
                let (cap, digest_layers, arity_schedule) = Self::read_merkle_layers(&merkle_layers);
                Some((out_values, cap, digest_layers, arity_schedule))
            } else {
                // DIF+fused bitrev: last DIF stage writes bitrev'd result
                // directly to natural_buf, eliminating a separate bitrev pass.
                self.encode_dif_ntt(
                    &enc, &dif_buf, &natural_buf, &tw,
                    log_n, out_height as u32, out_width as u32,
                );
                let merkle_layers = self.encode_merkle_tree(
                    &enc, &natural_buf, out_height as u32, out_width as u32,
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                if cmd.status() == MTLCommandBufferStatus::Error {
                    return None;
                }
                let mut out_values = vec![BabyBear::ZERO; total_out];
                unsafe {
                    fast_memcpy_from_gpu(
                        out_values.as_mut_ptr().cast::<u8>(),
                        natural_buf.contents() as *const u8,
                        total_bytes as usize,
                    );
                }
                let (cap, digest_layers, arity_schedule) = Self::read_merkle_layers(&merkle_layers);
                Some((out_values, cap, digest_layers, arity_schedule))
            }
        });

        Self::release_buf(&self.data_buf_cache, dif_buf);
        Self::release_buf(&self.temp_buf_cache, natural_buf);
        result
    }

    /// Encode just the DIF butterfly stages (no bitrev) with zero-copy input.
    /// First R16 is out-of-place (zc→managed), remaining stages in-place on managed.
    fn encode_dif_stages_inplace(
        &self,
        enc: &ComputeCommandEncoderRef,
        zc_buf: &metal::BufferRef,
        data: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let tg_w = width.min(32);
        let mut dif_stage = 0u32;
        let mut first_dispatch = true;

        while dif_stage < log_n {
            let gap = log_n - dif_stage;
            if gap >= 4 {
                let num_units = height >> 4;
                if first_dispatch {
                    let max_tg = self.dif_r16_oop_ps.max_total_threads_per_threadgroup() as u32;
                    enc.set_compute_pipeline_state(&self.dif_r16_oop_ps);
                    enc.set_buffer(0, Some(zc_buf), 0);
                    enc.set_buffer(1, Some(data), 0);
                    enc.set_buffer(2, Some(twiddles), 0);
                    set_u32(enc, 3, height);
                    set_u32(enc, 4, width);
                    set_u32(enc, 5, dif_stage);
                    enc.dispatch_threads(
                        MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                        MTLSize {
                            width: tg_w as u64,
                            height: (max_tg / tg_w).min(num_units) as u64,
                            depth: 1,
                        },
                    );
                    first_dispatch = false;
                } else {
                    let max_tg = self.dif_r16_ps.max_total_threads_per_threadgroup() as u32;
                    enc.set_compute_pipeline_state(&self.dif_r16_ps);
                    enc.set_buffer(0, Some(data), 0);
                    enc.set_buffer(1, Some(twiddles), 0);
                    set_u32(enc, 2, height);
                    set_u32(enc, 3, width);
                    set_u32(enc, 4, dif_stage);
                    enc.dispatch_threads(
                        MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                        MTLSize {
                            width: tg_w as u64,
                            height: (max_tg / tg_w).min(num_units) as u64,
                            depth: 1,
                        },
                    );
                }
                dif_stage += 4;
            } else if gap >= 3 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r8_ps, 3);
                dif_stage += 3;
            } else if gap == 2 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r4_ps, 2);
                dif_stage += 2;
            } else {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r2_ps, 1);
                dif_stage += 1;
            }
        }
    }

    /// Encode DIF butterfly stages in-place on a managed buffer (no zero-copy input).
    fn encode_dif_stages_only(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let tg_w = width.min(32);
        let mut dif_stage = 0u32;

        while dif_stage < log_n {
            let gap = log_n - dif_stage;
            if gap >= 4 {
                let num_units = height >> 4;
                let max_tg = self.dif_r16_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r16_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 4;
            } else if gap >= 3 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r8_ps, 3);
                dif_stage += 3;
            } else if gap == 2 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r4_ps, 2);
                dif_stage += 2;
            } else {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r2_ps, 1);
                dif_stage += 1;
            }
        }
    }

    /// Helper: dispatch a single in-place DIF radix stage.
    fn dispatch_dif_inplace(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        dif_stage: u32,
        tg_w: u32,
        ps: &ComputePipelineState,
        log_radix: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let num_units = height >> log_radix;
        let max_tg = ps.max_total_threads_per_threadgroup() as u32;
        enc.set_compute_pipeline_state(ps);
        enc.set_buffer(0, Some(data), 0);
        enc.set_buffer(1, Some(twiddles), 0);
        set_u32(enc, 2, height);
        set_u32(enc, 3, width);
        set_u32(enc, 4, dif_stage);
        enc.dispatch_threads(
            MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
            MTLSize {
                width: tg_w as u64,
                height: (max_tg / tg_w).min(num_units) as u64,
                depth: 1,
            },
        );
    }

    /// Dispatch the shared-memory DIF + bitrev kernel for the tail stages.
    /// Handles `log_block` DIF stages in shared memory and writes output
    /// in natural (bit-reversed) order. Out-of-place: src → dst.
    fn dispatch_dif_shared_bitrev(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &metal::BufferRef,
        dst: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        start_stage: u32,
        log_n: u32,
        log_block: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let block_size = 1u32 << log_block;
        let half_block = block_size >> 1;
        let num_blocks = height >> log_block;

        enc.set_compute_pipeline_state(&self.dif_shared_bitrev_ps);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        enc.set_buffer(2, Some(twiddles), 0);
        set_u32(enc, 3, height);
        set_u32(enc, 4, width);
        set_u32(enc, 5, start_stage);
        set_u32(enc, 6, log_n);
        set_u32(enc, 7, log_block);
        enc.dispatch_thread_groups(
            MTLSize { width: width as u64, height: num_blocks as u64, depth: 1 },
            MTLSize { width: half_block as u64, height: 1, depth: 1 },
        );
    }

    /// Encode a bitrev gather: src[bitrev(row)] → dst[row] (coalesced writes).
    fn encode_bitrev_gather(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &metal::BufferRef,
        dst: &metal::BufferRef,
        height: u32,
        width: u32,
        log_n: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let tg_w = width.min(32);
        let max_tg = self.bitrev_gather_ps.max_total_threads_per_threadgroup() as u32;
        enc.set_compute_pipeline_state(&self.bitrev_gather_ps);
        enc.set_buffer(0, Some(src), 0);
        enc.set_buffer(1, Some(dst), 0);
        set_u32(enc, 2, height);
        set_u32(enc, 3, width);
        set_u32(enc, 4, log_n);
        enc.dispatch_threads(
            MTLSize { width: width as u64, height: height as u64, depth: 1 },
            MTLSize {
                width: tg_w as u64,
                height: (max_tg / tg_w).min(height) as u64,
                depth: 1,
            },
        );
    }

    /// Build Merkle digest layers on GPU from a raw BabyBear slice.
    /// The slice must contain `height * width` elements in row-major order.
    pub fn gpu_build_merkle_digests_raw(
        &self,
        data: &[BabyBear],
        height: usize,
        width: usize,
    ) -> (
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        Vec<Vec<[BabyBear; 8]>>,
        Vec<usize>,
    ) {
        let total_bytes = (height * width * size_of::<u32>()) as u64;
        let opts = MTLResourceOptions::CPUCacheModeDefaultCache
            | MTLResourceOptions::StorageModeShared;
        let data_buf = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            total_bytes,
            opts,
        );

        let layers = autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let layers = self.encode_merkle_tree(
                &enc, &data_buf, height as u32, width as u32,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            layers
        });

        Self::read_merkle_layers(&layers)
    }

    /// Build Merkle digest layers on GPU from a row-major matrix.
    pub fn gpu_build_merkle_digests(
        &self,
        mat: &RowMajorMatrix<BabyBear>,
    ) -> (
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        Vec<Vec<[BabyBear; 8]>>,
        Vec<usize>,
    ) {
        self.gpu_build_merkle_digests_raw(&mat.values, mat.height(), mat.width())
    }

    /// Commit a matrix using GPU Merkle: hash all rows, build compression tree.
    /// Returns `(MerkleCap, MerkleTree)` compatible with plonky3's `MerkleTreeMmcs`.
    pub fn gpu_commit_matrix(
        &self,
        mat: RowMajorMatrix<BabyBear>,
    ) -> (
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        p3_merkle_tree::MerkleTree<BabyBear, BabyBear, RowMajorMatrix<BabyBear>, 2, 8>,
    ) {
        let (cap, digest_layers, arity_schedule) = self.gpu_build_merkle_digests(&mat);
        let tree = p3_merkle_tree::MerkleTree::from_parts(
            vec![mat],
            digest_layers,
            arity_schedule,
        );
        (cap, tree)
    }

    /// Try to wrap the Vec's memory directly as a Metal buffer (zero-copy).
    /// Requires page-aligned pointer and page-aligned size.
    fn try_zero_copy_buffer(&self, values: &mut Vec<BabyBear>, total_bytes: u64) -> Option<Buffer> {
        const PAGE_SIZE: usize = 16384;
        let ptr = values.as_mut_ptr() as usize;
        let len = total_bytes as usize;
        if ptr % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
            return None;
        }
        let buf = self.device.new_buffer_with_bytes_no_copy(
            values.as_mut_ptr().cast(),
            total_bytes,
            MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared,
            None,
        );
        Some(buf)
    }

    fn try_gpu_dft_inplace(&self, values: &mut Vec<BabyBear>, height: usize, width: usize) -> Option<()> {
        let log_n = log2_strict_usize(height) as u32;
        if log_n < self.gpu_min_log_n || log_n > Self::GPU_MAX_LOG_N {
            return None;
        }
        let total_bytes_val = height * width * size_of::<u32>();
        if total_bytes_val < 8 * 1024 * 1024 || total_bytes_val > Self::GPU_MAX_TOTAL_BYTES {
            return None;
        }

        let total_bytes = (height * width * size_of::<u32>()) as u64;
        let tw = self.twiddle_buffer(log_n);
        let use_four_step = self.use_four_step(log_n);

        // Try zero-copy: wrap values directly as a Metal buffer.
        if let Some(zc_buf) = self.try_zero_copy_buffer(values, total_bytes) {
            let data_buf = Self::acquire_buf(&self.data_buf_cache, &self.device, total_bytes);
            let ok = autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                if use_four_step {
                    // Four-step: zc → temp → zc (natural order)
                    self.encode_four_step_ntt_oop(&enc, &zc_buf, &zc_buf, &data_buf, &tw,
                        log_n, height as u32, width as u32);
                } else {
                    self.encode_dif_ntt_zc(
                        &enc, &zc_buf, &data_buf, &zc_buf, &tw,
                        log_n, height as u32, width as u32,
                    );
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                cmd.status() != MTLCommandBufferStatus::Error
            });
            Self::release_buf(&self.data_buf_cache, data_buf);
            if ok { return Some(()); }
        }

        // Fallback: managed buffers with memcpy both ways.
        let data_buf = Self::acquire_buf(&self.data_buf_cache, &self.device, total_bytes);
        unsafe {
            fast_memcpy(
                data_buf.contents() as *mut u8,
                values.as_ptr().cast::<u8>(),
                total_bytes as usize,
            );
        }
        let dst_buf = Self::acquire_buf(&self.temp_buf_cache, &self.device, total_bytes);

        let ok = autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            if use_four_step {
                // Four-step in-place on data_buf, using dst_buf as temp
                self.encode_four_step_ntt(&enc, &data_buf, &dst_buf, &tw,
                    log_n, height as u32, width as u32);
            } else {
                self.encode_dif_ntt(
                    &enc, &data_buf, &dst_buf, &tw,
                    log_n, height as u32, width as u32,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            cmd.status() != MTLCommandBufferStatus::Error
        });

        if !ok {
            Self::release_buf(&self.data_buf_cache, data_buf);
            Self::release_buf(&self.temp_buf_cache, dst_buf);
            return None;
        }

        // Four-step result is in data_buf; DIF result is in dst_buf
        let src_buf = if use_four_step { &data_buf } else { &dst_buf };
        unsafe {
            fast_memcpy_from_gpu(
                values.as_mut_ptr().cast::<u8>(),
                src_buf.contents() as *const u8,
                total_bytes as usize,
            );
        }

        Self::release_buf(&self.data_buf_cache, data_buf);
        Self::release_buf(&self.temp_buf_cache, dst_buf);
        Some(())
    }

    /// Dispatch the Stockham radix-4 kernel.
    /// Computes a `block_size`-point NTT on each contiguous block of `block_size` rows.
    /// No bit-reversal dispatch needed — Stockham autosort handles ordering.
    fn dispatch_stockham(
        &self,
        enc: &ComputeCommandEncoderRef,
        buf: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        log_block: u32,
    ) {
        let block_size = 1u32 << log_block;
        let threads = block_size >> 2; // radix-4: 4 elements per thread
        let num_blocks = height / block_size;
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        enc.set_compute_pipeline_state(&self.stockham_ps);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(twiddles), 0);
        set_u32(enc, 2, height);
        set_u32(enc, 3, width);
        set_u32(enc, 4, log_block);

        enc.dispatch_thread_groups(
            MTLSize { width: num_blocks as u64, height: 1, depth: 1 },
            MTLSize { width: threads as u64, height: 1, depth: 1 },
        );
    }

    /// Dispatch a shared-memory kernel on `buf`, with optional bit-reversal.
    fn dispatch_shared_mem(
        &self,
        enc: &ComputeCommandEncoderRef,
        buf: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        log_block: u32,
        do_bitrev: u32,
    ) {
        let block_size = 1u32 << log_block;
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        enc.set_compute_pipeline_state(&self.shared_mem_ps);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(twiddles), 0);
        set_u32(enc, 2, height);
        set_u32(enc, 3, width);
        set_u32(enc, 4, log_block);
        set_u32(enc, 5, do_bitrev);

        let num_blocks = height / block_size;
        enc.dispatch_thread_groups(
            MTLSize { width: num_blocks as u64, height: 1, depth: 1 },
            MTLSize { width: block_size as u64, height: 1, depth: 1 },
        );
    }

    /// Dispatch the gather-scatter shared-memory DFT kernel.
    fn dispatch_shared_mem_gs(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        output: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        log_block: u32,
        load_stride: u32,
        store_stride: u32,
        apply_twiddle: u32,
    ) {
        let block_size = 1u32 << log_block;
        let num_blocks = height / block_size;
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        enc.set_compute_pipeline_state(&self.shared_mem_gs_ps);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_buffer(2, Some(twiddles), 0);
        set_u32(enc, 3, height);
        set_u32(enc, 4, width);
        set_u32(enc, 5, log_block);
        set_u32(enc, 6, load_stride);
        set_u32(enc, 7, store_stride);
        set_u32(enc, 8, apply_twiddle);

        enc.dispatch_thread_groups(
            MTLSize { width: num_blocks as u64, height: 1, depth: 1 },
            MTLSize { width: block_size as u64, height: 1, depth: 1 },
        );
    }

    /// Dispatch the Stockham radix-4 gather-scatter kernel.
    /// Handles up to 4096-point sub-DFTs (4 elements per thread, 1024 threads).
    fn dispatch_stockham_gs(
        &self,
        enc: &ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        output: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        height: u32,
        width: u32,
        log_block: u32,
        load_stride: u32,
        store_stride: u32,
        apply_twiddle: u32,
    ) {
        let block_size = 1u32 << log_block;
        let threads = block_size >> 2; // radix-4: 4 elements per thread
        let num_blocks = height / block_size;
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        enc.set_compute_pipeline_state(&self.stockham_gs_ps);
        enc.set_buffer(0, Some(input), 0);
        enc.set_buffer(1, Some(output), 0);
        enc.set_buffer(2, Some(twiddles), 0);
        set_u32(enc, 3, height);
        set_u32(enc, 4, width);
        set_u32(enc, 5, log_block);
        set_u32(enc, 6, load_stride);
        set_u32(enc, 7, store_stride);
        set_u32(enc, 8, apply_twiddle);

        enc.dispatch_thread_groups(
            MTLSize { width: num_blocks as u64, height: 1, depth: 1 },
            MTLSize { width: threads as u64, height: 1, depth: 1 },
        );
    }

    /// Pure Stockham NTT: shared-mem Stockham for first 12 stages, then
    /// Stockham global R2 for remaining stages with ping-pong buffers.
    /// No bit-reversal needed. All memory accesses are coalesced.
    fn encode_stockham_full(
        &self,
        enc: &ComputeCommandEncoderRef,
        data_buf: &metal::BufferRef,
        temp_buf: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let log_block = self.max_log_stockham_block.min(log_n);

        // Phase 1: Stockham shared-mem for first log_block stages (in-place on data_buf)
        self.dispatch_stockham(enc, data_buf, twiddles, height, width, log_block);

        // Phase 2: Stockham global R2 for remaining stages (ping-pong)
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let half_n = height >> 1;
        let max_tg = self.stockham_global_r2_ps.max_total_threads_per_threadgroup() as u32;
        let tg_w = width.min(32);

        let mut src: &metal::BufferRef = data_buf;
        let mut dst: &metal::BufferRef = temp_buf;

        for stage in log_block..log_n {
            let p = 1u32 << stage;
            let src_res: &metal::ResourceRef = src.deref();
            enc.memory_barrier_with_resources(&[src_res]);

            enc.set_compute_pipeline_state(&self.stockham_global_r2_ps);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_buffer(2, Some(twiddles), 0);
            set_u32(enc, 3, height);
            set_u32(enc, 4, width);
            set_u32(enc, 5, p);

            enc.dispatch_threads(
                MTLSize { width: width as u64, height: half_n as u64, depth: 1 },
                MTLSize {
                    width: tg_w as u64,
                    height: (max_tg / tg_w).min(half_n) as u64,
                    depth: 1,
                },
            );

            std::mem::swap(&mut src, &mut dst);
        }
    }

    /// Four-step FFT: 2 dispatches with balanced split and fused twiddle.
    ///
    /// Cooley-Tukey decomposition N = N1 × N2 (balanced: N1 ≈ N2 ≈ √N):
    ///   1. Column DFTs: stride-N2 gather → N1-point DFT → contiguous store
    ///      data_buf → temp_buf
    ///   2. Twiddle + Row DFTs: stride-N1 gather from temp (with fused
    ///      twiddle ω_N^{k1·n2}) → N2-point DFT → stride-N1 scatter
    ///      temp_buf → data_buf   (scatter produces natural output order)
    /// Result lives in `data_buf`.
    fn encode_four_step_ntt(
        &self,
        enc: &ComputeCommandEncoderRef,
        data_buf: &metal::BufferRef,
        temp_buf: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        self.encode_four_step_ntt_buffers(enc, data_buf, data_buf, temp_buf, twiddles,
            log_n, height, width);
    }

    /// Four-step FFT out-of-place: reads from `src`, writes natural-order
    /// result to `dst`, using `temp` as scratch.
    fn encode_four_step_ntt_oop(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &metal::BufferRef,
        dst: &metal::BufferRef,
        temp: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        self.encode_four_step_ntt_buffers(enc, src, dst, temp, twiddles,
            log_n, height, width);
    }

    /// Core four-step implementation: reads from `src`, writes to `dst`.
    /// Uses Stockham radix-4 gather-scatter kernel for sub-DFTs up to 4096
    /// points (log_block ≤ 12), covering NTTs up to 2^24.
    fn encode_four_step_ntt_buffers(
        &self,
        enc: &ComputeCommandEncoderRef,
        src: &metal::BufferRef,
        dst: &metal::BufferRef,
        temp: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let log_n1 = log_n / 2;
        let log_n2 = log_n - log_n1;
        let n1 = 1u32 << log_n1;
        let n2 = 1u32 << log_n2;

        // Step 1: N2 groups of N1-point column DFTs.
        // Gather from src (stride N2), contiguous store to temp.
        self.dispatch_stockham_gs(
            enc, src, temp, twiddles,
            height, width, log_n1,
            n2, // load_stride = N2 (gather columns)
            1,  // store_stride = 1 (contiguous blocks)
            0,  // no twiddle
        );

        // Step 2: N1 groups of N2-point row DFTs with fused twiddle.
        // Gather from temp (stride N1) with twiddle ω_N^{k1·n2},
        // scatter-store to dst (stride N1) → natural output order.
        {
            let res: &metal::ResourceRef = temp.deref();
            enc.memory_barrier_with_resources(&[res]);
            self.dispatch_stockham_gs(
                enc, temp, dst, twiddles,
                height, width, log_n2,
                n1, // load_stride = N1 (gather rows from column-DFT output)
                n1, // store_stride = N1 (scatter → natural order)
                1,  // apply twiddle ω_N^{tgid·tid}
            );
        }
    }

    /// DIF NTT using R16/R8/R4/R2 kernels.
    /// The LAST dispatch fuses bit-reversal into its write (out-of-place).
    /// `dst_buf` receives the natural-order result.
    fn encode_dif_ntt(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        dst: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        let shared_log_block = log_n.min(self.max_log_shared_block);

        if shared_log_block == log_n {
            self.dispatch_dif_shared_bitrev(
                enc, data, dst, twiddles, height, width, 0, log_n, log_n,
            );
            return;
        }

        let global_stages = log_n - shared_log_block;
        let tg_w = width.min(32);
        let mut dif_stage = 0u32;

        while dif_stage < global_stages {
            let gap = global_stages - dif_stage;
            if gap >= 4 {
                let num_units = height >> 4;
                let max_tg = self.dif_r16_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r16_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 4;
            } else if gap >= 3 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r8_ps, 3);
                dif_stage += 3;
            } else if gap == 2 {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r4_ps, 2);
                dif_stage += 2;
            } else {
                self.dispatch_dif_inplace(enc, data, twiddles, height, width, dif_stage, tg_w, &self.dif_r2_ps, 1);
                dif_stage += 1;
            }
        }

        // Tail: shared-memory DIF + bitrev (out-of-place: data → dst)
        self.dispatch_dif_shared_bitrev(
            enc, data, dst, twiddles, height, width,
            dif_stage, log_n, shared_log_block,
        );
    }

    /// DIF NTT with zero-copy roundtrip: first R16 dispatch reads from
    /// `zc_buf` and writes to `data` (out-of-place). Remaining stages
    /// run in-place on `data` (fast managed memory). Final bitrev gather
    /// writes back to `zc_buf` with coalesced sequential writes.
    fn encode_dif_ntt_zc(
        &self,
        enc: &ComputeCommandEncoderRef,
        zc_buf: &metal::BufferRef,
        data: &metal::BufferRef,
        _dst_unused: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let shared_lb = log_n.min(self.max_log_shared_block);
        let global_stages = log_n - shared_lb;

        if global_stages > 0 {
            // Partial DIF stages: zc→managed (first OOP), then in-place.
            self.encode_dif_stages_inplace(enc, zc_buf, data, twiddles,
                global_stages, height, width);
            // Shared-memory tail + bitrev: data → zc_buf (natural order).
            self.dispatch_dif_shared_bitrev(
                enc, data, zc_buf, twiddles,
                height, width, global_stages, log_n, shared_lb,
            );
        } else {
            // Entire NTT fits in shared memory. Copy zc→data first.
            self.encode_dif_stages_inplace(enc, zc_buf, data, twiddles,
                log_n, height, width);
            self.encode_bitrev_gather(enc, data, zc_buf, height, width, log_n);
        }
    }

    /// DIF NTT with all stages in-place on `data`, then bitrev gather copy
    /// from `data` to `output`. Reads from `data` are random (bitrev pattern,
    /// but cached in GPU memory); writes to `output` are sequential/coalesced.
    fn encode_dif_ntt_gather(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        output: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };
        let tg_w = width.min(32);

        // All DIF stages in-place on `data` (no bitrev fusion).
        let mut dif_stage = 0u32;
        while dif_stage < log_n {
            let gap = log_n - dif_stage;
            if gap >= 4 {
                let num_units = height >> 4;
                let max_tg = self.dif_r16_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r16_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 4;
            } else if gap >= 3 {
                let num_units = height >> 3;
                let max_tg = self.dif_r8_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r8_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 3;
            } else if gap == 2 {
                let num_units = height >> 2;
                let max_tg = self.dif_r4_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r4_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 2;
            } else {
                let num_butterflies = height >> 1;
                let max_tg = self.dif_r2_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.dif_r2_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, dif_stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_butterflies as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_butterflies) as u64,
                        depth: 1,
                    },
                );
                dif_stage += 1;
            }
        }

        // Bitrev gather: data[bitrev(row)] → output[row] (coalesced writes).
        enc.set_compute_pipeline_state(&self.bitrev_gather_ps);
        enc.set_buffer(0, Some(data), 0);
        enc.set_buffer(1, Some(output), 0);
        set_u32(enc, 2, height);
        set_u32(enc, 3, width);
        set_u32(enc, 4, log_n);
        {
            let max_tg = self.bitrev_gather_ps.max_total_threads_per_threadgroup() as u32;
            enc.dispatch_threads(
                MTLSize { width: width as u64, height: height as u64, depth: 1 },
                MTLSize {
                    width: tg_w as u64,
                    height: (max_tg / tg_w).min(height) as u64,
                    depth: 1,
                },
            );
        }
    }

    /// Classic NTT: bitrev → shared-mem DIT stages → global R8/R4/R2.
    ///
    /// On Apple GPUs, dispatches within a single compute command encoder
    /// are ordered automatically — explicit memory barriers are not needed.
    fn encode_classic_ntt(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let log_block = self.effective_log_block(log_n);

        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        // ── Step 1: bit-reverse permutation ──
        enc.set_compute_pipeline_state(&self.bitrev_ps);
        enc.set_buffer(0, Some(data), 0);
        set_u32(enc, 1, height);
        set_u32(enc, 2, width);
        set_u32(enc, 3, log_n);
        {
            let grid = MTLSize { width: width as u64, height: height as u64, depth: 1 };
            let tg = MTLSize {
                width: width.min(32) as u64,
                height: (self.bitrev_ps.max_total_threads_per_threadgroup() as u32
                    / width.min(32))
                .min(height) as u64,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }

        // ── Step 2: fused stages in shared memory (DIT, already bit-reversed) ──
        self.dispatch_shared_mem(enc, data, twiddles, height, width, log_block, 0);

        // ── Step 3: remaining global-memory butterfly stages (R8 > R4 > R2) ──
        if log_block < log_n {
            let remaining = log_n - log_block;
            let num_r8 = remaining / 3;
            let leftover = remaining % 3;
            let tg_w = width.min(32);
            let mut stage = log_block;

            if num_r8 > 0 {
                let num_units = height >> 3;
                let max_tg = self.butterfly_r8_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.butterfly_r8_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                for _ in 0..num_r8 {
                    set_u32(enc, 4, stage);
                    enc.dispatch_threads(
                        MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                        MTLSize {
                            width: tg_w as u64,
                            height: (max_tg / tg_w).min(num_units) as u64,
                            depth: 1,
                        },
                    );
                    stage += 3;
                }
            }

            if leftover == 2 {
                let num_units = height >> 2;
                let max_tg = self.butterfly_r4_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.butterfly_r4_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_units) as u64,
                        depth: 1,
                    },
                );
                stage += 2;
            }

            if leftover == 1 {
                let num_butterflies = height >> 1;
                let max_tg = self.butterfly_ps.max_total_threads_per_threadgroup() as u32;
                enc.set_compute_pipeline_state(&self.butterfly_ps);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, stage);
                enc.dispatch_threads(
                    MTLSize { width: width as u64, height: num_butterflies as u64, depth: 1 },
                    MTLSize {
                        width: tg_w as u64,
                        height: (max_tg / tg_w).min(num_butterflies) as u64,
                        depth: 1,
                    },
                );
                stage += 1;
            }

            debug_assert_eq!(stage, log_n);
        }
    }

    /// Extract `col_count` columns starting at `col_start` from a row-major matrix.
    fn extract_columns(
        values: &[BabyBear], height: usize, full_width: usize,
        col_start: usize, col_count: usize,
    ) -> Vec<BabyBear> {
        let mut out = vec![BabyBear::ZERO; height * col_count];
        out.par_chunks_mut(col_count)
            .enumerate()
            .for_each(|(r, dst)| {
                let base = r * full_width + col_start;
                dst.copy_from_slice(&values[base..base + col_count]);
            });
        out
    }

    /// Merge two column groups back into a full-width row-major matrix.
    fn merge_columns(
        values: &mut [BabyBear],
        left: &[BabyBear], right: &[BabyBear],
        height: usize, full_width: usize,
        left_width: usize, right_width: usize,
    ) {
        values.par_chunks_mut(full_width)
            .enumerate()
            .for_each(|(r, row)| {
                row[..left_width].copy_from_slice(&left[r * left_width..(r + 1) * left_width]);
                row[left_width..].copy_from_slice(&right[r * right_width..(r + 1) * right_width]);
            });
    }
}

impl TwoAdicSubgroupDft<BabyBear> for MetalBabyBearDft {
    type Evaluations = RowMajorMatrix<BabyBear>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<BabyBear>) -> Self::Evaluations {
        let height = mat.height();
        let width = mat.width();
        if self.try_gpu_dft_inplace(&mut mat.values, height, width).is_some() {
            return mat;
        }
        self.cpu.dft_batch(mat)
    }

    fn idft_batch(&self, mat: RowMajorMatrix<BabyBear>) -> RowMajorMatrix<BabyBear> {
        self.cpu.idft_batch(mat)
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<BabyBear>,
        added_bits: usize,
        shift: BabyBear,
    ) -> Self::Evaluations {
        // Delegate to the CPU's fused IDFT+DFT which is significantly faster
        // than doing them as separate passes (cache-friendly, single traversal).
        self.cpu.coset_lde_batch(mat, added_bits, shift)
    }
}

use p3_commit::{BatchOpening, BatchOpeningRef};
use p3_field::PackedValue;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

/// GPU-accelerated MMCS that uses Metal Poseidon2 for Merkle tree construction.
/// Delegates `open_batch` and `verify_batch` to the inner `MerkleTreeMmcs`.
pub struct GpuMmcs<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> {
    inner: MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>,
    gpu: MetalBabyBearDft,
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> std::fmt::Debug
    for GpuMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMmcs").finish_non_exhaustive()
    }
}

impl<P: Copy, PW: Copy, H: Clone, C: Clone, const N: usize, const DIGEST_ELEMS: usize> Clone
    for GpuMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            gpu: self.gpu.clone(),
        }
    }
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    GpuMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
    pub fn new(
        inner: MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>,
        gpu: MetalBabyBearDft,
    ) -> Self {
        Self { inner, gpu }
    }

}

impl<P, PW, H, C> GpuMmcs<P, PW, H, C, 2, 8> {
    /// Fused DFT + Merkle commit in a single GPU command buffer.
    /// Runs NTT, bitrev, Poseidon2 leaf hashing, and all compression layers
    /// without any CPU round-trip between DFT and Merkle.
    pub fn dft_and_commit_matrix(
        &self,
        mut mat: RowMajorMatrix<BabyBear>,
    ) -> Option<(
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        p3_merkle_tree::MerkleTree<BabyBear, BabyBear, RowMajorMatrix<BabyBear>, 2, 8>,
    )> {
        let height = mat.height();
        let width = mat.width();

        let (cap, digest_layers, arity_schedule) =
            self.gpu.gpu_dft_and_merkle(&mut mat.values, height, width)?;

        let tree = p3_merkle_tree::MerkleTree::from_parts(
            vec![mat],
            digest_layers,
            arity_schedule,
        );

        Some((cap, tree))
    }

    /// Fused extension-field DFT + Merkle commit.
    pub fn dft_algebra_and_commit_matrix<EF>(
        &self,
        mat: RowMajorMatrix<EF>,
    ) -> Option<(
        p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>,
        p3_merkle_tree::MerkleTree<
            BabyBear,
            BabyBear,
            p3_matrix::extension::FlatMatrixView<BabyBear, EF, RowMajorMatrix<EF>>,
            2,
            8,
        >,
    )>
    where
        EF: p3_field::ExtensionField<BabyBear> + BasedVectorSpace<BabyBear> + Clone + Send + Sync,
    {
        let init_width = mat.width();
        let height = mat.height();
        let mut base_values = EF::flatten_to_base(mat.values);
        let base_width = init_width * EF::DIMENSION;

        let (cap, digest_layers, arity_schedule) =
            self.gpu.gpu_dft_and_merkle(&mut base_values, height, base_width)?;

        let ef_values = EF::reconstitute_from_base(base_values);
        let ef_mat = RowMajorMatrix::new(ef_values, init_width);
        let flat_view = p3_matrix::extension::FlatMatrixView::new(ef_mat);

        let tree = p3_merkle_tree::MerkleTree::from_parts(
            vec![flat_view],
            digest_layers,
            arity_schedule,
        );

        Some((cap, tree))
    }
}

impl<P, PW, H, C> Mmcs<BabyBear> for GpuMmcs<P, PW, H, C, 2, 8>
where
    P: PackedValue<Value = BabyBear>,
    PW: PackedValue<Value = BabyBear>,
    H: CryptographicHasher<BabyBear, [BabyBear; 8]>
        + CryptographicHasher<P, [PW; 8]>
        + Sync,
    C: PseudoCompressionFunction<[BabyBear; 8], 2>
        + PseudoCompressionFunction<[PW; 8], 2>
        + Sync,
{
    type ProverData<M> =
        p3_merkle_tree::MerkleTree<BabyBear, BabyBear, M, 2, 8>;
    type Commitment = p3_symmetric::MerkleCap<BabyBear, [BabyBear; 8]>;
    type Proof = Vec<[BabyBear; 8]>;
    type Error = p3_merkle_tree::MerkleTreeError;

    fn commit<M: p3_matrix::Matrix<BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        if inputs.len() == 1 {
            let height = inputs[0].height();
            let width = inputs[0].width();
            let total_bytes = height * width * size_of::<u32>();
            if total_bytes >= 8 * 1024 * 1024 {
                let is_contiguous = height >= 2 && {
                    let s0 = inputs[0].row_slice(0).unwrap();
                    let s1 = inputs[0].row_slice(1).unwrap();
                    let p0 = (*s0).as_ptr();
                    let p1 = (*s1).as_ptr();
                    unsafe { p0.add(width) == p1 }
                };

                let (cap, digest_layers, arity_schedule) = if is_contiguous {
                    let s0 = inputs[0].row_slice(0).unwrap();
                    let ptr = (*s0).as_ptr();
                    let total = height * width;
                    let slice = unsafe { std::slice::from_raw_parts(ptr, total) };
                    self.gpu.gpu_build_merkle_digests_raw(slice, height, width)
                } else {
                    let mut flat = vec![BabyBear::ZERO; height * width];
                    flat.par_chunks_mut(width)
                        .enumerate()
                        .for_each(|(r, dst)| {
                            for (i, v) in inputs[0].row(r).unwrap().into_iter().enumerate() {
                                dst[i] = v;
                            }
                        });
                    let rm = RowMajorMatrix::new(flat, width);
                    self.gpu.gpu_build_merkle_digests(&rm)
                };

                let tree = p3_merkle_tree::MerkleTree::from_parts(
                    inputs,
                    digest_layers,
                    arity_schedule,
                );
                return (cap, tree);
            }
        }
        self.inner.commit(inputs)
    }

    fn commit_matrix<M: p3_matrix::Matrix<BabyBear>>(
        &self,
        input: M,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        self.commit(vec![input])
    }

    fn open_batch<M: p3_matrix::Matrix<BabyBear>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<BabyBear, Self> {
        let inner_opening = self.inner.open_batch(index, prover_data);
        BatchOpening::new(inner_opening.opened_values, inner_opening.opening_proof)
    }

    fn get_matrices<'a, M: p3_matrix::Matrix<BabyBear>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        self.inner.get_matrices(prover_data)
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[p3_matrix::Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, BabyBear, Self>,
    ) -> Result<(), Self::Error> {
        let inner_ref = BatchOpeningRef::new(
            batch_opening.opened_values,
            batch_opening.opening_proof,
        );
        self.inner.verify_batch(commit, dimensions, index, inner_ref)
    }
}

impl<P, PW, H, C> DftCommitFusion<BabyBear> for GpuMmcs<P, PW, H, C, 2, 8>
where
    P: PackedValue<Value = BabyBear>,
    PW: PackedValue<Value = BabyBear>,
    H: CryptographicHasher<BabyBear, [BabyBear; 8]>
        + CryptographicHasher<P, [PW; 8]>
        + Sync,
    C: PseudoCompressionFunction<[BabyBear; 8], 2>
        + PseudoCompressionFunction<[PW; 8], 2>
        + Sync,
{
    fn dft_and_commit(
        &self,
        mut mat: RowMajorMatrix<BabyBear>,
    ) -> Result<
        (Self::Commitment, Self::ProverData<RowMajorMatrix<BabyBear>>),
        RowMajorMatrix<BabyBear>,
    > {
        let height = mat.height();
        let width = mat.width();
        match self.gpu.gpu_dft_and_merkle(&mut mat.values, height, width) {
            Some((cap, digest_layers, arity_schedule)) => {
                let tree = p3_merkle_tree::MerkleTree::from_parts(
                    vec![mat], digest_layers, arity_schedule,
                );
                Ok((cap, tree))
            }
            None => Err(mat),
        }
    }

    fn dft_algebra_and_commit<EF>(
        &self,
        mat: RowMajorMatrix<EF>,
    ) -> Result<
        (
            Self::Commitment,
            Self::ProverData<p3_matrix::extension::FlatMatrixView<BabyBear, EF, RowMajorMatrix<EF>>>,
        ),
        RowMajorMatrix<EF>,
    >
    where
        EF: p3_field::ExtensionField<BabyBear> + BasedVectorSpace<BabyBear> + Clone + Send + Sync,
    {
        let init_width = mat.width();
        let height = mat.height();
        let mut base_values = EF::flatten_to_base(mat.values);
        let base_width = init_width * EF::DIMENSION;

        match self.gpu.gpu_dft_and_merkle(&mut base_values, height, base_width) {
            Some((cap, digest_layers, arity_schedule)) => {
                let ef_values = EF::reconstitute_from_base(base_values);
                let ef_mat = RowMajorMatrix::new(ef_values, init_width);
                let flat_view = p3_matrix::extension::FlatMatrixView::new(ef_mat);
                let tree = p3_merkle_tree::MerkleTree::from_parts(
                    vec![flat_view], digest_layers, arity_schedule,
                );
                Ok((cap, tree))
            }
            None => {
                let ef_values = EF::reconstitute_from_base(base_values);
                Err(RowMajorMatrix::new(ef_values, init_width))
            }
        }
    }

    fn transpose_pad_dft_and_commit(
        &self,
        data: &[BabyBear],
        in_rows: usize,
        in_cols: usize,
        padded_height: usize,
    ) -> Option<(Self::Commitment, Self::ProverData<RowMajorMatrix<BabyBear>>)> {
        let (values, cap, digest_layers, arity_schedule) =
            self.gpu.gpu_transpose_dft_and_merkle(data, in_rows, in_cols, padded_height, 1)?;
        let mat = RowMajorMatrix::new(values, in_rows);
        let tree = p3_merkle_tree::MerkleTree::from_parts(
            vec![mat], digest_layers, arity_schedule,
        );
        Some((cap, tree))
    }

    fn transpose_pad_dft_algebra_and_commit<EF>(
        &self,
        data: &[EF],
        in_rows: usize,
        in_cols: usize,
        padded_height: usize,
    ) -> Option<(
        Self::Commitment,
        Self::ProverData<p3_matrix::extension::FlatMatrixView<BabyBear, EF, RowMajorMatrix<EF>>>,
    )>
    where
        EF: p3_field::ExtensionField<BabyBear> + BasedVectorSpace<BabyBear> + Clone + Send + Sync,
    {
        let d = EF::DIMENSION;
        let base_data = EF::flatten_to_base(data.to_vec());
        let (values, cap, digest_layers, arity_schedule) =
            self.gpu.gpu_transpose_dft_and_merkle(&base_data, in_rows, in_cols, padded_height, d)?;
        let ef_values = EF::reconstitute_from_base(values);
        let ef_mat = RowMajorMatrix::new(ef_values, in_rows);
        let flat_view = p3_matrix::extension::FlatMatrixView::new(ef_mat);
        let tree = p3_merkle_tree::MerkleTree::from_parts(
            vec![flat_view], digest_layers, arity_schedule,
        );
        Some((cap, tree))
    }
}

/// No-op fusion for CPU-only MerkleTreeMmcs.
impl<P, PW, H, C> DftCommitFusion<BabyBear> for MerkleTreeMmcs<P, PW, H, C, 2, 8>
where
    P: PackedValue<Value = BabyBear>,
    PW: PackedValue<Value = BabyBear>,
    H: CryptographicHasher<BabyBear, [BabyBear; 8]>
        + CryptographicHasher<P, [PW; 8]>
        + Sync,
    C: PseudoCompressionFunction<[BabyBear; 8], 2>
        + PseudoCompressionFunction<[PW; 8], 2>
        + Sync,
{}

// ═══════════════════════════════════════════════════════════════════════
// GpuChallenger: DuplexChallenger wrapper with GPU-accelerated PoW grinding
// ═══════════════════════════════════════════════════════════════════════

use p3_baby_bear::Poseidon2BabyBear;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_field::{ExtensionField, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Hash, MerkleCap};

type InnerChallenger = DuplexChallenger<BabyBear, Poseidon2BabyBear<16>, 16, 8>;

/// A `DuplexChallenger` wrapper that offloads proof-of-work grinding to GPU.
///
/// All other challenger operations (observe, sample, etc.) delegate to the
/// inner `DuplexChallenger`. The `grind` call uses a Metal compute kernel
/// that parallelizes the Poseidon2 brute-force search across GPU threads.
#[derive(Clone)]
pub struct GpuChallenger {
    pub inner: InnerChallenger,
    dft: MetalBabyBearDft,
}

impl GpuChallenger {
    pub fn new(perm: Poseidon2BabyBear<16>, dft: MetalBabyBearDft) -> Self {
        Self {
            inner: InnerChallenger::new(perm),
            dft,
        }
    }
}

impl CanObserve<BabyBear> for GpuChallenger {
    fn observe(&mut self, value: BabyBear) {
        self.inner.observe(value);
    }
    fn observe_slice(&mut self, values: &[BabyBear]) {
        self.inner.observe_slice(values);
    }
}

impl<const N: usize> CanObserve<[BabyBear; N]> for GpuChallenger {
    fn observe(&mut self, values: [BabyBear; N]) {
        self.inner.observe(values);
    }
}

impl<const N: usize> CanObserve<Hash<BabyBear, BabyBear, N>> for GpuChallenger {
    fn observe(&mut self, values: Hash<BabyBear, BabyBear, N>) {
        self.inner.observe(values);
    }
}

impl CanObserve<Vec<Vec<BabyBear>>> for GpuChallenger {
    fn observe(&mut self, values: Vec<Vec<BabyBear>>) {
        self.inner.observe(values);
    }
}

impl<const N: usize> CanObserve<MerkleCap<BabyBear, [BabyBear; N]>> for GpuChallenger {
    fn observe(&mut self, values: MerkleCap<BabyBear, [BabyBear; N]>) {
        self.inner.observe(values);
    }
}

impl<const N: usize> CanObserve<&MerkleCap<BabyBear, [BabyBear; N]>> for GpuChallenger {
    fn observe(&mut self, values: &MerkleCap<BabyBear, [BabyBear; N]>) {
        self.inner.observe(values);
    }
}

impl<EF> CanSample<EF> for GpuChallenger
where
    EF: BasedVectorSpace<BabyBear>,
{
    fn sample(&mut self) -> EF {
        self.inner.sample()
    }
}

impl CanSampleBits<usize> for GpuChallenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.inner.sample_bits(bits)
    }
}

impl FieldChallenger<BabyBear> for GpuChallenger {}

impl GrindingChallenger for GpuChallenger {
    type Witness = BabyBear;

    #[tracing::instrument(name = "grind for proof-of-work witness", skip_all)]
    fn grind(&mut self, bits: usize) -> BabyBear {
        if bits == 0 {
            return BabyBear::ZERO;
        }

        // Build the pre-permutation state from sponge_state + input_buffer,
        // matching the CPU DuplexChallenger::grind logic.
        let witness_idx = self.inner.input_buffer.len();
        let mut base_state_monty = [0u32; 16];
        for i in 0..16 {
            let val = if i < self.inner.input_buffer.len() {
                self.inner.input_buffer[i]
            } else {
                self.inner.sponge_state[i]
            };
            // BabyBear is #[repr(transparent)] over u32 in Montgomery form
            base_state_monty[i] = unsafe { std::mem::transmute::<BabyBear, u32>(val) };
        }

        // Try GPU
        if let Some(canonical_nonce) = self.dft.gpu_pow_grind(
            &base_state_monty,
            witness_idx as u32,
            bits as u32,
        ) {
            // Convert canonical nonce back to BabyBear
            let witness = BabyBear::new(canonical_nonce);
            // Verify and update the challenger state (observe + sample)
            assert!(
                self.inner.check_witness(bits, witness),
                "GPU PoW witness failed verification"
            );
            return witness;
        }

        // GPU didn't find (extremely unlikely) — fall back to CPU
        self.inner.grind(bits)
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;

    /// Measure raw GPU memory bandwidth and per-phase NTT timings.
    #[test]
    fn gpu_bandwidth_and_phase_timing() {
        for &(test_log_n, test_width) in &[(20u32, 64usize), (22, 64)] {
            run_timing_test(test_log_n, test_width);
        }
    }

    fn run_timing_test(log_n: u32, width: usize) {
        let gpu = MetalBabyBearDft::default();
        let n = 1usize << log_n;
        let total = n * width;
        let total_bytes = (total * 4) as u64;
        let data_gb = total_bytes as f64 / 1e9;

        let opts = MTLResourceOptions::CPUCacheModeDefaultCache
            | MTLResourceOptions::StorageModeShared;
        let buf_a = gpu.device.new_buffer(total_bytes, opts);
        let buf_b = gpu.device.new_buffer(total_bytes, opts);
        let tw = gpu.twiddle_buffer(log_n);

        let bw_ps = {
            let lib = gpu.device.new_library_with_source(SHADER_MSL, &CompileOptions::new()).unwrap();
            let f = lib.get_function("bb_bandwidth_test", None).unwrap();
            gpu.device.new_compute_pipeline_state_with_function(&f).unwrap()
        };

        let time_cmd = |label: &str, encode: &dyn Fn(&metal::ComputeCommandEncoderRef)| -> f64 {
            // Warmup
            autoreleasepool(|| {
                let cmd = gpu.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                encode(&enc);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });
            // Timed runs
            let iters = 5;
            let start = std::time::Instant::now();
            for _ in 0..iters {
                autoreleasepool(|| {
                    let cmd = gpu.queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    encode(&enc);
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                });
            }
            let elapsed = start.elapsed().as_secs_f64() / iters as f64;
            let bw = (2.0 * data_gb) / elapsed; // read + write
            eprintln!("  {label:30}: {:.2} ms  ({:.1} GB/s)", elapsed * 1e3, bw);
            elapsed
        };

        let height = n as u32;
        let set_u32 = |enc: &metal::ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        eprintln!("\n=== GPU Bandwidth & Phase Timing (log_n={log_n}, width={width}, {:.2} GB) ===", data_gb);

        // 1) Raw bandwidth: read + write every element
        time_cmd("raw_bandwidth (r+w)", &|enc| {
            enc.set_compute_pipeline_state(&bw_ps);
            enc.set_buffer(0, Some(&buf_a), 0);
            let max_tg = bw_ps.max_total_threads_per_threadgroup() as u64;
            enc.dispatch_threads(
                MTLSize { width: total as u64, height: 1, depth: 1 },
                MTLSize { width: max_tg, height: 1, depth: 1 },
            );
        });

        // 2) Bitrev pass
        time_cmd("bitrev", &|enc| {
            enc.set_compute_pipeline_state(&gpu.bitrev_ps);
            enc.set_buffer(0, Some(&buf_a), 0);
            set_u32(enc, 1, height);
            set_u32(enc, 2, width as u32);
            set_u32(enc, 3, log_n);
            let tg_w = (width as u32).min(32);
            let tg_h = (gpu.bitrev_ps.max_total_threads_per_threadgroup() as u32 / tg_w).min(height);
            enc.dispatch_threads(
                MTLSize { width: width as u64, height: height as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
        });

        // 3) Shared-mem pass (10 stages fused)
        let log_block = gpu.effective_log_block(log_n);
        time_cmd(&format!("shared_mem ({log_block} stages)"), &|enc| {
            gpu.dispatch_shared_mem(enc, &buf_a, &tw, height, width as u32, log_block, 0);
        });

        // 4) One R8 dispatch
        time_cmd("single_R8 (3 stages)", &|enc| {
            let tg_w = (width as u32).min(32);
            let num_units = height >> 3;
            let max_tg = gpu.butterfly_r8_ps.max_total_threads_per_threadgroup() as u32;
            enc.set_compute_pipeline_state(&gpu.butterfly_r8_ps);
            enc.set_buffer(0, Some(&buf_a), 0);
            enc.set_buffer(1, Some(&tw), 0);
            set_u32(enc, 2, height);
            set_u32(enc, 3, width as u32);
            set_u32(enc, 4, log_block); // stage = log_block
            enc.dispatch_threads(
                MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: (max_tg / tg_w).min(num_units) as u64, depth: 1 },
            );
        });

        // 5) Stockham radix-4 (12 stages fused)
        let stockham_log_block = gpu.max_log_stockham_block.min(log_n);
        time_cmd(&format!("stockham_r4 ({stockham_log_block} stages)"), &|enc| {
            gpu.dispatch_stockham(enc, &buf_a, &tw, height, width as u32, stockham_log_block);
        });

        // 6) Full classic NTT (all phases combined)
        time_cmd("full_classic_NTT", &|enc| {
            gpu.encode_classic_ntt(enc, &buf_a, &tw, log_n, height, width as u32);
        });

        // 7) DIF NTT (no bitrev, no shared-mem, fused bitrev at end)
        time_cmd("dif_ntt (fused bitrev)", &|enc| {
            gpu.encode_dif_ntt(enc, &buf_a, &buf_b, &tw, log_n, height, width as u32);
        });

        // 8) Single DIF R8 dispatch
        let tg_w2 = (width as u32).min(32);
        time_cmd("single_DIF_R8 (3 stages)", &|enc| {
            let num_units = height >> 3;
            let max_tg = gpu.dif_r8_ps.max_total_threads_per_threadgroup() as u32;
            enc.set_compute_pipeline_state(&gpu.dif_r8_ps);
            enc.set_buffer(0, Some(&buf_a), 0);
            enc.set_buffer(1, Some(&tw), 0);
            set_u32(enc, 2, height);
            set_u32(enc, 3, width as u32);
            set_u32(enc, 4, 0u32);
            enc.dispatch_threads(
                MTLSize { width: width as u64, height: num_units as u64, depth: 1 },
                MTLSize {
                    width: tg_w2 as u64,
                    height: (max_tg / tg_w2).min(num_units) as u64,
                    depth: 1,
                },
            );
        });

        eprintln!("===\n");
    }

    /// Simulate the full benchmark flow with per-phase timing.
    #[test]
    fn gpu_full_flow_timing() {
        let gpu = MetalBabyBearDft::default();
        for &(log_n, width) in &[(20u32, 64usize), (22, 64)] {
            let n = 1usize << log_n;
            let total = n * width;
            let total_bytes = (total * 4) as u64;
            let data_gb = total_bytes as f64 / 1e9;

            let opts = MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared;
            let buf_a = gpu.device.new_buffer(total_bytes, opts);
            let buf_b = gpu.device.new_buffer(total_bytes, opts);
            let tw = gpu.twiddle_buffer(log_n);
            let height = n as u32;

            let iters = if log_n <= 20 { 5 } else { 2 };

            // Warmup
            autoreleasepool(|| {
                let cmd = gpu.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                gpu.encode_dif_ntt(&enc, &buf_a, &buf_b, &tw, log_n, height, width as u32);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            });

            // Pure GPU compute (no memcpy)
            let start = std::time::Instant::now();
            for _ in 0..iters {
                autoreleasepool(|| {
                    let cmd = gpu.queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    gpu.encode_dif_ntt(&enc, &buf_a, &buf_b, &tw, log_n, height, width as u32);
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                });
            }
            let gpu_ms = start.elapsed().as_secs_f64() / iters as f64 * 1e3;

            // Memcpy timing
            let mut rng = SmallRng::seed_from_u64(42);
            let values: Vec<BabyBear> = (0..total).map(|_| rng.random()).collect();
            let copy_start = std::time::Instant::now();
            for _ in 0..iters {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        values.as_ptr().cast::<u8>(),
                        buf_a.contents() as *mut u8,
                        total_bytes as usize,
                    );
                }
            }
            let copy_in_ms = copy_start.elapsed().as_secs_f64() / iters as f64 * 1e3;

            let mut out: Vec<BabyBear> = vec![BabyBear::ZERO; total];
            let copy_start = std::time::Instant::now();
            for _ in 0..iters {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf_b.contents() as *const BabyBear,
                        out.as_mut_ptr(),
                        total,
                    );
                }
            }
            let copy_out_ms = copy_start.elapsed().as_secs_f64() / iters as f64 * 1e3;

            // Full flow (clone + memcpy_in + gpu + memcpy_out)
            let full_start = std::time::Instant::now();
            for _ in 0..iters {
                let mut v = values.clone();
                gpu.try_gpu_dft_inplace(&mut v, n, width);
            }
            let full_ms = full_start.elapsed().as_secs_f64() / iters as f64 * 1e3;

            eprintln!("[{log_n}x{width} ({data_gb:.2}GB)] gpu_pure={gpu_ms:.1}ms copy_in={copy_in_ms:.1}ms copy_out={copy_out_ms:.1}ms full_flow={full_ms:.1}ms overhead={:.1}ms",
                full_ms - gpu_ms);
        }
    }

    #[test]
    fn metal_dft_matches_cpu() {
        let mut rng = SmallRng::seed_from_u64(42);
        let cpu_dft = Radix2DFTSmallBatch::<BabyBear>::default();
        let gpu_dft = MetalBabyBearDft {
            gpu_min_log_n: 0,
            ..MetalBabyBearDft::default()
        };

        for log_n in 2..=20 {
            let n = 1usize << log_n;
            for &width in &[1, 4, 16, 32, 64] {
                let values: Vec<BabyBear> = (0..n * width).map(|_| rng.random()).collect();
                let cpu_result =
                    cpu_dft.dft_batch(RowMajorMatrix::new(values.clone(), width));
                let gpu_result = gpu_dft.dft_batch(RowMajorMatrix::new(values, width));
                if cpu_result.values != gpu_result.values {
                    let first_diff = cpu_result.values.iter().zip(&gpu_result.values)
                        .enumerate()
                        .find(|(_, (a, b))| a != b);
                    if let Some((idx, (cpu_v, gpu_v))) = first_diff {
                        eprintln!("mismatch at log_n={log_n} w={width}: idx={idx} cpu={cpu_v:?} gpu={gpu_v:?}");
                    }
                    panic!("mismatch at log_n={log_n}, width={width}");
                }
            }
        }
    }

    #[test]
    fn metal_poseidon2_matches_cpu() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_symmetric::Permutation;

        let gpu = MetalBabyBearDft::default(); // uses seed 1

        // Create CPU permutation with the same seed
        let perm = Poseidon2BabyBear::<16>::new_from_rng_128(
            &mut SmallRng::seed_from_u64(1),
        );

        let input: [BabyBear; 16] = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);

        // CPU reference
        let mut expected = input;
        perm.permute_mut(&mut expected);

        // GPU compress: input[0..8] = left, input[8..16] = right → output first 8 of permute(input)
        let gpu_result = gpu.gpu_poseidon2_permute(&input);

        for i in 0..8 {
            assert_eq!(
                gpu_result[i], expected[i],
                "Poseidon2 mismatch at index {i}: gpu={:?} expected={:?}",
                gpu_result[i], expected[i]
            );
        }
        eprintln!("GPU Poseidon2 matches CPU reference (first 8 elements)!");
    }

    #[test]
    fn metal_poseidon2_leaf_hash_matches_cpu() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_symmetric::PaddingFreeSponge;
        use p3_symmetric::CryptographicHasher;

        let gpu = MetalBabyBearDft::default();
        let mut rng = SmallRng::seed_from_u64(123);

        // Create CPU hasher (same config as benchmark)
        let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
        let hasher = PaddingFreeSponge::<Poseidon2BabyBear<16>, 16, 8, 8>::new(perm);

        // Test with various leaf widths
        for leaf_width in [8, 16, 32, 64] {
            let num_leaves = 64u32;
            let data: Vec<BabyBear> = (0..(num_leaves as usize * leaf_width))
                .map(|_| rng.random())
                .collect();

            // CPU hash
            let cpu_digests: Vec<[BabyBear; 8]> = (0..num_leaves as usize)
                .map(|i| {
                    let row = &data[i * leaf_width..(i + 1) * leaf_width];
                    hasher.hash_iter(row.iter().copied())
                })
                .collect();

            // GPU hash
            let opts = MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared;
            let data_buf = gpu.device.new_buffer_with_data(
                data.as_ptr().cast(),
                (data.len() * size_of::<u32>()) as u64,
                opts,
            );
            let layers = gpu.gpu_merkle_tree(&data_buf, num_leaves, leaf_width as u32);
            let gpu_leaf_u32 = &layers[0];

            // Compare
            for i in 0..num_leaves as usize {
                for j in 0..8 {
                    let cpu_val: u32 = unsafe { std::mem::transmute(cpu_digests[i][j]) };
                    let gpu_val = gpu_leaf_u32[i * 8 + j];
                    assert_eq!(
                        cpu_val, gpu_val,
                        "Leaf hash mismatch: leaf={i} elem={j} width={leaf_width} cpu={cpu_val:#x} gpu={gpu_val:#x}"
                    );
                }
            }
            eprintln!("Leaf hash matches CPU for width={leaf_width}, {num_leaves} leaves");
        }
    }

    #[test]
    fn metal_dft_algebra_batch_matches_cpu() {
        use p3_field::extension::BinomialExtensionField;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let mut rng = SmallRng::seed_from_u64(99);
        let cpu_dft = Radix2DFTSmallBatch::<BabyBear>::default();
        let gpu_dft = MetalBabyBearDft {
            gpu_min_log_n: 0,
            ..MetalBabyBearDft::default()
        };

        for log_n in 2..=18 {
            let n = 1usize << log_n;
            for &ef_width in &[1, 4, 16] {
                let values: Vec<EF> = (0..n * ef_width).map(|_| rng.random()).collect();
                let cpu_result = cpu_dft.dft_algebra_batch(
                    RowMajorMatrix::new(values.clone(), ef_width),
                );
                let gpu_result = gpu_dft.dft_algebra_batch(
                    RowMajorMatrix::new(values, ef_width),
                );
                assert_eq!(
                    cpu_result.values, gpu_result.values,
                    "algebra_batch mismatch at log_n={log_n}, ef_width={ef_width}"
                );
            }
        }
    }

    #[test]
    fn gpu_mmcs_matches_cpu_mmcs() {
        use p3_baby_bear::Poseidon2BabyBear;
        use p3_commit::Mmcs;
        use p3_merkle_tree::MerkleTreeMmcs;
        use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

        type Perm = Poseidon2BabyBear<16>;
        type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
        type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
        type PackedF = <BabyBear as p3_field::Field>::Packing;
        type CpuMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let cpu_mmcs = CpuMmcs::new(hash, compress, 0);

        let gpu = MetalBabyBearDft::default();
        let gpu_mmcs = GpuMmcs::new(cpu_mmcs.clone(), gpu);

        let mut data_rng = SmallRng::seed_from_u64(42);

        for (height, width) in [(64, 16), (128, 8), (256, 32)] {
            let data: Vec<BabyBear> = (0..height * width)
                .map(|_| data_rng.random())
                .collect();

            let (cpu_cap, cpu_data) =
                cpu_mmcs.commit_matrix(RowMajorMatrix::new(data.clone(), width));
            let (gpu_cap, gpu_data) =
                gpu_mmcs.commit_matrix(RowMajorMatrix::new(data.clone(), width));

            assert_eq!(
                cpu_cap, gpu_cap,
                "Cap mismatch for {height}x{width}"
            );

            for idx in [0, 1, height / 2, height - 1] {
                let cpu_opening = cpu_mmcs.open_batch(idx, &cpu_data);
                let gpu_opening = gpu_mmcs.open_batch(idx, &gpu_data);
                assert_eq!(
                    cpu_opening.opened_values, gpu_opening.opened_values,
                    "Opened values mismatch at idx={idx} for {height}x{width}"
                );
                assert_eq!(
                    cpu_opening.opening_proof, gpu_opening.opening_proof,
                    "Proof mismatch at idx={idx} for {height}x{width}"
                );
            }
            eprintln!("GPU MMCS matches CPU for {height}x{width}");
        }
    }
}
