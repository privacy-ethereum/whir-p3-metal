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
use std::sync::Mutex;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use objc::rc::autoreleasepool;
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;

const SHADER_MSL: &str = include_str!("../shaders/babybear_ntt.metal");

/// Max threadgroup shared memory in uint32 elements (32 KB / 4).
const MAX_SHARED_ELEMS: u32 = 8192;

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

/// Reinterpret `Vec<u32>` as `Vec<BabyBear>` (zero-cost, same layout).
fn u32_vec_to_babybear(v: Vec<u32>) -> Vec<BabyBear> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: BabyBear is #[repr(transparent)] over u32 — identical layout.
    unsafe { Vec::from_raw_parts(ptr.cast::<BabyBear>(), len, cap) }
}

/// Metal-backed DFT for BabyBear (macOS).
/// Falls back to CPU for small sizes or if the GPU fails.
pub struct MetalBabyBearDft {
    cpu: Radix2DFTSmallBatch<BabyBear>,
    gpu_min_log_n: u32,
    device: Device,
    queue: metal::CommandQueue,
    bitrev_ps: ComputePipelineState,
    butterfly_ps: ComputePipelineState,
    shared_mem_ps: ComputePipelineState,
    /// Hard upper bound on the shared-memory block log from the device
    /// (max threads per threadgroup). The actual `log_block` used per call
    /// is `min(this, log2(MAX_SHARED_ELEMS / width), log_n)`.
    max_log_shared_block: u32,
    twiddle_bufs: Mutex<HashMap<u32, Buffer>>,
}

impl Clone for MetalBabyBearDft {
    fn clone(&self) -> Self {
        Self::new_with_min_log_n(self.cpu.clone(), self.gpu_min_log_n)
    }
}

impl Default for MetalBabyBearDft {
    fn default() -> Self {
        Self::new_with_min_log_n(Radix2DFTSmallBatch::default(), 10)
    }
}

impl MetalBabyBearDft {
    pub fn new(max_fft_size: usize) -> Self {
        Self::new_with_min_log_n(Radix2DFTSmallBatch::new(max_fft_size), 10)
    }

    fn new_with_min_log_n(cpu: Radix2DFTSmallBatch<BabyBear>, gpu_min_log_n: u32) -> Self {
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
        let butterfly_ps = make_ps("bb_ntt_butterfly");
        let shared_mem_ps = make_ps("bb_ntt_shared_mem");

        let max_tg = shared_mem_ps.max_total_threads_per_threadgroup() as u32;
        let max_log_shared_block = max_tg.min(1024).ilog2();

        Self {
            cpu,
            gpu_min_log_n,
            device,
            queue,
            bitrev_ps,
            butterfly_ps,
            shared_mem_ps,
            max_log_shared_block,
            twiddle_bufs: Mutex::new(HashMap::new()),
        }
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

    /// Effective log_block for a given (log_n, width): the number of butterfly
    /// stages fused inside the shared-memory kernel.
    fn effective_log_block(&self, log_n: u32, width: u32) -> u32 {
        // block_size * width ≤ MAX_SHARED_ELEMS  →  block_size ≤ MAX_SHARED_ELEMS / width
        let max_block_for_width = (MAX_SHARED_ELEMS / width).max(1).ilog2();
        self.max_log_shared_block
            .min(max_block_for_width)
            .min(log_n)
    }

    fn try_gpu_dft_batch(&self, mat: &RowMajorMatrix<BabyBear>) -> Option<RowMajorMatrix<BabyBear>> {
        let height = mat.height();
        let width = mat.width();
        let log_n = log2_strict_usize(height) as u32;
        if log_n < self.gpu_min_log_n {
            return None;
        }

        // Zero-copy: BabyBear is #[repr(transparent)] over u32 (Montgomery form).
        // GPU shader now operates in Montgomery form, so no conversion needed.
        let mut buf = babybear_vec_to_u32(mat.values.clone());

        autoreleasepool(|| {
            self.ntt_row_major(&mut buf, log_n, height as u32, width as u32)
                .ok()
        })?;

        // Reinterpret u32 results (Montgomery form) back as BabyBear.
        let result = u32_vec_to_babybear(buf);
        Some(RowMajorMatrix::new(result, width))
    }

    /// Run the NTT on **row-major** data in-place. `buf` has `height * width`
    /// elements laid out as `buf[row * width + col]`.
    fn ntt_row_major(
        &self,
        buf: &mut [u32],
        log_n: u32,
        height: u32,
        width: u32,
    ) -> Result<(), String> {
        let tw = self.twiddle_buffer(log_n);
        let total_bytes = (buf.len() * size_of::<u32>()) as u64;
        let data_buf = self.device.new_buffer(
            total_bytes,
            MTLResourceOptions::CPUCacheModeDefaultCache | MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                data_buf.contents() as *mut u32,
                buf.len(),
            );
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        self.encode_batch_ntt(&enc, &data_buf, &tw, log_n, height, width);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if cmd.status() == MTLCommandBufferStatus::Error {
            return Err("Metal command buffer failed".into());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data_buf.contents() as *const u32,
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        Ok(())
    }

    /// Encode the full batched NTT (bitrev + shared-mem fused stages +
    /// global butterfly stages) into `enc`.
    fn encode_batch_ntt(
        &self,
        enc: &ComputeCommandEncoderRef,
        data: &metal::BufferRef,
        twiddles: &metal::BufferRef,
        log_n: u32,
        height: u32,
        width: u32,
    ) {
        let log_block = self.effective_log_block(log_n, width);
        let block_size = 1u32 << log_block;

        // Helper: push a u32 constant at `index`.
        let set_u32 = |enc: &ComputeCommandEncoderRef, index: u64, val: u32| {
            enc.set_bytes(index, size_of::<u32>() as u64, (&val as *const u32).cast());
        };

        // ── Step 1: bit-reverse permutation (2D: x=col, y=row) ──
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

        // ── Step 2 (Phase B): fused stages in shared memory ──
        {
            let res: &metal::ResourceRef = data.deref();
            enc.memory_barrier_with_resources(&[res]);

            enc.set_compute_pipeline_state(&self.shared_mem_ps);
            enc.set_buffer(0, Some(data), 0);
            enc.set_buffer(1, Some(twiddles), 0);
            set_u32(enc, 2, height);
            set_u32(enc, 3, width);
            set_u32(enc, 4, log_block);

            let num_blocks = height / block_size;
            enc.dispatch_thread_groups(
                MTLSize { width: num_blocks as u64, height: 1, depth: 1 },
                MTLSize { width: block_size as u64, height: 1, depth: 1 },
            );
        }

        // ── Step 3 (Phase A): remaining global-memory butterfly stages ──
        if log_block < log_n {
            enc.set_compute_pipeline_state(&self.butterfly_ps);
            let num_butterflies = height >> 1;

            for stage in log_block..log_n {
                let res: &metal::ResourceRef = data.deref();
                enc.memory_barrier_with_resources(&[res]);
                enc.set_buffer(0, Some(data), 0);
                enc.set_buffer(1, Some(twiddles), 0);
                set_u32(enc, 2, height);
                set_u32(enc, 3, width);
                set_u32(enc, 4, stage);
                let grid = MTLSize {
                    width: width as u64,
                    height: num_butterflies as u64,
                    depth: 1,
                };
                let tg = MTLSize {
                    width: width.min(32) as u64,
                    height: (self.butterfly_ps.max_total_threads_per_threadgroup() as u32
                        / width.min(32))
                    .min(num_butterflies) as u64,
                    depth: 1,
                };
                enc.dispatch_threads(grid, tg);
            }
        }
    }
}

impl TwoAdicSubgroupDft<BabyBear> for MetalBabyBearDft {
    type Evaluations = RowMajorMatrix<BabyBear>;

    fn dft_batch(&self, mat: RowMajorMatrix<BabyBear>) -> Self::Evaluations {
        if let Some(result) = self.try_gpu_dft_batch(&mat) {
            return result;
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
        let coeffs = self.idft_batch(mat);
        let h = coeffs.height();
        let w = coeffs.width();
        let new_h = h << added_bits;
        let mut padded_values = coeffs.values;
        padded_values.resize(new_h * w, <BabyBear as PrimeCharacteristicRing>::ZERO);
        let padded = RowMajorMatrix::new(padded_values, w);
        self.coset_dft_batch(padded, shift)
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;

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
            for &width in &[1, 4, 16] {
                let values: Vec<BabyBear> = (0..n * width).map(|_| rng.random()).collect();
                let cpu_result =
                    cpu_dft.dft_batch(RowMajorMatrix::new(values.clone(), width));
                let gpu_result = gpu_dft.dft_batch(RowMajorMatrix::new(values, width));
                assert_eq!(
                    cpu_result.values, gpu_result.values,
                    "mismatch at log_n={log_n}, width={width}"
                );
            }
        }
    }
}
