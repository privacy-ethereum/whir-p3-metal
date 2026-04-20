use std::vec;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{Field, extension::BinomialExtensionField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_commit::Mmcs;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    gpu_dft::{GpuMmcs, MetalBabyBearDft},
    parameters::{
        DEFAULT_MAX_POW, FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
        WhirConfig,
    },
    whir::{committer::writer::CommitmentWriter, proof::WhirProof, prover::Prover},
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyGpuMmcs = GpuMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

fn make_whir_config(num_variables: usize) -> (
    WhirConfig<EF, F, MyMmcs, MyChallenger>,
    ProtocolParameters<MyMmcs>,
    Perm,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm.clone());
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    let whir_params = ProtocolParameters {
        security_level: 100,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::Constant(4),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 3,
    };

    let params = WhirConfig::new(num_variables, whir_params.clone());
    (params, whir_params, perm)
}

fn make_gpu_whir_config(num_variables: usize, gpu: MetalBabyBearDft) -> (
    WhirConfig<EF, F, MyGpuMmcs, MyChallenger>,
    ProtocolParameters<MyGpuMmcs>,
    Perm,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm.clone());
    let inner_mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);
    let mmcs = MyGpuMmcs::new(inner_mmcs, gpu);

    let whir_params = ProtocolParameters {
        security_level: 100,
        pow_bits: DEFAULT_MAX_POW,
        folding_factor: FoldingFactor::Constant(4),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
        rs_domain_initial_reduction_factor: 3,
    };

    let params = WhirConfig::new(num_variables, whir_params.clone());
    (params, whir_params, perm)
}

fn benchmark_raw_dft(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_dft");

    for log_n in [14, 16, 18, 20, 22] {
        for width in [1, 16, 64] {
            let n: usize = 1 << log_n;
            let mut rng = SmallRng::seed_from_u64(0);
            let values: Vec<F> = (0..n * width).map(|_| rng.random()).collect();

            let cpu_dft = Radix2DFTSmallBatch::<F>::new(n);
            group.bench_function(&format!("cpu_{log_n}x{width}"), |b| {
                b.iter_batched(
                    || RowMajorMatrix::new(values.clone(), width),
                    |mat| cpu_dft.dft_batch(mat),
                    criterion::BatchSize::LargeInput,
                );
            });

            let gpu_dft = MetalBabyBearDft::new(n);
            group.bench_function(&format!("metal_gpu_{log_n}x{width}"), |b| {
                b.iter_batched(
                    || RowMajorMatrix::new(values.clone(), width),
                    |mat| gpu_dft.dft_batch(mat),
                    criterion::BatchSize::LargeInput,
                );
            });
        }
    }

    group.finish();
}

fn benchmark_algebra_dft(c: &mut Criterion) {
    let mut group = c.benchmark_group("algebra_dft");

    // Sizes matching the WHIR prover rounds after flattening:
    // Round 1: log_n=17, EF width=16 → flattened: 17x64 = 32MB
    // Round 2: log_n=13, EF width=16 → flattened: 13x64 = 2MB
    for (log_n, ef_width) in [(17, 16), (13, 16), (20, 16)] {
        let n: usize = 1 << log_n;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<EF> = (0..n * ef_width).map(|_| rng.random()).collect();

        let cpu_dft = Radix2DFTSmallBatch::<F>::new(n);
        group.bench_function(&format!("cpu_{log_n}x{ef_width}ef"), |b| {
            b.iter_batched(
                || RowMajorMatrix::new(values.clone(), ef_width),
                |mat| cpu_dft.dft_algebra_batch(mat),
                criterion::BatchSize::LargeInput,
            );
        });

        let gpu_dft = MetalBabyBearDft::new(n);
        group.bench_function(&format!("metal_gpu_{log_n}x{ef_width}ef"), |b| {
            b.iter_batched(
                || RowMajorMatrix::new(values.clone(), ef_width),
                |mat| gpu_dft.dft_algebra_batch(mat),
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_merkle(c: &mut Criterion) {
    use metal::MTLResourceOptions;

    let mut group = c.benchmark_group("merkle");

    for (log_n, width) in [(21, 16), (18, 64)] {
        let num_leaves: u32 = 1 << log_n;
        let total = (num_leaves as usize) * width;
        let mut rng = SmallRng::seed_from_u64(0);
        let data: Vec<F> = (0..total).map(|_| rng.random()).collect();

        let (_, whir_params, _) = make_whir_config(24);
        let mmcs = whir_params.mmcs.clone();
        group.bench_function(&format!("cpu_{log_n}x{width}"), |b| {
            b.iter_batched(
                || RowMajorMatrix::new(data.clone(), width),
                |mat| mmcs.commit_matrix(mat),
                criterion::BatchSize::LargeInput,
            );
        });

        let gpu = MetalBabyBearDft::new(1 << log_n);
        let opts = MTLResourceOptions::CPUCacheModeDefaultCache
            | MTLResourceOptions::StorageModeShared;
        group.bench_function(&format!("gpu_{log_n}x{width}"), |b| {
            b.iter_batched(
                || {
                    gpu.device.new_buffer_with_data(
                        data.as_ptr().cast(),
                        (total * 4) as u64,
                        opts,
                    )
                },
                |buf| gpu.gpu_merkle_tree(&buf, num_leaves, width as u32),
                criterion::BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn benchmark_dft_plus_merkle(c: &mut Criterion) {
    let mut group = c.benchmark_group("dft_plus_merkle");

    // Full DFT+Merkle pipeline for the initial commit (21x16) - the biggest operation.
    let log_n = 21;
    let width = 16;
    let n: usize = 1 << log_n;
    let mut rng = SmallRng::seed_from_u64(0);
    let values: Vec<F> = (0..n * width).map(|_| rng.random()).collect();

    let (_, whir_params, _) = make_whir_config(24);
    let mmcs = whir_params.mmcs.clone();
    let cpu_dft = Radix2DFTSmallBatch::<F>::new(n);

    group.bench_function("cpu_21x16", |b| {
        b.iter_batched(
            || RowMajorMatrix::new(values.clone(), width),
            |mat| {
                let dft_out = cpu_dft.dft_batch(mat).to_row_major_matrix();
                mmcs.commit_matrix(dft_out)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    let gpu_dft = MetalBabyBearDft::new(n);
    group.bench_function("gpu_21x16", |b| {
        b.iter_batched(
            || RowMajorMatrix::new(values.clone(), width),
            |mat| {
                let dft_out = gpu_dft.dft_batch(mat).to_row_major_matrix();
                gpu_dft.gpu_commit_matrix(dft_out)
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn benchmark_whir_prove(c: &mut Criterion) {
    let num_variables = 24;
    let (params, whir_params, perm) = make_whir_config(num_variables);

    let num_coeffs = 1 << num_variables;
    let mut rng = SmallRng::seed_from_u64(0);
    let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng.random()).collect());
    let mut initial_statement = params.initial_statement(polynomial, SumcheckStrategy::Svo);
    let _ = initial_statement.evaluate(&Point::rand(&mut rng, num_variables));

    let mut domainsep = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, 8>(&params);
    domainsep.add_whir_proof::<_, _, 8>(&params);

    let max_fft = 1 << params.max_fft_size();

    c.bench_function("whir_prove_cpu", |b| {
        let dft = Radix2DFTSmallBatch::<F>::new(max_fft);
        b.iter(|| {
            let mut challenger = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut challenger);
            let mut proof =
                WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);
            let committer = CommitmentWriter::new(&params);
            let mut stmt = initial_statement.clone();
            let prover_data = committer
                .commit(&dft, &mut proof, &mut challenger, &mut stmt)
                .unwrap();
            let prover = Prover(&params);
            prover
                .prove(&dft, &mut proof, &mut challenger, &stmt, prover_data)
                .unwrap();
        });
    });

    c.bench_function("whir_prove_metal_gpu", |b| {
        let dft = MetalBabyBearDft::new(max_fft);
        b.iter(|| {
            let mut challenger = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut challenger);
            let mut proof =
                WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);
            let committer = CommitmentWriter::new(&params);
            let mut stmt = initial_statement.clone();
            let prover_data = committer
                .commit(&dft, &mut proof, &mut challenger, &mut stmt)
                .unwrap();
            let prover = Prover(&params);
            prover
                .prove(&dft, &mut proof, &mut challenger, &stmt, prover_data)
                .unwrap();
        });
    });

    // GPU DFT + GPU Merkle (full GPU pipeline)
    {
        let dft = MetalBabyBearDft::new(max_fft);
        let (gpu_params, gpu_whir_params, _) =
            make_gpu_whir_config(num_variables, dft.clone());

        let mut gpu_domainsep = DomainSeparator::new(vec![]);
        gpu_domainsep.commit_statement::<_, _, 8>(&gpu_params);
        gpu_domainsep.add_whir_proof::<_, _, 8>(&gpu_params);

        c.bench_function("whir_prove_gpu_dft_merkle", |b| {
            b.iter(|| {
                let mut challenger = MyChallenger::new(perm.clone());
                gpu_domainsep.observe_domain_separator(&mut challenger);
                let mut proof = WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(
                    &gpu_whir_params,
                    num_variables,
                );
                let committer = CommitmentWriter::new(&gpu_params);
                let mut stmt = initial_statement.clone();
                let prover_data = committer
                    .commit_fused(&dft, &mut proof, &mut challenger, &mut stmt)
                    .unwrap();
                let prover = Prover(&gpu_params);
                prover
                    .prove(&dft, &mut proof, &mut challenger, &stmt, prover_data)
                    .unwrap();
            });
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_raw_dft, benchmark_algebra_dft, benchmark_merkle, benchmark_dft_plus_merkle, benchmark_whir_prove
);
criterion_main!(benches);
