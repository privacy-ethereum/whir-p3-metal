use std::vec;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::{point::Point, poly::Poly};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use whir_p3::{
    fiat_shamir::domain_separator::DomainSeparator,
    gpu_dft::MetalBabyBearDft,
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

fn benchmark_raw_dft(c: &mut Criterion) {
    let mut group = c.benchmark_group("raw_dft");

    for log_n in [14, 16, 18, 20, 22] {
        let n: usize = 1 << log_n;
        let width = 16;
        let mut rng = SmallRng::seed_from_u64(0);
        let values: Vec<F> = (0..n * width).map(|_| rng.random()).collect();

        let cpu_dft = Radix2DFTSmallBatch::<F>::new(n);
        group.bench_function(&format!("cpu_{log_n}x{width}"), |b| {
            b.iter(|| {
                let mat = RowMajorMatrix::new(values.clone(), width);
                cpu_dft.dft_batch(mat)
            });
        });

        let gpu_dft = MetalBabyBearDft::new(n);
        group.bench_function(&format!("metal_gpu_{log_n}x{width}"), |b| {
            b.iter(|| {
                let mat = RowMajorMatrix::new(values.clone(), width);
                gpu_dft.dft_batch(mat)
            });
        });

        group.bench_function(&format!("clone_overhead_{log_n}x{width}"), |b| {
            b.iter(|| {
                values.clone()
            });
        });
    }

    group.finish();
}

fn benchmark_whir_prove(c: &mut Criterion) {
    let num_variables = 20;
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
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_raw_dft, benchmark_whir_prove
);
criterion_main!(benches);
