use std::io::Write;
use std::time::Instant;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{Field, extension::BinomialExtensionField};
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

fn run_cpu(n: usize, f: usize, r: usize) -> Option<f64> {
    let result = std::panic::catch_unwind(|| {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level: 100,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(f),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: r,
            rs_domain_initial_reduction_factor: 3,
        };
        let params = WhirConfig::new(n, whir_params.clone());
        let max_fft = 1 << params.max_fft_size();
        let dft = Radix2DFTSmallBatch::<F>::new(max_fft);

        let num_coeffs = 1usize << n;
        let mut rng2 = SmallRng::seed_from_u64(0);
        let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng2.random()).collect());
        let mut initial_statement = params.initial_statement(polynomial, SumcheckStrategy::Svo);
        let _ = initial_statement.evaluate(&Point::rand(&mut rng2, n));

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        // Warmup
        {
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof = WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
        }

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof = WhirProof::<F, EF, MyMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[1]
    });
    result.ok()
}

fn run_gpu(n: usize, f: usize, r: usize) -> Option<f64> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm.clone());
        let inner = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params_cpu = ProtocolParameters {
            security_level: 100,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(f),
            mmcs: inner.clone(),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: r,
            rs_domain_initial_reduction_factor: 3,
        };
        let params_cpu = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(n, whir_params_cpu);
        let max_fft = 1 << params_cpu.max_fft_size();
        let dft = MetalBabyBearDft::new(max_fft);
        let mmcs = MyGpuMmcs::new(inner, dft.clone());

        let whir_params = ProtocolParameters {
            security_level: 100,
            pow_bits: DEFAULT_MAX_POW,
            folding_factor: FoldingFactor::Constant(f),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: r,
            rs_domain_initial_reduction_factor: 3,
        };
        let params = WhirConfig::new(n, whir_params.clone());

        let num_coeffs = 1usize << n;
        let mut rng2 = SmallRng::seed_from_u64(0);
        let polynomial = Poly::<F>::new((0..num_coeffs).map(|_| rng2.random()).collect());
        let mut initial_statement = params.initial_statement(polynomial, SumcheckStrategy::Svo);
        let _ = initial_statement.evaluate(&Point::rand(&mut rng2, n));

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        // Warmup
        {
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof = WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
        }

        let mut times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut ch = MyChallenger::new(perm.clone());
            domainsep.observe_domain_separator(&mut ch);
            let mut proof = WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
            let comm = CommitmentWriter::new(&params);
            let mut s = initial_statement.clone();
            let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
            Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[1]
    }));
    result.ok()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Single config mode: sweep <n> <f> <r> <mode>
    if args.len() == 5 {
        let n: usize = args[1].parse().unwrap();
        let f: usize = args[2].parse().unwrap();
        let r: usize = args[3].parse().unwrap();
        let mode = &args[4];
        let result = match mode.as_str() {
            "cpu" => run_cpu(n, f, r),
            "gpu" => run_gpu(n, f, r),
            _ => None,
        };
        match result {
            Some(t) => println!("{:.4}", t),
            None => std::process::exit(1),
        }
        return;
    }

    // Orchestrator mode
    let exe = &args[0];

    // Results file that persists across restarts
    let results_path = "sweep_results.txt";

    // Load already-completed configs
    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(contents) = std::fs::read_to_string(results_path) {
        for line in contents.lines() {
            if line.starts_with(|c: char| c.is_ascii_digit()) {
                let key: String = line.split_whitespace().take(3).collect::<Vec<_>>().join("_");
                done.insert(key);
            }
        }
    }

    let mut configs: Vec<(usize, usize, usize)> = Vec::new();
    let folds = &[4, 6, 8, 10, 12, 14, 16];
    let rates = &[1, 2, 3, 4, 6, 8, 10, 12, 14, 16];
    for &n in &[22, 24] {
        for &f in folds {
            for &r in rates {
                if r > f { continue; }
                configs.push((n, f, r));
            }
        }
    }

    // GPU memory safety limit: total domain = 2^(n+r) elements.
    // Each element is 4 bytes; DFT needs ~3x (in + out + merkle).
    // Hard cap at n+r <= 25 (128 MB domain, ~384 MB GPU total).
    // This avoids the Metal kernel panics that crashed the system.
    let max_log_domain_gpu: usize = 25;
    let max_cpu_secs = 15.0;

    let header = format!(
        "{:<6} {:<6} {:<6} {:>10} {:>10} {:>8}  {}",
        "n", "fold", "rate", "CPU(ms)", "GPU(ms)", "speedup", "notes"
    );
    let sep = "-".repeat(70);

    // Print header (and append to file if starting fresh)
    if done.is_empty() {
        let mut f = std::fs::OpenOptions::new()
            .create(true).append(true).open(results_path).unwrap();
        writeln!(f, "{header}").unwrap();
        writeln!(f, "{sep}").unwrap();
    }
    println!("{header}");
    println!("{sep}");

    // Print already-done lines
    if let Ok(contents) = std::fs::read_to_string(results_path) {
        for line in contents.lines() {
            if line.starts_with(|c: char| c.is_ascii_digit()) {
                println!("{line}");
            }
        }
    }

    for &(n, f, r) in &configs {
        let key = format!("{n}_{f}_{r}");
        if done.contains(&key) {
            continue;
        }

        // Log which config we're about to try
        {
            let mut file = std::fs::OpenOptions::new()
                .create(true).append(true).open(results_path).unwrap();
            writeln!(file, "# STARTING n={n} f={f} r={r} ...").unwrap();
        }

        // Run CPU in subprocess
        let cpu_out = std::process::Command::new(exe)
            .args([&n.to_string(), &f.to_string(), &r.to_string(), "cpu"])
            .output();

        let cpu_time = cpu_out.ok().and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()?.trim().parse::<f64>().ok()
            } else {
                None
            }
        });

        let cpu_time = match cpu_time {
            None => continue,  // invalid config, skip silently
            Some(t) if t > max_cpu_secs => {
                let line = format!(
                    "{:<6} {:<6} {:<6} {:>10.1} {:>10} {:>8}  CPU too slow",
                    n, f, r, t * 1000.0, "-", "-"
                );
                println!("{line}");
                let mut file = std::fs::OpenOptions::new()
                    .create(true).append(true).open(results_path).unwrap();
                writeln!(file, "{line}").unwrap();
                continue;
            }
            Some(t) => t,
        };

        // Check GPU memory limit
        let log_domain = n + r;
        if log_domain > max_log_domain_gpu {
            let line = format!(
                "{:<6} {:<6} {:<6} {:>10.1} {:>10} {:>8}  domain 2^{log_domain} > limit",
                n, f, r, cpu_time * 1000.0, "-", "-"
            );
            println!("{line}");
            let mut file = std::fs::OpenOptions::new()
                .create(true).append(true).open(results_path).unwrap();
            writeln!(file, "{line}").unwrap();
            continue;
        }

        // Run GPU in subprocess
        let gpu_child = std::process::Command::new(exe)
            .args([&n.to_string(), &f.to_string(), &r.to_string(), "gpu"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let gpu_time = gpu_child.ok()
            .and_then(|c| c.wait_with_output().ok())
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()?.trim().parse::<f64>().ok()
                } else {
                    None
                }
            });

        let line = match gpu_time {
            Some(g) => {
                let speedup = cpu_time / g;
                format!(
                    "{:<6} {:<6} {:<6} {:>10.1} {:>10.1} {:>7.2}x",
                    n, f, r, cpu_time * 1000.0, g * 1000.0, speedup
                )
            }
            None => {
                format!(
                    "{:<6} {:<6} {:<6} {:>10.1} {:>10} {:>8}  GPU fail",
                    n, f, r, cpu_time * 1000.0, "-", "-"
                )
            }
        };

        println!("{line}");
        let mut file = std::fs::OpenOptions::new()
            .create(true).append(true).open(results_path).unwrap();
        writeln!(file, "{line}").unwrap();
    }
}
