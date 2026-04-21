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
use tracing_forest::ForestLayer;
use tracing_subscriber::{Registry, layer::SubscriberExt};
use whir_p3::{
    GpuChallenger,
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
            rs_domain_initial_reduction_factor: f.min(3),
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

fn run_gpu(n: usize, f: usize, r: usize, fuse_rounds: bool) -> Option<f64> {
    run_gpu_inner(n, f, r, fuse_rounds, false)
}

fn run_gpu_grind(n: usize, f: usize, r: usize) -> Option<f64> {
    run_gpu_inner(n, f, r, true, true)
}

fn run_gpu_inner(n: usize, f: usize, r: usize, fuse_rounds: bool, gpu_grind: bool) -> Option<f64> {
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
            rs_domain_initial_reduction_factor: f.min(3),
        };
        let params_cpu = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(n, whir_params_cpu);
        let max_fft = 1 << params_cpu.max_fft_size();
        let dft = MetalBabyBearDft::new(max_fft);
        let mmcs = MyGpuMmcs::new(inner, dft.clone());

        if gpu_grind {
            // Use GpuChallenger for GPU-accelerated PoW grinding
            let whir_params = ProtocolParameters {
                security_level: 100,
                pow_bits: DEFAULT_MAX_POW,
                folding_factor: FoldingFactor::Constant(f),
                mmcs,
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: r,
                rs_domain_initial_reduction_factor: f.min(3),
            };
            let params = WhirConfig::<EF, F, MyGpuMmcs, GpuChallenger>::new(n, whir_params.clone());

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
                let mut ch = GpuChallenger::new(perm.clone(), dft.clone());
                domainsep.observe_domain_separator(&mut ch);
                let mut proof = WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
                let comm = CommitmentWriter::new(&params);
                let mut s = initial_statement.clone();
                let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
                Prover(&params).prove_fused(&dft, &mut proof, &mut ch, &s, pd).unwrap();
            }

            let mut times = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                let mut ch = GpuChallenger::new(perm.clone(), dft.clone());
                domainsep.observe_domain_separator(&mut ch);
                let mut proof = WhirProof::<F, EF, MyGpuMmcs>::from_protocol_parameters(&whir_params, n);
                let comm = CommitmentWriter::new(&params);
                let mut s = initial_statement.clone();
                let pd = comm.commit_fused(&dft, &mut proof, &mut ch, &mut s).unwrap();
                Prover(&params).prove_fused(&dft, &mut proof, &mut ch, &s, pd).unwrap();
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[1]
        } else {
            let whir_params = ProtocolParameters {
                security_level: 100,
                pow_bits: DEFAULT_MAX_POW,
                folding_factor: FoldingFactor::Constant(f),
                mmcs,
                soundness_type: SecurityAssumption::CapacityBound,
                starting_log_inv_rate: r,
                rs_domain_initial_reduction_factor: f.min(3),
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
                if fuse_rounds {
                    Prover(&params).prove_fused(&dft, &mut proof, &mut ch, &s, pd).unwrap();
                } else {
                    Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
                }
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
                if fuse_rounds {
                    Prover(&params).prove_fused(&dft, &mut proof, &mut ch, &s, pd).unwrap();
                } else {
                    Prover(&params).prove(&dft, &mut proof, &mut ch, &s, pd).unwrap();
                }
                times.push(t0.elapsed().as_secs_f64());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            times[1]
        }
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
        if mode == "profile" || mode == "profile_grind" {
            let subscriber = Registry::default().with(ForestLayer::default());
            tracing::subscriber::set_global_default(subscriber).ok();
            let t = if mode == "profile_grind" {
                run_gpu_grind(n, f, r)
            } else {
                run_gpu(n, f, r, true)
            };
            match t {
                Some(t) => eprintln!("Total: {:.1} ms", t * 1000.0),
                None => eprintln!("FAILED"),
            }
            return;
        }
        let result = match mode.as_str() {
            "cpu" => run_cpu(n, f, r),
            "gpu" => run_gpu(n, f, r, false),
            "gpu_fused" => run_gpu(n, f, r, true),
            "gpu_grind" => run_gpu_grind(n, f, r),
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
    let folds = &[1, 2, 3, 4, 6, 8];
    let rates = &[1, 2, 3];
    for &n in &[18, 20, 22, 24] {
        for &f in folds {
            for &r in rates {
                configs.push((n, f, r));
            }
        }
    }

    let max_log_domain_gpu: usize = 27;
    let max_cpu_secs = 120.0;

    let header = format!(
        "{:<6} {:<6} {:<6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "n", "fold", "rate", "CPU(ms)", "GPU(ms)", "FUSED(ms)", "GRIND(ms)", "gpu/cpu", "fused/cpu", "grind/cpu"
    );
    let sep = "-".repeat(120);

    if done.is_empty() {
        let mut f = std::fs::OpenOptions::new()
            .create(true).append(true).open(results_path).unwrap();
        writeln!(f, "{header}").unwrap();
        writeln!(f, "{sep}").unwrap();
    }
    println!("{header}");
    println!("{sep}");

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

        {
            let mut file = std::fs::OpenOptions::new()
                .create(true).append(true).open(results_path).unwrap();
            writeln!(file, "# STARTING n={n} f={f} r={r} ...").unwrap();
        }

        // Run CPU
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
            None => continue,
            Some(t) if t > max_cpu_secs => {
                let line = format!(
                    "{:<6} {:<6} {:<6} {:>10.1} {:>10} {:>10} {:>8} {:>8}  CPU too slow",
                    n, f, r, t * 1000.0, "-", "-", "-", "-"
                );
                println!("{line}");
                let mut file = std::fs::OpenOptions::new()
                    .create(true).append(true).open(results_path).unwrap();
                writeln!(file, "{line}").unwrap();
                continue;
            }
            Some(t) => t,
        };

        let log_domain = n + r;
        if log_domain > max_log_domain_gpu {
            let line = format!(
                "{:<6} {:<6} {:<6} {:>10.1} {:>10} {:>10} {:>8} {:>8}  domain 2^{log_domain} > limit",
                n, f, r, cpu_time * 1000.0, "-", "-", "-", "-"
            );
            println!("{line}");
            let mut file = std::fs::OpenOptions::new()
                .create(true).append(true).open(results_path).unwrap();
            writeln!(file, "{line}").unwrap();
            continue;
        }

        // Run GPU (commit fused, rounds NOT fused)
        let gpu_time = std::process::Command::new(exe)
            .args([&n.to_string(), &f.to_string(), &r.to_string(), "gpu"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .ok()
            .and_then(|c| c.wait_with_output().ok())
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()?.trim().parse::<f64>().ok()
                } else {
                    None
                }
            });

        // Run GPU (commit fused + rounds fused)
        let fused_time = std::process::Command::new(exe)
            .args([&n.to_string(), &f.to_string(), &r.to_string(), "gpu_fused"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .ok()
            .and_then(|c| c.wait_with_output().ok())
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()?.trim().parse::<f64>().ok()
                } else {
                    None
                }
            });

        // Run GPU (fused + GPU PoW grinding)
        let grind_time = std::process::Command::new(exe)
            .args([&n.to_string(), &f.to_string(), &r.to_string(), "gpu_grind"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .ok()
            .and_then(|c| c.wait_with_output().ok())
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8(o.stdout).ok()?.trim().parse::<f64>().ok()
                } else {
                    None
                }
            });

        let gpu_str = gpu_time
            .map(|g| format!("{:>10.1}", g * 1000.0))
            .unwrap_or_else(|| format!("{:>10}", "fail"));
        let fused_str = fused_time
            .map(|g| format!("{:>10.1}", g * 1000.0))
            .unwrap_or_else(|| format!("{:>10}", "fail"));
        let grind_str = grind_time
            .map(|g| format!("{:>10.1}", g * 1000.0))
            .unwrap_or_else(|| format!("{:>10}", "fail"));
        let gpu_speedup = gpu_time
            .map(|g| format!("{:>7.2}x", cpu_time / g))
            .unwrap_or_else(|| format!("{:>8}", "-"));
        let fused_speedup = fused_time
            .map(|g| format!("{:>7.2}x", cpu_time / g))
            .unwrap_or_else(|| format!("{:>8}", "-"));
        let grind_speedup = grind_time
            .map(|g| format!("{:>7.2}x", cpu_time / g))
            .unwrap_or_else(|| format!("{:>8}", "-"));

        let line = format!(
            "{:<6} {:<6} {:<6} {:>10.1} {} {} {} {} {} {}",
            n, f, r, cpu_time * 1000.0, gpu_str, fused_str, grind_str,
            gpu_speedup, fused_speedup, grind_speedup
        );

        println!("{line}");
        let mut file = std::fs::OpenOptions::new()
            .create(true).append(true).open(results_path).unwrap();
        writeln!(file, "{line}").unwrap();
    }
}
