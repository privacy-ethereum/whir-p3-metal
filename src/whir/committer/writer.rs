use core::ops::Deref;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{Matrix, dense::RowMajorMatrixView};
use p3_multilinear_util::point::Point;
use tracing::{info_span, instrument};

use crate::{
    constraints::statement::initial::InitialStatement,
    fiat_shamir::errors::FiatShamirError,
    parameters::WhirConfig,
    whir::{committer::DenseMatrix, proof::WhirProof},
};

/// Responsible for committing polynomials using a Merkle-based scheme.
///
/// The `Committer` processes a polynomial, expands and folds its evaluations,
/// and constructs a Merkle tree from the resulting values.
///
/// It provides a commitment that can be used for proof generation and verification.
#[derive(Debug)]
pub struct CommitmentWriter<'a, EF, F, MT: Mmcs<F>, Challenger>(
    /// Reference to the WHIR protocol configuration.
    &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> CommitmentWriter<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    /// Create a new writer that borrows the WHIR protocol configuration.
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    #[instrument(skip_all)]
    pub fn commit<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &mut InitialStatement<F, EF>,
    ) -> Result<MT::ProverData<DenseMatrix<F>>, FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = statement.num_variables();
            let mut mat = RowMajorMatrixView::new(
                statement.poly.as_slice(),
                1 << (num_vars - self.folding_factor.at_round(0)),
            )
            .transpose();
            mat.pad_to_height(
                1 << (num_vars + self.starting_log_inv_rate - self.folding_factor.at_round(0)),
                F::ZERO,
            );
            mat
        });

        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());

        let (root, prover_data) =
            info_span!("commit_matrix").in_scope(|| self.mmcs.commit_matrix(folded_matrix));

        proof.initial_commitment = Some(root.clone());
        challenger.observe(root);

        (0..self.0.commitment_ood_samples).for_each(|_| {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = info_span!("ood evaluation").in_scope(|| statement.evaluate(&point));
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
        });

        Ok(prover_data)
    }
}

/// Fused DFT+Merkle commit path for MMCS types that support `DftCommitFusion`.
#[cfg(feature = "gpu-metal")]
impl<'a, EF, F, MT, Challenger> CommitmentWriter<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: crate::gpu_dft::DftCommitFusion<F>,
{
    /// Like `commit`, but attempts to fuse DFT and Merkle tree construction
    /// in a single GPU command buffer. Falls back to separate DFT + commit
    /// if the MMCS doesn't support fusion or the matrix is too small.
    #[instrument(skip_all)]
    pub fn commit_fused<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &mut InitialStatement<F, EF>,
    ) -> Result<MT::ProverData<DenseMatrix<F>>, FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_vars = statement.num_variables();
        let fold0 = self.folding_factor.at_round(0);
        let in_cols = 1 << (num_vars - fold0);
        let in_rows = 1 << fold0;
        let padded_height = 1 << (num_vars + self.starting_log_inv_rate - fold0);

        let (root, prover_data) =
            if let Some(result) = info_span!("fused_transpose_dft_commit").in_scope(|| {
                self.mmcs.transpose_pad_dft_and_commit(
                    statement.poly.as_slice(), in_rows, in_cols, padded_height,
                )
            }) {
                result
            } else {
                let padded = info_span!("transpose & pad").in_scope(|| {
                    let mut mat = RowMajorMatrixView::new(
                        statement.poly.as_slice(), in_cols,
                    )
                    .transpose();
                    mat.pad_to_height(padded_height, F::ZERO);
                    mat
                });

                match info_span!("fused_dft_commit")
                    .in_scope(|| self.mmcs.dft_and_commit(padded))
                {
                    Ok((root, tree)) => (root, tree),
                    Err(padded) => {
                        let folded = info_span!("dft", height = padded.height(), width = padded.width())
                            .in_scope(|| dft.dft_batch(padded).to_row_major_matrix());
                        info_span!("commit_matrix").in_scope(|| self.mmcs.commit_matrix(folded))
                    }
                }
            };

        proof.initial_commitment = Some(root.clone());
        challenger.observe(root);

        (0..self.0.commitment_ood_samples).for_each(|_| {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = info_span!("ood evaluation").in_scope(|| statement.evaluate(&point));
            proof.initial_ood_answers.push(eval);
            challenger.observe_algebra_element(eval);
        });

        Ok(prover_data)
    }
}

impl<EF, F, MT: Mmcs<F>, Challenger> Deref for CommitmentWriter<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::*;
    use crate::{
        BabyBearDft,
        fiat_shamir::domain_separator::DomainSeparator,
        parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy},
    };

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

    #[test]
    fn test_basic_commitment() {
        // Set up Whir protocol parameters.
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        // Define multivariate parameters for the polynomial.
        let params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        // Generate a random polynomial with 32 coefficients.
        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = Poly::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        // Set up the DomainSeparator and initialize a ProverState narg_string.
        let mut domainsep: DomainSeparator<F, F> = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);
        domainsep.add_whir_proof::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        // Run the Commitment Phase
        let committer = CommitmentWriter::new(&params);
        let dft = BabyBearDft::default();
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();

        // Ensure OOD (out-of-domain) points are generated.
        assert!(!statement.is_empty(), "OOD points should be generated");

        // Validate the number of generated OOD points.
        assert_eq!(
            statement.len(),
            params.commitment_ood_samples,
            "OOD points count should match expected samples"
        );

        // Check that OOD answers match expected evaluations
        let poly = &statement.poly;
        let statement = statement.normalize();
        for (i, (ood_point, ood_eval)) in statement.iter().enumerate() {
            let expected_eval = poly.eval_base(ood_point);
            assert_eq!(
                *ood_eval, expected_eval,
                "OOD answer at index {i} should match expected evaluation"
            );
        }
    }

    #[test]
    fn test_large_polynomial() {
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 10;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = Poly::<BabyBear>::new(vec![rng.random(); 1024]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));
        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = BabyBearDft::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();
    }

    #[test]
    fn test_commitment_without_ood_samples() {
        let security_level = 100;
        let pow_bits = 20;
        let num_variables = 5;
        let starting_rate = 1;
        let folding_factor = 4;
        let first_round_folding_factor = 4;

        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);

        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        let whir_params = ProtocolParameters {
            security_level,
            pow_bits,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::ConstantFromSecondRound(
                first_round_folding_factor,
                folding_factor,
            ),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: starting_rate,
        };

        let mut params =
            WhirConfig::<F, F, MyMmcs, MyChallenger>::new(num_variables, whir_params.clone());

        // Explicitly set OOD samples to 0
        params.commitment_ood_samples = 0;

        let mut rng = SmallRng::seed_from_u64(1);
        let polynomial = Poly::<BabyBear>::new(vec![rng.random(); 32]);

        let mut proof =
            WhirProof::<F, F, MyMmcs>::from_protocol_parameters(&whir_params, num_variables);

        let mut domainsep = DomainSeparator::new(vec![]);
        domainsep.commit_statement::<_, _, 8>(&params);

        let mut rng = SmallRng::seed_from_u64(1);
        let mut challenger = MyChallenger::new(Perm::new_from_rng_128(&mut rng));

        domainsep.observe_domain_separator(&mut challenger);

        let mut statement = params.initial_statement(polynomial, SumcheckStrategy::Classic);
        let dft = BabyBearDft::default();
        let committer = CommitmentWriter::new(&params);
        let _ = committer
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
            .unwrap();

        assert!(
            statement.is_empty(),
            "There should be no OOD points when committment_ood_samples is 0"
        );
    }
}
