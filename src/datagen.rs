/*!
The datagen module provides structs for generating dummy data for binary classifier scores.
The purpose of the module is to enable consistent, reproducible tests and examples for the
hmeasure module. The ChaCha8Rng is employed to enable seeding of the random number generator
underlying the data generation. The dummy data is generated using Beta distributions for the pair
of classifier scores, where the parameters of the class distributions can be varied in order to example
the performance of the H-Measure as the class0 and class1 score distributions are made more/less
distinguishable.
The module is only compiled when the datagen feature is enabled.
*/
use rand::distr::Distribution;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Beta;

use crate::hm::{BetaParams, BinaryClassScores, HMeasureError};

/**
A struct to hold the Beta distribution parameters for each class.
*/
#[derive(Debug)]
pub struct BinaryClassParams {
    /** The parameters of the Beta distribution of the class 0 scores. */
    pub class0: BetaParams,
    /** The parameters of the Beta distribution of the class 1 scores. */
    pub class1: BetaParams,
}

/**
A struct to enable consistent, reproducible, generation of scores for a pair of binary classes.
*/
#[derive(Debug)]
pub struct BinaryClassifierScores<'a> {
    class_params: &'a BinaryClassParams,
    c0_sample_size: usize,
    c1_sample_size: usize,
    /** The generated scores of each class. */
    pub scores: BinaryClassScores,
}

impl BinaryClassifierScores<'_> {
    /**
    Generates c0_sample_size class 0 scores and c1_sample_size class 1 scores, drawing from rng.
    Returns HMeasureError::InvalidBetaParams unless the Beta parameters of both classes are finite and
    greater than zero.
    */
    pub fn new<'a>(
        class_params: &'a BinaryClassParams,
        c0_sample_size: usize,
        c1_sample_size: usize,
        rng: &'a mut ChaCha8Rng,
    ) -> Result<BinaryClassifierScores<'a>, HMeasureError> {
        let c0s: Vec<f64> = vec![0.0; c0_sample_size];
        let c1s: Vec<f64> = vec![0.0; c1_sample_size];
        let bin_class_scores = BinaryClassScores {
            class0: c0s,
            class1: c1s,
        };
        let mut bcs = BinaryClassifierScores {
            class_params,
            c0_sample_size,
            c1_sample_size,
            scores: bin_class_scores,
        };
        bcs.generate_samples(rng)?;
        Ok(bcs)
    }
    /**
    Replaces the scores with a fresh sample from the class distributions, drawing from rng.
    */
    pub fn generate_samples(&mut self, mut rng: &mut ChaCha8Rng) -> Result<(), HMeasureError> {
        let beta_0 = beta_distribution(&self.class_params.class0)?;
        for i in 0..self.c0_sample_size {
            let v_0 = beta_0.sample(&mut rng);
            self.scores.class0[i] = v_0;
        }
        let beta_1 = beta_distribution(&self.class_params.class1)?;
        for i in 0..self.c1_sample_size {
            let v_1 = beta_1.sample(&mut rng);
            self.scores.class1[i] = v_1;
        }
        Ok(())
    }
    /**
    A random number generator seeded with seed, so that the generated scores are reproducible.
    */
    pub fn generate_rng(seed: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(seed)
    }
}

fn beta_distribution(params: &BetaParams) -> Result<Beta, HMeasureError> {
    Beta::new(params.alpha, params.beta).map_err(|_| HMeasureError::InvalidBetaParams {
        alpha: params.alpha,
        beta: params.beta,
    })
}
