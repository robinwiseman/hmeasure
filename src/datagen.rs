/*!
The datagen module provides structs for generating dummy data for binary classifier scores.
The purpose of the module is to enable consistent, reproducible tests and examples for the
hmeasure module. The ChaCha8Rng is employed to enable seeding of the random number generator
underlying the data generation. The dummy data is generated using Beta distributions for the pair
of classifier scores, where the parameters of the class distributions can be varied in order to example
the performance of the H-Measure as the class0 and class1 score distributions are made more/less
distinguishable.
*/
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Beta;

/**
A struct to hold the Beta distribution parameters.
 */
#[derive(Debug)]
pub struct BetaParams {
    pub alpha: f64,
    pub beta: f64
}

/**
A struct to hold the Beta distribution parameters for each class.
*/
#[derive(Debug)]
pub struct BinaryClassParams {
    pub class0: BetaParams,
    pub class1: BetaParams
}

/**
A struct to hold the scores for each binary class.
*/
#[derive(Debug)]
pub struct BinaryClassScores {
    pub class0: Vec<f64>,
    pub class1: Vec<f64>
}

/**
A struct to enable consistent, reproducible, generation of scores for a pair of binary classes.
*/
#[derive(Debug)]
pub struct BinaryClassifierScores<'a> {
    class_params: &'a BinaryClassParams,
    c0_sample_size: usize,
    c1_sample_size: usize,
    pub scores: BinaryClassScores
}

impl BinaryClassifierScores<'_> {
    pub fn new<'a>(class_params: &'a BinaryClassParams, c0_sample_size: usize,
            c1_sample_size: usize, rng: &'a mut ChaCha8Rng) -> BinaryClassifierScores<'a> {
        let c0s : Vec<f64> = vec![0.0;c0_sample_size];
        let c1s : Vec<f64> = vec![0.0;c1_sample_size];
        let bin_class_scores = BinaryClassScores{
            class0: c0s,
            class1: c1s
        };
        let mut bcs = BinaryClassifierScores{
            class_params: &class_params,
            c0_sample_size: c0_sample_size,
            c1_sample_size: c1_sample_size,
            scores: bin_class_scores
        };
        bcs.generate_samples(rng);
        bcs
    }
    pub fn generate_samples(&mut self, mut rng: &mut ChaCha8Rng) {
        for i in 0..self.c0_sample_size {
            let beta_0 = Beta::new(self.class_params.class0.alpha, self.class_params.class0.beta).unwrap();
            let v_0 = beta_0.sample(&mut rng);
            self.scores.class0[i] = v_0;
        };
        for i in 0..self.c1_sample_size {
            let beta_1 = Beta::new(self.class_params.class1.alpha, self.class_params.class1.beta).unwrap();
            let v_1 = beta_1.sample(&mut rng);
            self.scores.class1[i] = v_1;
        }
    }
    pub fn generate_rng(seed: u64) -> ChaCha8Rng{
        let rng = ChaCha8Rng::seed_from_u64(seed);
        rng
    }
}