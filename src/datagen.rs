use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use statrs::distribution::Beta;

#[derive(Debug)]
pub struct BetaParams {
    pub alpha: f64,
    pub beta: f64
}

#[derive(Debug)]
pub struct BinaryClassParams {
    pub class0: BetaParams,
    pub class1: BetaParams
}

#[derive(Debug)]
pub struct BinaryClassScores {
    pub class0: Vec<f64>,
    pub class1: Vec<f64>
}

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