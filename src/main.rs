use rand::thread_rng;
use rand_distr::{Distribution, Beta};

#[derive(Debug)]
struct BinaryClassParams {
    class0_alpha: f64,
    class1_alpha: f64,
    class0_beta: f64,
    class1_beta: f64
}

#[derive(Debug)]
struct BinaryClassifierScores<'a> {
    class_params: &'a BinaryClassParams,
    c0_sample_size: usize,
    c1_sample_size: usize,
    c0_scores: Vec<f64>,
    c1_scores: Vec<f64>
}

impl BinaryClassifierScores<'_> {
    fn new(class_params: &BinaryClassParams, c0_sample_size: usize,
            c1_sample_size: usize) -> BinaryClassifierScores {
        let c0s : Vec<f64> = vec![0.0;c0_sample_size];
        let c1s : Vec<f64> = vec![0.0;c1_sample_size];
        let mut bcs = BinaryClassifierScores{
            class_params: &class_params,
            c0_sample_size: c0_sample_size,
            c1_sample_size: c1_sample_size,
            c0_scores: c0s,
            c1_scores: c1s
        };
        bcs.generate_samples();
        bcs
    }
    fn generate_samples(&mut self) {
        for i in 0..self.c0_sample_size {
            let beta_0 = Beta::new(self.class_params.class0_alpha, self.class_params.class0_beta).unwrap();
            let v_0 = beta_0.sample(&mut thread_rng());
            self.c0_scores[i] = v_0;
        };
        for i in 0..self.c1_sample_size {
            let beta_1 = Beta::new(self.class_params.class1_alpha, self.class_params.class1_beta).unwrap();
            let v_1 = beta_1.sample(&mut thread_rng());
            self.c1_scores[i] = v_1;
        }
    }
}

fn main() {
    let c0_a: f64 = 2.0;
    let c1_a: f64 = 1.0;
    let c0_b: f64 = 2.0;
    let c1_b: f64 = 1.0;
    let c0_n: usize = 20;
    let c1_n: usize = 20;
    let bcp = &BinaryClassParams { class0_alpha: c0_a,
                                class1_alpha: c1_a,
                                class0_beta: c0_b,
                                class1_beta: c1_b
                                };
    let bcs = BinaryClassifierScores::new(&bcp,c0_n,c1_n);
    println!("{:?}", bcp);
    println!("{:?}", bcs);
}
