mod datagen;
mod hmeasure;

use crate::hmeasure::CostRatioDensity;
pub use self::datagen::{BetaParams, BinaryClassParams, BinaryClassifierScores};


fn main() {
    let c0_a: f64 = 2.0;
    let c1_a: f64 = 1.0;
    let c0_b: f64 = 2.0;
    let c1_b: f64 = 1.0;
    let c0_n: usize = 20;
    let c1_n: usize = 20;
    let class0_params = BetaParams {alpha: c0_a, beta: c0_b};
    let class1_params = BetaParams {alpha: c1_a, beta: c1_b};
    let bcp = &BinaryClassParams { class0: class0_params, class1: class1_params};
    let bcs = BinaryClassifierScores::new(&bcp,c0_n,c1_n);
    let b_params = BetaParams {alpha: 1.1, beta: 1.9};
    let crd = &CostRatioDensity::new(b_params);
    println!("{:?}", crd.cdf(0.6));
}
