use hmeasure::{CostRatioDensity, HMeasure};
use hmeasure::{BetaParams, BinaryClassParams, BinaryClassifierScores};

/**
An example of the hmeasure and datagen modules in action.
*/
pub fn example_hmeasure(){
    /* Choose beta parameters for the model scores distributions that illustrate
    the case of a model that is able to reasonably discriminate between the two classes
    => the score distributions generated by the model are fairly distinct, albeit not
     completely partitioned.*/
    let mut c0_a: f64 = 2.0;
    let mut c1_a: f64 = 6.0;
    let mut c0_b: f64 = 6.0;
    let mut c1_b: f64 = 2.0;
    // illustrate different population size in each class: mimicking an imbalanced classification problem.
    let c0_n: usize = 2000;
    let c1_n: usize = 1600;
    // seed the random number generator for reproducibility.
    let mut rng = BinaryClassifierScores::generate_rng(13);
    /* Gradually update the sample score distribution parameters to make the class 0 and class 1 score
    distributions less distinguishable down to being indistinguishable (they have the same distribution).
    Observe the monotonic decrease in H-measure as the distributions trend towards indistinguishability.*/
    for i in 0..9 {
        // generate dummy score data for the pair of binary classes: class0 and class1
        let class0_params = BetaParams { alpha: c0_a, beta: c0_b };
        let class1_params = BetaParams { alpha: c1_a, beta: c1_b };
        let bcp = &BinaryClassParams { class0: class0_params, class1: class1_params };
        let mut bcs = BinaryClassifierScores::new(&bcp, c0_n, c1_n, &mut rng);

        // specify a cost ratio density
        let cost_density_params = BetaParams { alpha: 2.0, beta: 2.0 };
        let crd = CostRatioDensity::new(cost_density_params);

        // calculate the H-Measure given the cost ratio density and scores
        let mut hm = HMeasure::new(crd, None, None);
        let scores = &mut bcs.scores;
        let hmr = hm.h_measure(scores);
        println!("H-Measure results: {:?} : {:?}", (c0_a, c0_b, c1_a, c1_b), hmr.h);
        let result_set = format!("{}_{}_{}_{}_{}", c0_a, c0_b, c1_a, c1_b, i);
        let file_path = "./";
        // save the results to file
        hmr.save(result_set.as_str(), file_path);
        /*Update the class distribution parameters to gradually make the pair of distributions for class0 and class1
        less distinct.*/
        c0_a += 0.25;
        c0_b -= 0.25;
        c1_a -= 0.25;
        c1_b += 0.25;
    }
}

fn main(){
    example_hmeasure()
}
