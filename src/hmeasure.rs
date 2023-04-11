use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use array2d::Array2D;

use crate::datagen;
use datagen::{BetaParams, BinaryClassScores};


#[derive(Debug)]
pub struct CostRatioDensity{
    pub c_density_obj: Beta
}

impl CostRatioDensity {
    pub fn new(bparms: BetaParams) -> CostRatioDensity {
        CostRatioDensity {
            c_density_obj: Beta::new(bparms.alpha, bparms.beta).unwrap()
        }
    }
    pub fn uc(&self, cost: f64) -> f64 {
        cost*self.c_density_obj.pdf(cost)
    }
    pub fn u1mc(&self, cost: f64) -> f64 {
        (1.0-cost)*self.c_density_obj.pdf(cost)
    }
    pub fn cdf(&self, cost: f64) -> f64 {
        self.c_density_obj.cdf(cost)
    }
    pub fn pdf(&self, cost: f64) -> f64 {
        self.c_density_obj.pdf(cost)
    }
}


#[derive(Debug)]
pub struct HMeasure{
    cost_distribution: CostRatioDensity,
    class0_prior: Option<f64>,
    class1_prior: Option<f64>,
    c0_num: Option<usize>,
    c1_num: Option<usize>,
    pub h: Option<f64>
}

impl HMeasure{
    pub fn new(cost_distribution: CostRatioDensity, class0_prior: Option<f64>, class1_prior: Option<f64>) -> HMeasure {
        let mut mh = HMeasure {
            cost_distribution: cost_distribution,
            class0_prior: class0_prior,
            class1_prior: class1_prior,
            c0_num: None,
            c1_num: None,
            h: None
        };

        mh
    }

    pub fn h_measure(&mut self, scores: BinaryClassScores) -> f64 {
        let mut c0_scores = scores.class0;
        let mut c1_scores = scores.class1;
        self.c0_num = Some(c0_scores.len());
        self.c1_num = Some(c0_scores.len());
        let num_scores = self.c0_num.unwrap_or(0)+self.c1_num.unwrap_or(0);
        if self.class0_prior.is_none() || self.class1_prior.is_none() {
            self.class0_prior = Some(self.c0_num.unwrap_or(0) as f64/num_scores as f64);
            self.class1_prior = Some(self.c1_num.unwrap_or(0) as f64/num_scores as f64);
        }
        c0_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c1_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if self.h.is_none(){
            let (c_scores, c_classes) = self.merge_scores(c0_scores, c1_scores);
            let roc_curve = self.build_roc(&c_scores, &c_classes);
        }

        1.0
    }

    fn merge_scores(&self, c0: Vec<f64>, c1: Vec<f64>) -> (Vec<f64>, Vec<u8>) {
        let mut c0_i = 0;
        let mut c1_i = 0;
        let mut cm_i = 0;
        let c0_num= self.c0_num.unwrap_or(0);
        let c1_num= self.c1_num.unwrap_or(0);
        let vec_size = c0_num + c1_num;
        let mut c_scores = vec![0.0;vec_size];
        let mut c_classes = vec![0;vec_size];

        while c0_i < c0_num || c1_i < c1_num {
            if c0_i < c0_num && c1_i < c1_num {
                if c0[c0_i] <= c1[c1_i] {
                    c_scores[cm_i] = c0[c0_i];
                    c_classes[cm_i] = 0;
                    c0_i += 1;
                    cm_i += 1;
                } else if c1[c1_i] <= c0[c0_i] {
                    c_scores[cm_i] = c1[c1_i];
                    c_classes[cm_i] = 1;
                    c1_i += 1;
                    cm_i += 1;
                }
            } else if c0_i < c0_num && c1_i >= c1_num {
                c_scores[cm_i] = c0[c0_i];
                c_classes[cm_i] = 0;
                c0_i += 1;
                cm_i += 1;
            } else if c1_i < c1_num && c0_i >= c0_num {
                c_scores[cm_i] = c1[c1_i];
                c_classes[cm_i] = 1;
                c1_i += 1;
                cm_i += 1;
            }
        }

        (c_scores, c_classes)
    }

    fn build_roc(&self, c_scores: &Vec<f64>, c_classes: &Vec<u8>) -> Array2D<f64> {
        let num_rows = c_scores.len();
        let num_cols = 2;
        let mut roc_points = Array2D::filled_with(0.0, num_rows, num_cols);
        let mut roc_point = [0.0, 0.0];
        let mut i = 0;
        while i < num_rows {
            let i_score = c_scores[i];
            let (duplicate_0, duplicate_1) = HMeasure::duplicate_classes(c_scores, c_classes, i_score);
            let duplicate_i = duplicate_0 + duplicate_1;
            roc_point[0] += duplicate_1 as f64 / self.c1_num.unwrap_or(1) as f64;
            roc_point[1] += duplicate_0 as f64 / self.c0_num.unwrap_or(1) as f64;
            roc_points[(i,0)] = roc_point[0];
            roc_points[(i,1)] = roc_point[1];
            i += duplicate_i;
        }
        roc_points
    }

    fn duplicate_classes(scores: &Vec<f64>, classes: &Vec<u8>, target: f64) -> (usize,usize) {
        let mut dup_class_0 = vec![];
        let mut dup_class_1 = vec![];
        for (i, i_score) in scores.iter().enumerate() {
            if *i_score == target {
                if classes[i] == 0 {
                    dup_class_0.push(classes[i]);
                }
                else {
                    dup_class_1.push(classes[i]);
                }
            }
        }

        (dup_class_0.len(), dup_class_1.len())
    }
}

