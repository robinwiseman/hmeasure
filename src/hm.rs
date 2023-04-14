/*!
The hmeasure module provides an implementation of:
CostRatioDensity: a struct to enable generating cost distributions using the Beta distribution.
HMeasure: a struct to enable generating the H-Measure metric, given a cost distribution and binary classification scores
for a pair of classes (class0 and class1). Priors for the class scores may be optionally specified, or inferred from
the data provided.
The scores are generated exogenously by a model (the hmeasure module is agnostic to the source of the scores).
HMeasureResults: a struct to receive the results of the HMeasure calculation and enable saving to file.
*/
use std::io::Write;
use std::fs::{OpenOptions, File};

use statrs::distribution::{Beta, Continuous, ContinuousCDF};
use array2d::Array2D;
use quadrature::double_exponential;

use crate::datagen;
use datagen::{BetaParams, BinaryClassScores};

/**
CostRatioDensity: a struct to enable generating cost distributions using the Beta distribution
*/
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
    /**
    A convenience helper function for a component of the H-Measure numerical integration.
    */
    pub fn uc(&self, cost: f64) -> f64 {
        cost*self.c_density_obj.pdf(cost)
    }
    /**
    A convenience helper function for a component of the H-Measure numerical integration.
    */
    pub fn u1mc(&self, cost: f64) -> f64 {
        (1.0-cost)*self.c_density_obj.pdf(cost)
    }
    /**
    The cumulative probability density function of the CostRatio distribution.
    */
    pub fn cdf(&self, cost: f64) -> f64 {self.c_density_obj.cdf(cost) }
    /**
    The probability density function of the CostRatio distribution.
    */
    pub fn pdf(&self, cost: f64) -> f64 {
        self.c_density_obj.pdf(cost)
    }
}

/**
HMeasure: a struct to enable generating the H-Measure metric, given a cost distribution and binary classification scores
for a pair of classes (class0 and class1). Priors for the class scores may be optionally specified, or inferred from
the data provided.
The scores are generated exogenously by a model (the hmeasure module is agnostic to the source of the scores).
*/
#[derive(Debug)]
pub struct HMeasure{
    cost_distribution: CostRatioDensity,
    class0_prior: Option<f64>,
    class1_prior: Option<f64>,
    c0_num: Option<usize>,
    c1_num: Option<usize>,
    pub h: Option<f64>
}

/**
HMeasureResults: a struct to receive the results of the HMeasure calculation and enable saving to file
*/
#[derive(Debug)]
pub struct HMeasureResults{
    pub h: f64,
    pub convex_hull: Vec<Vec<f64>>,
    pub roc_curve: Vec<Vec<f64>>,
    pub int_components: Vec<Vec<f64>>,
    pub class0_score: Vec<f64>,
    pub class1_score: Vec<f64>
}

/**
A trait for saving model results to file given a file handle.
*/
pub trait SaveData<File> {
    fn save(&self, f: &mut File);
}

impl<File: Write> SaveData<File> for Vec<f64>{
    fn save(&self, f: &mut File){
        let sep = ",";
        for (idx, element) in self.iter().enumerate() {
                f.write_all(idx.to_string().as_bytes()).expect("the unexpected");
                f.write_all(sep.as_bytes()).expect("the unexpected");
                f.write_all(element.to_string().as_bytes()).expect("the unexpected");
                f.write_all(b"\n").expect("the unexpected");
            }
    }
}

impl<File: Write> SaveData<File> for Array2D<f64>{
    fn save(&self, f: &mut File){
        let num_rows = self.column_len();
        let num_cols = self.row_len();
        let mut i = 0;
        let sep = ",";
        while i < num_rows {
            let mut j = 0;
            f.write_all(i.to_string().as_bytes()).expect("the unexpected");
            f.write_all(sep.as_bytes()).expect("the unexpected");
            while j < num_cols {
                let element = self[(i,j)];
                f.write_all(element.to_string().as_bytes()).expect("the unexpected");
                if j < num_cols-1 {
                    f.write_all(sep.as_bytes()).expect("the unexpected");
                }
                j += 1;
            }
            f.write_all(b"\n").expect("the unexpected");
            i += 1;
        }
    }
}

impl<File: Write> SaveData<File> for Vec<Vec<f64>>{
    fn save(&self, f: &mut File){
        let num_rows = self.len();
        let num_cols = self[0].len();
        let mut i = 0;
        let sep = ",";
        while i < num_rows {
            let mut j = 0;
            f.write_all(i.to_string().as_bytes()).expect("the unexpected");
            f.write_all(sep.as_bytes()).expect("the unexpected");
            while j < num_cols {
                let element = self[i][j];
                f.write_all(element.to_string().as_bytes()).expect("the unexpected");
                if j < num_cols-1 {
                    f.write_all(sep.as_bytes()).expect("the unexpected");
                }
                j += 1;
            }
            f.write_all(b"\n").expect("the unexpected");
            i += 1;
        }
    }
}

/**
A generic for saving model results to file for types that have implemented the SaveData trait.
*/
pub fn write_vec<T: SaveData<File>>(vec: &T, filepath: &str, header: &str){
    let f = &mut OpenOptions::new()
        .append(true)
        .create(true)
        .open(filepath)
        .expect("the unexpected");

    f.write_all(header.as_bytes()).expect("the unexpected");
    f.write_all(b"\n").expect("the unexpected");
    vec.save(f)
}

/**
Implementation of the save function for persisting H-Measure model results to file.
*/
impl HMeasureResults{
    pub fn save(&self, result_set: &str, filepath: &str){
        let chull_name = self.result_set_path("ch", filepath, result_set);
        write_vec(&self.convex_hull, chull_name.as_str(), "idx,chull0,chull1");

        let roc_name = self.result_set_path("rc", filepath, result_set);
        write_vec(&self.roc_curve, roc_name.as_str(), "idx,roc0,roc1");

        let intcomp_name = self.result_set_path("ic", filepath, result_set);
        write_vec(&self.int_components, intcomp_name.as_str(), "idx,ic0,ic1,ic2,ic3");

        let score0_name = self.result_set_path("s0", filepath, result_set);
        write_vec(&self.class0_score, score0_name.as_str(), "idx,score");

        let score1_name = self.result_set_path("s1", filepath, result_set);
        write_vec(&self.class1_score, score1_name.as_str(), "idx,score");
    }
    pub fn result_set_path(&self, set_name: &str, filepath: &str, resultset: &str) -> String{
        let name = format!("{}/{}_{}.csv", filepath, set_name, resultset);
        name
    }
}

/**
Implementation of the HMeasure functions for calculating the H-Measure metric.
*/
impl HMeasure{
    pub fn new(cost_distribution: CostRatioDensity, class0_prior: Option<f64>, class1_prior: Option<f64>) -> HMeasure {
        HMeasure {
            cost_distribution: cost_distribution,
            class0_prior: class0_prior,
            class1_prior: class1_prior,
            c0_num: None,
            c1_num: None,
            h: None
        }
    }

    pub fn h_measure(&mut self, scores: &mut BinaryClassScores) -> HMeasureResults {
        self.c0_num = Some(scores.class0.len());
        self.c1_num = Some(scores.class1.len());
        let num_scores = self.c0_num.unwrap_or(0)+self.c1_num.unwrap_or(0);
        if self.class0_prior.is_none() || self.class1_prior.is_none() {
            self.class0_prior = Some(self.c0_num.unwrap_or(0) as f64/num_scores as f64);
            self.class1_prior = Some(self.c1_num.unwrap_or(0) as f64/num_scores as f64);
        }
        scores.class0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scores.class1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (c_scores, c_classes) = self.merge_scores(&scores.class0, &scores.class1);
        let roc_curve = self.build_roc(&c_scores, &c_classes);
        let (convex_hull, int_components) = self.build_chull(&roc_curve);
        self.h = self._build_h(&int_components);

        let h = self.h.unwrap();
        let class0_score = scores.class0.clone();
        let class1_score = scores.class1.clone();
        HMeasureResults{ h,
                        convex_hull,
                        roc_curve,
                        int_components,
                        class0_score,
                        class1_score
        }
    }

    fn merge_scores(&self, c0: &Vec<f64>, c1: &Vec<f64>) -> (Vec<f64>, Vec<u8>) {
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

    fn build_roc(&self, c_scores: &Vec<f64>, c_classes: &Vec<u8>) -> Vec<Vec<f64>> {
        let num_rows = c_scores.len();
        let mut roc_points: Vec<Vec<f64>> = vec![];
        let mut roc_point = [0.0, 0.0];
        roc_points.push(vec![roc_point[0], roc_point[1]]);
        let mut i = 0;
        while i < num_rows {
            let i_score = c_scores[i];
            let (duplicate_0, duplicate_1) = HMeasure::duplicate_classes(c_scores, c_classes, i_score);
            let duplicate_i = duplicate_0 + duplicate_1;
            roc_point[0] += duplicate_1 as f64 / self.c1_num.unwrap_or(1) as f64;
            roc_point[1] += duplicate_0 as f64 / self.c0_num.unwrap_or(1) as f64;
            roc_points.push(vec![roc_point[0], roc_point[1]]);
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

    fn build_chull(&self, roc_c: &Vec<Vec<f64>>) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let num_rows = roc_c.len();
        let mut chull: Vec<Vec<f64>> = vec![];
        let mut int_components: Vec<Vec<f64>> = vec![];
        let mut cval_loc_prior = [0.0, 0.0];
        let mut cval_prior = 0.0;
        let mut i = 0;
        while i < num_rows-1 {
            let cvals = self._cvals(&roc_c, i);
            let c_argmin = HMeasure::argmin(&cvals);
            let cval_loc = [roc_c[i+c_argmin+1][0],roc_c[i+c_argmin+1][1]];
            chull.push(vec![cval_loc[0],cval_loc[1]]);
            int_components.push(vec![cvals[c_argmin], cval_prior, cval_loc_prior[0], cval_loc_prior[1]]);
            cval_loc_prior = cval_loc;
            cval_prior = cvals[c_argmin];
            i += c_argmin+1;
        }

        int_components.push(vec![1.0, cval_prior, cval_loc_prior[0], cval_loc_prior[1]]);

        (chull, int_components)
    }

    fn _cvals(&self, roc_c: &Vec<Vec<f64>>, current_idx: usize) -> Vec<f64> {
        let mut cvals: Vec<f64> = vec![];
        let current_point: [f64;2] = [roc_c[current_idx][0],roc_c[current_idx][1]];
        let mut i_point_idx = current_idx + 1;
        while i_point_idx < roc_c.len() {
            let i_point: [f64;2] = [roc_c[i_point_idx][0],roc_c[i_point_idx][1]];
            cvals.push(self._cval(&current_point, &i_point));
            i_point_idx += 1;
        }
        cvals
    }

    fn _cval(&self, current_point: &[f64; 2], i_point: &[f64;2]) -> f64 {
        let numerator = self.class1_prior.unwrap_or(0.5)*(i_point[0]-current_point[0]);
        let denominator = self.class0_prior.unwrap_or(0.5)*(i_point[1]-current_point[1]) + self.class1_prior.unwrap_or(0.5)*(i_point[0]-current_point[0]);
        numerator/denominator
    }

    fn argmin(v: &Vec<f64>) -> usize {
        let mut min_idx = 0;
        let mut min_val = v[0];
        for (i, &val) in v.iter().enumerate() {
            if val < min_val {
                min_idx = i;
                min_val = val;
            }
        }
        min_idx
    }

    fn _build_h(&self, int_components: &Vec<Vec<f64>>) -> Option<f64> {
        let l = self._build_l(&int_components).unwrap();
        let l_max = self._l_max().unwrap();
        // println!("l : l_max ~ {:?} : {:?}", l, l_max);
        Some(1.0-l/l_max)
    }

    fn _build_l(&self, int_components: &Vec<Vec<f64>>) -> Option<f64> {
        let mut l = 0.0;
        let mut i = 0;
        while i < int_components.len() {
            let int_vals = &int_components[i];
            l += self._int_l_step(int_vals);
            i += 1
        }
        Some(l)
    }

    fn _int_l_step(&self, int_vals: &Vec<f64>) -> f64 {
        let r0i = int_vals[3];
        let r1i = int_vals[2];
        let ci = int_vals[1];
        let cip1 = int_vals[0];
        let c0_prior = self.class0_prior.unwrap();
        let c1_prior = self.class1_prior.unwrap();
        let int_uc = double_exponential::integrate(|x| self.cost_distribution.uc(x), ci, cip1, 1e-9).integral;
        let int_u1mc = double_exponential::integrate(|x| self.cost_distribution.u1mc(x), ci, cip1, 1e-9).integral;

        let l_step = c0_prior*(1.0-r0i)*int_uc + c1_prior*r1i*int_u1mc;

        return l_step
    }

    fn _l_max(&self) -> Option<f64> {
        let c0_prior = self.class0_prior.unwrap();
        let c1_prior = self.class1_prior.unwrap();
        let mut l_max = double_exponential::integrate(|x| self.cost_distribution.uc(x), 0.0, c1_prior, 1e-9).integral*c0_prior;
        l_max += double_exponential::integrate(|x| self.cost_distribution.u1mc(x), c1_prior, 1.0,1e-9).integral*c1_prior;
        Some(l_max)
    }
}

#[cfg(test)]
mod tests {
    use super::{CostRatioDensity,HMeasure};
    use super::datagen::{BetaParams, BinaryClassifierScores, BinaryClassParams};
    #[test]
    fn assert_low_hmeasure() {
        // Choose class0 and class1 to have identical score distributions => the classifier model that generated the
        // scores is unable to discriminate between class0 and class1 (it is poor model). Demonstrate near-zero
        // H-Measure in this case
        let c0_a: f64 = 4.0;
        let c1_a: f64 = 4.0;
        let c0_b: f64 = 4.0;
        let c1_b: f64 = 4.0;
        let c0_n: usize = 2000;
        let c1_n: usize = 2000;
        // seed the random number generator for reproducibility.
        let mut rng = BinaryClassifierScores::generate_rng(13);
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
        // Note : the H-Measure will not in general be identically zero in this test, despite the
        // distributions of the class0 and class1 scores being identical. This is a natural property of
        // and random sample-based measure (the sampled values from the two distributions will in general be
        // different - although for large sample size, the effect should be small).
        assert!(hmr.h < 0.001);
    }
}