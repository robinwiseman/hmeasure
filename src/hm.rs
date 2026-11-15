/*!
The hmeasure module provides an implementation of:
CostRatioDensity: a struct to enable generating cost distributions using the Beta distribution.
CostDistribution: a trait for supplying any other cost distribution, evaluated by numerical quadrature unless
it provides its integrals in closed form.
HMeasure: a struct to enable generating the H-Measure metric, given a cost distribution and binary classification scores
for a pair of classes (class0 and class1). Priors for the class scores may be optionally specified, or inferred from
the data provided.
The scores are generated exogenously by a model (the hmeasure module is agnostic to the source of the scores).
HMeasureResults: a struct to receive the results of the HMeasure calculation and enable saving to file.
HMeasureError: the errors returned for invalid inputs.
*/
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Write};

use quadrature::double_exponential;
use statrs::distribution::{Beta, Continuous, ContinuousCDF};

/**
A struct to hold the Beta distribution parameters. Both must be finite and greater than zero.
 */
#[derive(Debug)]
pub struct BetaParams {
    /** The alpha (first shape) parameter. */
    pub alpha: f64,
    /** The beta (second shape) parameter. */
    pub beta: f64,
}

/**
A struct to hold the scores for each binary class.

Class 0 is the class the classifier is expected to score lower, and class 1 the class it is expected to
score higher. If the classes are supplied the other way round, the classifier appears to rank them worse
than chance and the H-Measure is at or near zero. Only the ordering of the scores matters, so they do
not have to be probabilities. Neither class may be empty and no score may be NaN.
*/
#[derive(Debug)]
pub struct BinaryClassScores {
    /** The scores of the observations whose true class is class 0, in any order. */
    pub class0: Vec<f64>,
    /** The scores of the observations whose true class is class 1, in any order. */
    pub class1: Vec<f64>,
}

/**
HMeasureError: the errors returned when the inputs to the H-Measure calculation are invalid.
*/
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum HMeasureError {
    /**
    The Beta distribution parameters are not both finite and greater than zero.
    */
    InvalidBetaParams {
        /** The alpha parameter that was supplied. */
        alpha: f64,
        /** The beta parameter that was supplied. */
        beta: f64,
    },
    /**
    The class priors are not both greater than zero and summing to one.
    */
    InvalidPriors {
        /** The class 0 prior that was supplied, or implied by the class 1 prior. */
        class0: f64,
        /** The class 1 prior that was supplied, or implied by the class 0 prior. */
        class1: f64,
    },
    /**
    The given class (0 or 1) has no scores.
    */
    EmptyClass {
        /** The class without scores: 0 or 1. */
        class: u8,
    },
    /**
    The given class (0 or 1) contains a NaN score.
    */
    NanScore {
        /** The class containing the NaN score: 0 or 1. */
        class: u8,
    },
}

impl fmt::Display for HMeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HMeasureError::InvalidBetaParams { alpha, beta } => write!(
                f,
                "Beta parameters must be finite and greater than zero, got alpha = {alpha}, beta = {beta}"
            ),
            HMeasureError::InvalidPriors { class0, class1 } => write!(
                f,
                "class priors must each be greater than zero and sum to one, got class0 = {class0}, class1 = {class1}"
            ),
            HMeasureError::EmptyClass { class } => write!(f, "class{class} has no scores"),
            HMeasureError::NanScore { class } => write!(f, "class{class} contains a NaN score"),
        }
    }
}

impl Error for HMeasureError {}

/**
CostRatioDensity: a struct to enable generating cost distributions using the Beta distribution

Beta(2, 2) is the cost distribution proposed by Hand for standardised comparisons between classifiers.
The H-Measure integrals over a Beta distribution are evaluated in closed form.
*/
#[derive(Debug)]
pub struct CostRatioDensity {
    c_density_obj: Beta,
    // c*pdf(c) and (1-c)*pdf(c) are proportional to the Beta(alpha+1, beta) and Beta(alpha, beta+1)
    // densities, whose CDFs give the H-Measure integrals in closed form.
    uc_cdf: Beta,
    u1mc_cdf: Beta,
}

impl CostRatioDensity {
    /**
    Returns HMeasureError::InvalidBetaParams unless alpha and beta are both finite and greater than zero.
    */
    pub fn new(bparms: BetaParams) -> Result<CostRatioDensity, HMeasureError> {
        let BetaParams { alpha, beta } = bparms;
        let invalid = |_| HMeasureError::InvalidBetaParams { alpha, beta };
        Ok(CostRatioDensity {
            c_density_obj: Beta::new(alpha, beta).map_err(invalid)?,
            uc_cdf: Beta::new(alpha + 1.0, beta).map_err(invalid)?,
            u1mc_cdf: Beta::new(alpha, beta + 1.0).map_err(invalid)?,
        })
    }
    /**
    The alpha parameter of the Beta distribution.
    */
    pub fn alpha(&self) -> f64 {
        self.c_density_obj.shape_a()
    }
    /**
    The beta parameter of the Beta distribution.
    */
    pub fn beta(&self) -> f64 {
        self.c_density_obj.shape_b()
    }
    /**
    A convenience helper function for a component of the H-Measure numerical integration.
    */
    pub fn uc(&self, cost: f64) -> f64 {
        cost * self.c_density_obj.pdf(cost)
    }
    /**
    A convenience helper function for a component of the H-Measure numerical integration.
    */
    pub fn u1mc(&self, cost: f64) -> f64 {
        (1.0 - cost) * self.c_density_obj.pdf(cost)
    }
    /**
    The cumulative probability density function of the CostRatio distribution.
    */
    pub fn cdf(&self, cost: f64) -> f64 {
        self.c_density_obj.cdf(cost)
    }
    /**
    The probability density function of the CostRatio distribution.
    */
    pub fn pdf(&self, cost: f64) -> f64 {
        self.c_density_obj.pdf(cost)
    }
}

/**
CostDistribution: a trait for the distribution of the cost ratio c over [0, 1] that weights the
misclassification losses in the H-Measure. The H-Measure is defined for any such distribution; the Beta
distribution provided by CostRatioDensity is the standard choice for comparable results.

The cost ratio c is the cost of misclassifying a class 0 observation, as a share of the combined cost of
misclassifying a class 0 observation and a class 1 observation. A distribution concentrated near zero
therefore treats errors on class 1 as the expensive ones, and one concentrated near one treats errors on
class 0 as the expensive ones.

Only `pdf` is required: the two integrals the H-Measure needs are then evaluated by numerical quadrature
between successive cost thresholds, to an absolute tolerance of 1e-9. The density does not have to
integrate to exactly one, because the H-Measure is a ratio of two integrals over it, but it should be of
order one for that tolerance to be meaningful, and smooth between thresholds. A distribution whose
integrals are known in closed form can override `uc_integral` and `u1mc_integral`, which is both faster
and exact; CostRatioDensity does this for the Beta distribution.

Any `Fn(f64) -> f64` closure is a CostDistribution, with the closure as its density:

```
use hmeasure::{BinaryClassScores, HMeasure};

let scores = BinaryClassScores {
    class0: vec![0.1, 0.3],
    class1: vec![0.2, 0.4],
};
// a bell-shaped cost density centred on a cost ratio of 0.3
let cost_density = |c: f64| (-(c - 0.3).powi(2) / 0.045).exp() / 0.375;
let results = HMeasure::new(cost_density, None, None)?.h_measure(&scores)?;
assert!((results.h - 0.5).abs() < 1e-6);
# Ok::<(), hmeasure::HMeasureError>(())
```
*/
pub trait CostDistribution {
    /**
    The density of the cost ratio at cost.
    */
    fn pdf(&self, cost: f64) -> f64;
    /**
    The integral of c * pdf(c) over [lo, hi].
    */
    fn uc_integral(&self, lo: f64, hi: f64) -> f64 {
        double_exponential::integrate(|c| c * self.pdf(c), lo, hi, QUADRATURE_TOLERANCE).integral
    }
    /**
    The integral of (1 - c) * pdf(c) over [lo, hi].
    */
    fn u1mc_integral(&self, lo: f64, hi: f64) -> f64 {
        double_exponential::integrate(|c| (1.0 - c) * self.pdf(c), lo, hi, QUADRATURE_TOLERANCE)
            .integral
    }
}

const QUADRATURE_TOLERANCE: f64 = 1e-9;

impl<F: Fn(f64) -> f64> CostDistribution for F {
    fn pdf(&self, cost: f64) -> f64 {
        self(cost)
    }
}

/**
For the Beta(a, b) density the integrals are evaluated in closed form through the Beta CDF:
the integral of c*pdf(c) over [lo, hi] is a/(a+b) * (F(hi) - F(lo)), with F the Beta(a+1, b) CDF, and
the integral of (1-c)*pdf(c) over [lo, hi] is b/(a+b) * (G(hi) - G(lo)), with G the Beta(a, b+1) CDF.
*/
impl CostDistribution for CostRatioDensity {
    fn pdf(&self, cost: f64) -> f64 {
        self.c_density_obj.pdf(cost)
    }

    fn uc_integral(&self, lo: f64, hi: f64) -> f64 {
        let (a, b) = (self.alpha(), self.beta());
        a / (a + b) * (self.uc_cdf.cdf(hi) - self.uc_cdf.cdf(lo))
    }

    fn u1mc_integral(&self, lo: f64, hi: f64) -> f64 {
        let (a, b) = (self.alpha(), self.beta());
        b / (a + b) * (self.u1mc_cdf.cdf(hi) - self.u1mc_cdf.cdf(lo))
    }
}

/**
HMeasure: a struct to enable generating the H-Measure metric, given a cost distribution and binary classification scores
for a pair of classes (class0 and class1). Priors for the class scores may be optionally specified, or inferred from
the data provided.
The scores are generated exogenously by a model (the hmeasure module is agnostic to the source of the scores).
The cost distribution is a CostRatioDensity (Beta) by default, and may be any CostDistribution.

The H-Measure runs from zero, for a classifier that separates the classes no better than chance, to one,
for a classifier that separates them perfectly. An HMeasure can be reused for any number of sets of scores.

```
use hmeasure::{BetaParams, BinaryClassScores, CostRatioDensity, HMeasure};

let scores = BinaryClassScores {
    class0: vec![0.10, 0.25, 0.30, 0.45, 0.60],
    class1: vec![0.35, 0.55, 0.70, 0.80, 0.90],
};
let cost_density = CostRatioDensity::new(BetaParams { alpha: 2.0, beta: 2.0 })?;
let mut hm = HMeasure::new(cost_density, None, None)?;
let results = hm.h_measure(&scores)?;
assert!(results.h > 0.0 && results.h < 1.0);
assert_eq!(hm.h, Some(results.h));
# Ok::<(), hmeasure::HMeasureError>(())
```
*/
#[derive(Debug)]
pub struct HMeasure<C = CostRatioDensity> {
    cost_distribution: C,
    priors: Option<(f64, f64)>,
    /** The H-Measure from the most recent call to h_measure, or None before the first call. */
    pub h: Option<f64>,
}

/**
HMeasureResults: a struct to receive the results of the HMeasure calculation and enable saving to file

Points on the ROC curve and its convex hull are [x, y] pairs, where for a score threshold t, x is the
fraction of class 1 scores at or below t and y is the fraction of class 0 scores at or below t.
*/
#[derive(Debug)]
pub struct HMeasureResults {
    /** The H-Measure: zero for no separation of the classes, one for perfect separation. */
    pub h: f64,
    /**
    The vertices of the upper convex hull of the ROC curve, in order, ending at [1, 1]. The starting
    vertex [0, 0] is left out.
    */
    pub convex_hull: Vec<Vec<f64>>,
    /** The ROC curve from [0, 0] to [1, 1], with one point for each distinct score. */
    pub roc_curve: Vec<Vec<f64>>,
    /**
    The components of the loss integral, with one row for each convex hull vertex including [0, 0].
    A row is [upper cost ratio, lower cost ratio, x, y]: the vertex [x, y] is the best operating
    point for every cost ratio between the lower and the upper value.
    */
    pub int_components: Vec<Vec<f64>>,
    /** The class 0 scores, sorted in ascending order. */
    pub class0_score: Vec<f64>,
    /** The class 1 scores, sorted in ascending order. */
    pub class1_score: Vec<f64>,
}

/**
A trait for saving model results to file given a file handle.
*/
pub trait SaveData<File> {
    fn save(&self, f: &mut File) -> io::Result<()>;
}

impl<File: Write> SaveData<File> for Vec<f64> {
    fn save(&self, f: &mut File) -> io::Result<()> {
        for (idx, element) in self.iter().enumerate() {
            writeln!(f, "{idx},{element}")?;
        }
        Ok(())
    }
}

impl<File: Write> SaveData<File> for Vec<Vec<f64>> {
    fn save(&self, f: &mut File) -> io::Result<()> {
        for (idx, row) in self.iter().enumerate() {
            let elements: Vec<String> = row.iter().map(|element| element.to_string()).collect();
            writeln!(f, "{idx},{}", elements.join(","))?;
        }
        Ok(())
    }
}

/**
A generic for saving model results to file for types that have implemented the SaveData trait.
An existing file at filepath is overwritten.
*/
pub fn write_vec<T: SaveData<File>>(vec: &T, filepath: &str, header: &str) -> io::Result<()> {
    let f = &mut File::create(filepath)?;
    writeln!(f, "{header}")?;
    vec.save(f)
}

/**
Implementation of the save function for persisting H-Measure model results to file.
*/
impl HMeasureResults {
    /**
    Writes five csv files into the existing directory filepath, overwriting any that exist: the
    convex hull (ch), ROC curve (rc), integral components (ic) and the sorted scores of each class
    (s0 and s1), each named by result_set_path. Returns the underlying io::Error if a file cannot be
    created or written.
    */
    pub fn save(&self, result_set: &str, filepath: &str) -> io::Result<()> {
        let chull_name = self.result_set_path("ch", filepath, result_set);
        write_vec(&self.convex_hull, chull_name.as_str(), "idx,chull0,chull1")?;

        let roc_name = self.result_set_path("rc", filepath, result_set);
        write_vec(&self.roc_curve, roc_name.as_str(), "idx,roc0,roc1")?;

        let intcomp_name = self.result_set_path("ic", filepath, result_set);
        write_vec(
            &self.int_components,
            intcomp_name.as_str(),
            "idx,ic0,ic1,ic2,ic3",
        )?;

        let score0_name = self.result_set_path("s0", filepath, result_set);
        write_vec(&self.class0_score, score0_name.as_str(), "idx,score")?;

        let score1_name = self.result_set_path("s1", filepath, result_set);
        write_vec(&self.class1_score, score1_name.as_str(), "idx,score")
    }
    /**
    The path of the csv file for one part of the results: filepath/set_name_resultset.csv.
    */
    pub fn result_set_path(&self, set_name: &str, filepath: &str, resultset: &str) -> String {
        let name = format!("{}/{}_{}.csv", filepath, set_name, resultset);
        name
    }
}

const PRIOR_SUM_TOLERANCE: f64 = 1e-9;

/**
Implementation of the HMeasure functions for calculating the H-Measure metric.
*/
impl<C: CostDistribution> HMeasure<C> {
    /**
    With neither prior given, the priors are inferred from the class sizes of each set of scores.
    With one prior given, the other is its complement. With both given, they must sum to one.
    Returns HMeasureError::InvalidPriors unless both priors are greater than zero and sum to one.
    */
    pub fn new(
        cost_distribution: C,
        class0_prior: Option<f64>,
        class1_prior: Option<f64>,
    ) -> Result<HMeasure<C>, HMeasureError> {
        let priors = match (class0_prior, class1_prior) {
            (None, None) => None,
            (Some(class0), None) => Some((class0, 1.0 - class0)),
            (None, Some(class1)) => Some((1.0 - class1, class1)),
            (Some(class0), Some(class1)) => Some((class0, class1)),
        };
        if let Some((class0, class1)) = priors {
            let valid = class0 > 0.0
                && class1 > 0.0
                && (class0 + class1 - 1.0).abs() <= PRIOR_SUM_TOLERANCE;
            if !valid {
                return Err(HMeasureError::InvalidPriors { class0, class1 });
            }
        }
        Ok(HMeasure {
            cost_distribution,
            priors,
            h: None,
        })
    }

    /**
    Calculates the H-Measure for the scores, which are left unchanged. Class 0 must be the class
    expected to receive the lower scores: see BinaryClassScores. Returns HMeasureError::EmptyClass if
    either class has no scores and HMeasureError::NanScore if any score is NaN.
    */
    pub fn h_measure(
        &mut self,
        scores: &BinaryClassScores,
    ) -> Result<HMeasureResults, HMeasureError> {
        for (class, class_scores) in [(0, &scores.class0), (1, &scores.class1)] {
            if class_scores.is_empty() {
                return Err(HMeasureError::EmptyClass { class });
            }
            if class_scores.iter().any(|score| score.is_nan()) {
                return Err(HMeasureError::NanScore { class });
            }
        }
        let c0_num = scores.class0.len();
        let c1_num = scores.class1.len();
        let num_scores = (c0_num + c1_num) as f64;
        let priors = self
            .priors
            .unwrap_or((c0_num as f64 / num_scores, c1_num as f64 / num_scores));

        let mut class0_score = scores.class0.clone();
        let mut class1_score = scores.class1.clone();
        class0_score.sort_by(|a, b| a.total_cmp(b));
        class1_score.sort_by(|a, b| a.total_cmp(b));
        let (c_scores, c_classes) = Self::merge_scores(&class0_score, &class1_score);
        let roc_curve = Self::build_roc(&c_scores, &c_classes, c0_num, c1_num);
        let (convex_hull, int_components) = Self::build_chull(&roc_curve, priors);
        let h = self.build_h(&int_components, priors);
        self.h = Some(h);

        Ok(HMeasureResults {
            h,
            convex_hull,
            roc_curve,
            int_components,
            class0_score,
            class1_score,
        })
    }

    fn merge_scores(c0: &[f64], c1: &[f64]) -> (Vec<f64>, Vec<u8>) {
        let mut c0_i = 0;
        let mut c1_i = 0;
        let mut cm_i = 0;
        let c0_num = c0.len();
        let c1_num = c1.len();
        let vec_size = c0_num + c1_num;
        let mut c_scores = vec![0.0; vec_size];
        let mut c_classes = vec![0; vec_size];

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

    fn build_roc(
        c_scores: &[f64],
        c_classes: &[u8],
        c0_num: usize,
        c1_num: usize,
    ) -> Vec<Vec<f64>> {
        let num_rows = c_scores.len();
        let mut roc_points: Vec<Vec<f64>> = vec![vec![0.0, 0.0]];
        // Cumulative counts keep each coordinate an exactly rounded fraction, so collinear ROC points
        // stay collinear and the curve ends at exactly (1, 1).
        let mut count_0 = 0;
        let mut count_1 = 0;
        let mut i = 0;
        while i < num_rows {
            let (duplicate_0, duplicate_1) = Self::duplicate_classes(c_scores, c_classes, i);
            count_0 += duplicate_0;
            count_1 += duplicate_1;
            roc_points.push(vec![
                count_1 as f64 / c1_num as f64,
                count_0 as f64 / c0_num as f64,
            ]);
            i += duplicate_0 + duplicate_1;
        }
        roc_points
    }

    fn duplicate_classes(scores: &[f64], classes: &[u8], idx: usize) -> (usize, usize) {
        let mut dup_class_0 = vec![];
        let mut dup_class_1 = vec![];
        let score_len = scores.len();
        // scores and classes are sorted already here, so only iterate over the range from idx upwards
        // and break once the scores are greater than scores[idx]
        for j in idx..score_len {
            if scores[j] == scores[idx] {
                if classes[j] == 0 {
                    dup_class_0.push(classes[j]);
                } else {
                    dup_class_1.push(classes[j]);
                }
            } else if scores[j] > scores[idx] {
                break;
            }
        }

        (dup_class_0.len(), dup_class_1.len())
    }

    fn build_chull(roc_c: &[Vec<f64>], priors: (f64, f64)) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // The ROC points are already ordered along the curve, so the upper convex hull follows from
        // a single monotone-chain pass: drop the last hull point while it lies on or below the line
        // from its predecessor to the incoming point.
        let mut hull: Vec<[f64; 2]> = vec![];
        for point in roc_c {
            let p = [point[0], point[1]];
            while hull.len() >= 2 {
                let o = hull[hull.len() - 2];
                let q = hull[hull.len() - 1];
                let cross = (q[0] - o[0]) * (p[1] - o[1]) - (q[1] - o[1]) * (p[0] - o[0]);
                if cross >= 0.0 {
                    hull.pop();
                } else {
                    break;
                }
            }
            hull.push(p);
        }

        let chull: Vec<Vec<f64>> = hull[1..].iter().map(|p| p.to_vec()).collect();
        let mut int_components: Vec<Vec<f64>> = vec![];
        let mut cval_prior = 0.0;
        for edge in hull.windows(2) {
            let mut cval = Self::cval(&edge[0], &edge[1], priors);
            // Rounding on almost collinear edges can leave a threshold an ulp below its predecessor.
            if cval < cval_prior {
                cval = cval_prior;
            }
            int_components.push(vec![cval, cval_prior, edge[0][0], edge[0][1]]);
            cval_prior = cval;
        }
        let last = hull[hull.len() - 1];
        int_components.push(vec![1.0, cval_prior, last[0], last[1]]);

        (chull, int_components)
    }

    fn cval(current_point: &[f64; 2], i_point: &[f64; 2], priors: (f64, f64)) -> f64 {
        let (c0_prior, c1_prior) = priors;
        let numerator = c1_prior * (i_point[0] - current_point[0]);
        let denominator =
            c0_prior * (i_point[1] - current_point[1]) + c1_prior * (i_point[0] - current_point[0]);
        numerator / denominator
    }

    fn build_h(&self, int_components: &[Vec<f64>], priors: (f64, f64)) -> f64 {
        let l = self.build_l(int_components, priors);
        let l_max = self.l_max(priors);
        1.0 - l / l_max
    }

    fn build_l(&self, int_components: &[Vec<f64>], priors: (f64, f64)) -> f64 {
        let mut l = 0.0;
        let mut i = 0;
        while i < int_components.len() {
            let int_vals = &int_components[i];
            l += self.int_l_step(int_vals, priors);
            i += 1
        }
        l
    }

    fn int_l_step(&self, int_vals: &[f64], priors: (f64, f64)) -> f64 {
        let r0i = int_vals[3];
        let r1i = int_vals[2];
        let ci = int_vals[1];
        let cip1 = int_vals[0];
        let (c0_prior, c1_prior) = priors;

        c0_prior * (1.0 - r0i) * self.cost_distribution.uc_integral(ci, cip1)
            + c1_prior * r1i * self.cost_distribution.u1mc_integral(ci, cip1)
    }

    fn l_max(&self, priors: (f64, f64)) -> f64 {
        let (c0_prior, c1_prior) = priors;
        c0_prior * self.cost_distribution.uc_integral(0.0, c1_prior)
            + c1_prior * self.cost_distribution.u1mc_integral(c1_prior, 1.0)
    }
}

#[cfg(all(test, feature = "datagen"))]
mod tests {
    use super::{BetaParams, CostRatioDensity, HMeasure};
    use crate::datagen::{BinaryClassParams, BinaryClassifierScores};
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
        let class0_params = BetaParams {
            alpha: c0_a,
            beta: c0_b,
        };
        let class1_params = BetaParams {
            alpha: c1_a,
            beta: c1_b,
        };
        let bcp = &BinaryClassParams {
            class0: class0_params,
            class1: class1_params,
        };
        let bcs = BinaryClassifierScores::new(bcp, c0_n, c1_n, &mut rng).unwrap();

        // specify a cost ratio density
        let cost_density_params = BetaParams {
            alpha: 2.0,
            beta: 2.0,
        };
        let crd = CostRatioDensity::new(cost_density_params).unwrap();

        // calculate the H-Measure given the cost ratio density and scores
        let mut hm = HMeasure::new(crd, None, None).unwrap();
        let hmr = hm.h_measure(&bcs.scores).unwrap();
        // Note : the H-Measure will not in general be identically zero in this test, despite the
        // distributions of the class0 and class1 scores being identical. This is a natural property of
        // a random sample-based measure (the sampled values from the two distributions will in general be
        // different - although for large sample size, the effect should be small).
        assert!(hmr.h < 0.001);
    }
}
