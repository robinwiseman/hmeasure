# H-Measure

The hmeasure crate provides a Rust implementation of the H-measure. The **H-measure is a coherent alternative**
to the widely used AUC measure (Area Under the ROC Curve) for assessing the relative quality
of binary classifiers. Whereas the AUC implicitly assumes a different cost distribution for the
"cost of being wrong" depending on the classifier it is applied to, the H-Measure, by contrast, enables
the researcher to fix the cost distribution consistently across all classifiers studied. The "cost of
being wrong" is an inherent property of the subject being modelled and should not depend on the
specific model being used to model the subject. H-Measure enables that consistency, whereas AUC does not.

The H-measure was introduced by David J. Hand in the paper:

"Measuring classifier performance: a coherent alternative to the area under the ROC curve"
Mach Learn (2009) 77: 103–123
<https://link.springer.com/article/10.1007/s10994-009-5119-5>

## Usage

```rust
use hmeasure::{BetaParams, BinaryClassScores, CostRatioDensity, HMeasure, HMeasureError};

fn main() -> Result<(), HMeasureError> {
    // Scores produced by a classifier, grouped by the true class of each observation.
    // Class 0 is the class the classifier is expected to score lower.
    let scores = BinaryClassScores {
        class0: vec![0.10, 0.20, 0.30, 0.45, 0.50, 0.75],
        class1: vec![0.40, 0.60, 0.70, 0.80, 0.90],
    };

    // Beta(2, 2) is the cost distribution proposed in the paper for standardised comparisons.
    let cost_density = CostRatioDensity::new(BetaParams { alpha: 2.0, beta: 2.0 })?;

    // With no priors given, they are inferred from the class sizes.
    let mut hm = HMeasure::new(cost_density, None, None)?;
    let results = hm.h_measure(&scores)?;

    println!("H-measure: {:.4}", results.h); // H-measure: 0.5212
    Ok(())
}
```

The H-measure runs from 0, for a classifier that separates the classes no better than chance, to 1, for
a classifier that separates them perfectly. The results also hold the ROC curve and its convex hull, and
can be written to csv files with `save`.

## Scores, classes and priors

- Class 0 is the class expected to receive the lower scores, and class 1 the class expected to receive
  the higher scores. If the classes are supplied the other way round, the classifier appears to rank them
  worse than chance and the H-measure is at or near 0.
- Only the ordering of the scores matters, so they do not have to be probabilities.
- The class priors are inferred from the class sizes unless they are passed to `HMeasure::new`. If one
  prior is given, the other is its complement. Priors must be greater than zero and sum to one.

## Cost distributions

The cost ratio is the cost of misclassifying a class 0 observation, as a share of the combined cost of
misclassifying a class 0 observation and a class 1 observation. The H-measure weights the classifier's
loss at each cost ratio by a distribution over the cost ratio.

The cost distribution defaults to a Beta distribution (`CostRatioDensity`), for which the H-measure
integrals are evaluated in closed form. Any other cost distribution can be supplied by implementing
the `CostDistribution` trait, or simply as a closure giving its density, in which case the integrals
are evaluated by numerical quadrature:

```rust
use hmeasure::{BinaryClassScores, HMeasure, HMeasureError};

fn main() -> Result<(), HMeasureError> {
    let scores = BinaryClassScores {
        class0: vec![0.10, 0.20, 0.30, 0.45, 0.50, 0.75],
        class1: vec![0.40, 0.60, 0.70, 0.80, 0.90],
    };

    // A bell-shaped cost density centred on a cost ratio of 0.3.
    let cost_density = |c: f64| (-(c - 0.3).powi(2) / 0.045).exp() / 0.375;
    let results = HMeasure::new(cost_density, None, None)?.h_measure(&scores)?;

    println!("H-measure: {:.4}", results.h); // H-measure: 0.5418
    Ok(())
}
```

Quadrature assumes a density that is smooth and of order one. A distribution with jumps, or one whose
integrals are known exactly, should implement `CostDistribution` and provide its integrals directly.

## Errors

Invalid inputs (Beta parameters, class priors, empty classes or NaN scores) are reported as an
`HMeasureError` rather than a panic.

## Generating test data

The generators of reproducible dummy classifier scores used by the tests, example and benchmarks
(`BinaryClassParams` and `BinaryClassifierScores`) are behind the `datagen` feature, which is off by default:

```toml
hmeasure = { version = "0.1", features = ["datagen"] }
```

The `monotonic` example uses them to show the H-measure falling as two score distributions converge:

```text
cargo run --example monotonic --features datagen
```

## Python

A rust pyo3-based wrapper to expose the hmeasure crate in python is provided at:
<https://github.com/robinwiseman/pyhmeasure>

Benchmark comparisons of the Rust hmeasure implementation with
an equivalent python implementation are also provided.

## Further reading

A discussion with further examples is provided in Chapter 2 of:
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>

Changes between versions are listed in the changelog:
<https://github.com/robinwiseman/hmeasure/blob/master/CHANGELOG.md>
