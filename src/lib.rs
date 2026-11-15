/*!
**H-Measure**

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

The cost distribution defaults to a Beta distribution (CostRatioDensity), for which the H-measure
integrals are evaluated in closed form. Any other cost distribution can be supplied by implementing
the CostDistribution trait, or simply as a closure giving its density, in which case the integrals
are evaluated by numerical quadrature.

Invalid inputs (Beta parameters, class priors, empty classes or NaN scores) are reported as an HMeasureError
rather than a panic.

The generators of reproducible dummy classifier scores used by the tests, example and benchmarks
(BinaryClassParams and BinaryClassifierScores) are behind the `datagen` feature, which is off by default:

```toml
hmeasure = { version = "0.1", features = ["datagen"] }
```

A rust pyo3-based wrapper to expose the hmeasure crate in python is provided at:
<https://github.com/robinwiseman/pyhmeasure>

Benchmark comparisons of the Rust hmeasure implementation with
an equivalent python implementation are also provided.

A discussion with further examples is provided in Chapter 2 of:
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>
*/
#[cfg(feature = "datagen")]
mod datagen;
mod hm;
#[cfg(feature = "datagen")]
pub use datagen::{BinaryClassParams, BinaryClassifierScores};
pub use hm::{
    BetaParams, BinaryClassScores, CostDistribution, CostRatioDensity, HMeasure, HMeasureError,
    HMeasureResults,
};
