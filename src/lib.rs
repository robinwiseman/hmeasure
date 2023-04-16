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

A rust pyo3-based wrapper to expose the hmeasure crate in python is provided at:
<https://github.com/robinwiseman/pyhmeasure>

Benchmark comparisons of the Rust hmeasure implementation with
an equivalent python implementation are also provided.

A discussion with further examples is provided in Chapter 2 of:
<https://github.com/robinwiseman/finML/blob/aa12845f01454c24f36f4df0d1cb6e0993ea7c7f/src/finML_2022.pdf>
*/
mod hm;
mod datagen;
pub use hm::{CostRatioDensity, HMeasureResults, HMeasure};
pub use datagen::{BetaParams, BinaryClassParams, BinaryClassScores, BinaryClassifierScores};