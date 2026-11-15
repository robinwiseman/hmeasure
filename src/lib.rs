#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

#[cfg(feature = "datagen")]
mod datagen;
mod hm;
#[cfg(feature = "datagen")]
pub use datagen::{BinaryClassParams, BinaryClassifierScores};
pub use hm::{
    BetaParams, BinaryClassScores, CostDistribution, CostRatioDensity, HMeasure, HMeasureError,
    HMeasureResults,
};
