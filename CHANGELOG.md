# Changelog

## 0.1.0 (2026-11-15)

### Breaking changes

- `CostRatioDensity::new`, `HMeasure::new`, `HMeasure::h_measure` and `BinaryClassifierScores::new`
  return a `Result` with the new `HMeasureError`, instead of panicking or returning `NaN`, for invalid
  Beta parameters, invalid priors, an empty class or a NaN score.
- `HMeasureResults::save` returns an `io::Result` instead of panicking, and overwrites existing files
  instead of appending to them.
- `HMeasure::h_measure` takes `&BinaryClassScores` and no longer sorts the caller's scores in place.
- Class priors are validated: each must be greater than zero and they must sum to one. If only one
  prior is given, the other is its complement; previously a single prior was ignored.
- `CostRatioDensity` no longer exposes its `statrs` distribution through the `c_density_obj` field.
  Use the new `alpha()` and `beta()` accessors.
- `BinaryClassParams` and `BinaryClassifierScores` are behind the `datagen` feature, which is off by
  default.
- `HMeasure` is generic over its cost distribution, as `HMeasure<C = CostRatioDensity>`.
- `HMeasureResults::convex_hull` and `int_components` hold only the vertices of the convex hull.
  Previously they held a row for every ROC point along its vertical and horizontal runs.

### Added

- The `CostDistribution` trait, so that any cost distribution can be used. A closure giving the density
  is enough; its integrals are then evaluated by numerical quadrature. A distribution can instead
  provide its integrals in closed form, as `CostRatioDensity` does for the Beta distribution.
- `HMeasureError`.
- Documentation for every public item, including the convention that class 0 is the lower-scoring
  class, and usage examples in the README.

### Fixed

- An `HMeasure` reused for a second set of scores kept the class priors inferred from the first set.
  Inferred priors are now computed for each call.

### Performance

- The convex hull of the ROC curve is built in a single pass over the ROC points, where it previously
  rescanned the remaining points at every hull step. On well separated classes this is about 20 times
  faster at 1,600 scores per class and over 200 times faster at 51,200 scores per class.
- The integrals over a Beta cost distribution are evaluated in closed form through the Beta CDF.

### Results

- The H-measure agrees with 0.0.6 to within 1.5e-10 on valid inputs, which is inside the 1e-9
  tolerance of the numerical quadrature that 0.0.6 used.
- ROC curve coordinates are computed from cumulative counts, so the curve ends at exactly `[1, 1]`.

### Dependencies

- `statrs` 0.19 (without its default features), `rand` 0.10, `rand_chacha` 0.10 and `criterion` 0.8.
- `rand` and `rand_chacha` are only required by the `datagen` feature.
- Removed the unused `array2d` and `rand_distr` dependencies.

### Migrating from 0.0.6

```rust
// 0.0.6
let crd = CostRatioDensity::new(beta_params);
let mut hm = HMeasure::new(crd, None, None);
let results = hm.h_measure(&mut scores);

// 0.1.0
let crd = CostRatioDensity::new(beta_params)?;
let mut hm = HMeasure::new(crd, None, None)?;
let results = hm.h_measure(&scores)?;
```
