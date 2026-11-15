use hmeasure::{
    BetaParams, BinaryClassParams, BinaryClassScores, BinaryClassifierScores, CostDistribution,
    CostRatioDensity, HMeasure, HMeasureResults,
};

fn h_measure(class0: &[f64], class1: &[f64], cost_alpha: f64, cost_beta: f64) -> HMeasureResults {
    let crd = CostRatioDensity::new(BetaParams {
        alpha: cost_alpha,
        beta: cost_beta,
    });
    let mut scores = BinaryClassScores {
        class0: class0.to_vec(),
        class1: class1.to_vec(),
    };
    HMeasure::new(crd, None, None).h_measure(&mut scores)
}

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "expected {expected}, got {actual}"
    );
}

fn generated_scores(
    c0: (f64, f64),
    c1: (f64, f64),
    n0: usize,
    n1: usize,
    seed: u64,
) -> BinaryClassScores {
    let mut rng = BinaryClassifierScores::generate_rng(seed);
    let bcp = BinaryClassParams {
        class0: BetaParams {
            alpha: c0.0,
            beta: c0.1,
        },
        class1: BetaParams {
            alpha: c1.0,
            beta: c1.1,
        },
    };
    BinaryClassifierScores::new(&bcp, n0, n1, &mut rng).scores
}

// Deterministic overlapping scores: class0 skewed low, class1 skewed high, unequal class sizes.
fn fixture_scores() -> (Vec<f64>, Vec<f64>) {
    let class0 = (0..200)
        .map(|i| ((i as f64 + 0.5) / 200.0).powi(2))
        .collect();
    let class1 = (0..150)
        .map(|j| ((j as f64 + 0.5) / 150.0).sqrt())
        .collect();
    (class0, class1)
}

#[test]
fn hand_derived_value() {
    // ROC hull is (0,0), (0,0.5), (0.5,1), (1,1) with equal priors, so the cost thresholds are
    // 0, 0.5 and 1. With a Beta(2,2) cost density, L = 5/64 and L_max = 5/32, giving H = 1/2.
    assert_close(h_measure(&[0.1, 0.3], &[0.2, 0.4], 2.0, 2.0).h, 0.5, 1e-9);
}

#[test]
fn tied_scores_across_classes() {
    // The tie at 0.3 is a single diagonal ROC step, giving the same hull as hand_derived_value.
    assert_close(h_measure(&[0.1, 0.3], &[0.3, 0.4], 2.0, 2.0).h, 0.5, 1e-9);
}

#[test]
fn perfect_separation_gives_one() {
    let class0: Vec<f64> = (0..500).map(|i| 0.4 * i as f64 / 500.0).collect();
    let class1: Vec<f64> = (0..500).map(|i| 0.6 + 0.4 * i as f64 / 500.0).collect();
    assert_close(h_measure(&class0, &class1, 2.0, 2.0).h, 1.0, 1e-9);
    assert_close(h_measure(&[0.2], &[0.7], 2.0, 2.0).h, 1.0, 1e-9);
}

#[test]
fn inverted_classifier_gives_zero() {
    let low: Vec<f64> = (0..500).map(|i| 0.4 * i as f64 / 500.0).collect();
    let high: Vec<f64> = (0..500).map(|i| 0.6 + 0.4 * i as f64 / 500.0).collect();
    assert_close(h_measure(&high, &low, 2.0, 2.0).h, 0.0, 1e-9);
}

#[test]
fn matches_independent_reference_values() {
    // Expected values come from a separate implementation of Hand (2009): monotone-chain upper
    // convex hull of the ROC curve, with the cost integrals evaluated in closed form through the
    // Beta CDF rather than by numerical quadrature.
    let (class0, class1) = fixture_scores();
    assert_close(
        h_measure(&class0, &class1, 2.0, 2.0).h,
        0.302038233468985,
        1e-9,
    );
    assert_close(
        h_measure(&class0, &class1, 3.0, 1.5).h,
        0.224238152549805,
        1e-9,
    );
    assert_close(
        h_measure(&class0, &class1, 1.0, 1.0).h,
        0.285532326169132,
        1e-9,
    );
}

#[test]
fn matches_independent_reference_values_with_many_ties() {
    // Rounding the fixture to one decimal place leaves only eleven distinct scores.
    let (class0, class1) = fixture_scores();
    let round = |v: &[f64]| -> Vec<f64> { v.iter().map(|s| (s * 10.0).round() / 10.0).collect() };
    let (class0, class1) = (round(&class0), round(&class1));
    assert_close(
        h_measure(&class0, &class1, 2.0, 2.0).h,
        0.294775660642526,
        1e-9,
    );
    assert_close(
        h_measure(&class0, &class1, 3.0, 1.5).h,
        0.216697515573145,
        1e-9,
    );
}

#[test]
fn perfect_separation_hull_has_only_its_vertices() {
    let class0: Vec<f64> = (0..500).map(|i| 0.4 * i as f64 / 500.0).collect();
    let class1: Vec<f64> = (0..500).map(|i| 0.6 + 0.4 * i as f64 / 500.0).collect();
    let results = h_measure(&class0, &class1, 2.0, 2.0);
    assert_eq!(results.convex_hull, vec![vec![0.0, 1.0], vec![1.0, 1.0]]);
}

#[test]
fn convex_hull_is_concave_and_dominates_roc_curve() {
    let separated = generated_scores((2.0, 6.0), (6.0, 2.0), 2000, 1600, 13);
    let identical = generated_scores((4.0, 4.0), (4.0, 4.0), 500, 700, 7);
    let cases = [
        fixture_scores(),
        (separated.class0, separated.class1),
        (identical.class0, identical.class1),
    ];
    for (class0, class1) in cases {
        let results = h_measure(&class0, &class1, 2.0, 2.0);
        let mut hull = vec![vec![0.0, 0.0]];
        hull.extend(results.convex_hull.iter().cloned());

        // The slope strictly decreases from each hull edge to the next.
        for w in hull.windows(3) {
            let cross = (w[1][0] - w[0][0]) * (w[2][1] - w[0][1])
                - (w[1][1] - w[0][1]) * (w[2][0] - w[0][0]);
            assert!(cross < 0.0, "hull is not strictly concave at {:?}", w[1]);
        }

        // Every ROC point lies on or below the hull.
        for point in &results.roc_curve {
            let edge = hull
                .windows(2)
                .find(|e| e[0][0] <= point[0] && point[0] <= e[1][0])
                .unwrap();
            let dx = edge[1][0] - edge[0][0];
            let hull_y = if dx == 0.0 {
                edge[1][1]
            } else {
                edge[0][1] + (edge[1][1] - edge[0][1]) * (point[0] - edge[0][0]) / dx
            };
            assert!(point[1] <= hull_y + 1e-12, "{point:?} is above the hull");
        }

        // The cost thresholds run from 0 to 1 without decreasing.
        assert_eq!(results.int_components[0][1], 0.0);
        assert_eq!(results.int_components.last().unwrap()[0], 1.0);
        assert!(results.int_components.iter().all(|row| row[1] <= row[0]));
    }
}

#[test]
fn input_order_does_not_matter() {
    let (class0, class1) = fixture_scores();
    let reversed0: Vec<f64> = class0.iter().rev().copied().collect();
    let reversed1: Vec<f64> = class1.iter().rev().copied().collect();
    let expected = h_measure(&class0, &class1, 2.0, 2.0).h;
    assert_close(
        h_measure(&reversed0, &reversed1, 2.0, 2.0).h,
        expected,
        1e-12,
    );
}

#[test]
fn explicit_empirical_priors_match_inferred_priors() {
    let (class0, class1) = fixture_scores();
    let crd = CostRatioDensity::new(BetaParams {
        alpha: 2.0,
        beta: 2.0,
    });
    let mut scores = BinaryClassScores {
        class0: class0.clone(),
        class1: class1.clone(),
    };
    let explicit =
        HMeasure::new(crd, Some(200.0 / 350.0), Some(150.0 / 350.0)).h_measure(&mut scores);
    assert_close(explicit.h, h_measure(&class0, &class1, 2.0, 2.0).h, 1e-12);
}

#[test]
fn results_are_consistent() {
    let (class0, class1) = fixture_scores();
    let crd = CostRatioDensity::new(BetaParams {
        alpha: 2.0,
        beta: 2.0,
    });
    let mut scores = BinaryClassScores {
        class0: class0.clone(),
        class1: class1.clone(),
    };
    let mut hm = HMeasure::new(crd, None, None);
    let results = hm.h_measure(&mut scores);

    assert_eq!(hm.h, Some(results.h));
    assert_eq!(results.class0_score.len(), class0.len());
    assert_eq!(results.class1_score.len(), class1.len());
    assert!(results.class0_score.windows(2).all(|w| w[0] <= w[1]));
    assert!(results.class1_score.windows(2).all(|w| w[0] <= w[1]));

    assert_eq!(results.roc_curve[0], vec![0.0, 0.0]);
    for end in [
        results.roc_curve.last().unwrap(),
        results.convex_hull.last().unwrap(),
    ] {
        assert_close(end[0], 1.0, 1e-9);
        assert_close(end[1], 1.0, 1e-9);
    }
    for points in [&results.roc_curve, &results.convex_hull] {
        assert!(points
            .windows(2)
            .all(|w| w[0][0] <= w[1][0] && w[0][1] <= w[1][1]));
    }
}

#[test]
fn h_decreases_as_class_distributions_converge() {
    let mut previous = f64::INFINITY;
    for step in 0..9 {
        let shift = 0.25 * step as f64;
        let scores = generated_scores(
            (2.0 + shift, 6.0 - shift),
            (6.0 - shift, 2.0 + shift),
            2000,
            1600,
            13,
        );
        let h = h_measure(&scores.class0, &scores.class1, 2.0, 2.0).h;
        assert!(h < previous, "step {step}: {h} is not below {previous}");
        previous = h;
    }
    assert!(previous < 0.01);
}

#[test]
fn generated_scores_are_reproducible() {
    let first = generated_scores((2.0, 6.0), (6.0, 2.0), 300, 200, 13);
    let second = generated_scores((2.0, 6.0), (6.0, 2.0), 300, 200, 13);
    let other_seed = generated_scores((2.0, 6.0), (6.0, 2.0), 300, 200, 14);

    assert_eq!(first.class0.len(), 300);
    assert_eq!(first.class1.len(), 200);
    assert_eq!(first.class0, second.class0);
    assert_eq!(first.class1, second.class1);
    assert_ne!(first.class0, other_seed.class0);
    assert!(first
        .class0
        .iter()
        .chain(&first.class1)
        .all(|s| (0.0..=1.0).contains(s)));
}

#[test]
fn save_overwrites_existing_files() {
    let (class0, class1) = fixture_scores();
    let results = h_measure(&class0, &class1, 2.0, 2.0);
    let dir = std::env::temp_dir().join(format!("hmeasure_save_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dir_str = dir.to_str().unwrap();

    results.save("run", dir_str);
    results.save("run", dir_str);

    let line_count = |name: &str| {
        std::fs::read_to_string(dir.join(name))
            .unwrap()
            .lines()
            .count()
    };
    assert_eq!(line_count("s0_run.csv"), class0.len() + 1);
    assert_eq!(line_count("s1_run.csv"), class1.len() + 1);
    assert_eq!(line_count("rc_run.csv"), results.roc_curve.len() + 1);
    assert_eq!(line_count("ch_run.csv"), results.convex_hull.len() + 1);
    assert_eq!(line_count("ic_run.csv"), results.int_components.len() + 1);

    let scores_csv = std::fs::read_to_string(dir.join("s0_run.csv")).unwrap();
    assert_eq!(scores_csv.lines().next(), Some("idx,score"));

    std::fs::remove_dir_all(&dir).unwrap();
}

fn h_measure_with<C: CostDistribution>(class0: &[f64], class1: &[f64], cost: C) -> f64 {
    let mut scores = BinaryClassScores {
        class0: class0.to_vec(),
        class1: class1.to_vec(),
    };
    HMeasure::new(cost, None, None).h_measure(&mut scores).h
}

fn bell_density(c: f64) -> f64 {
    (-(c - 0.3).powi(2) / 0.045).exp() / 0.375
}

#[test]
fn closure_cost_densities_match_independent_reference_values() {
    // Expected values come from a separate implementation that integrates each density with a
    // composite Simpson rule between the hull's cost thresholds.
    let (class0, class1) = fixture_scores();
    let wave = |c: f64| 1.0 + 0.8 * (4.0 * std::f64::consts::PI * c).cos();
    assert_close(
        h_measure_with(&class0, &class1, bell_density),
        0.388823630496393,
        1e-8,
    );
    assert_close(
        h_measure_with(&class0, &class1, wave),
        0.315544815804618,
        1e-8,
    );
}

#[test]
fn beta_density_by_quadrature_matches_closed_form() {
    let (class0, class1) = fixture_scores();
    let beta_2_2 = |c: f64| 6.0 * c * (1.0 - c);
    let closed_form = h_measure(&class0, &class1, 2.0, 2.0).h;
    assert_close(
        h_measure_with(&class0, &class1, beta_2_2),
        closed_form,
        1e-8,
    );
}

#[test]
fn cost_density_scale_does_not_matter() {
    let (class0, class1) = fixture_scores();
    let scaled = |c: f64| 100.0 * bell_density(c);
    let expected = h_measure_with(&class0, &class1, bell_density);
    assert_close(h_measure_with(&class0, &class1, scaled), expected, 1e-8);
}

struct UniformCost;

impl CostDistribution for UniformCost {
    fn pdf(&self, _cost: f64) -> f64 {
        1.0
    }
    fn uc_integral(&self, lo: f64, hi: f64) -> f64 {
        (hi * hi - lo * lo) / 2.0
    }
    fn u1mc_integral(&self, lo: f64, hi: f64) -> f64 {
        (hi - lo) - (hi * hi - lo * lo) / 2.0
    }
}

#[test]
fn custom_distribution_with_closed_form_integrals() {
    // A uniform cost density is Beta(1, 1), so this must reproduce that reference value.
    let (class0, class1) = fixture_scores();
    assert_close(
        h_measure_with(&class0, &class1, UniformCost),
        0.285532326169132,
        1e-9,
    );
}
