//! Resampling strategies for particle filtering in BART.
use rand::Rng;

/// Resampling interface for implementing different resampling methods.
pub trait ResamplingStrategy {
    /// Resample into a pre-allocated buffer, avoiding allocation.
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>);

    /// Convenience wrapper that allocates a new Vec.
    fn resample(&self, rng: &mut impl Rng, weights: &[f64]) -> Vec<usize> {
        let mut out = Vec::with_capacity(weights.len());
        self.resample_into(rng, weights, &mut out);
        out
    }
}

/// Systematic resampling strategy.
///
/// Builds a CDF and samples a single random offset to determine which bin
/// each particle lands in. Provides lower variance than multinomial resampling.
#[derive(Clone, Copy, Debug)]
pub struct SystematicResampling;

impl ResamplingStrategy for SystematicResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        let u = rng.random::<f64>();

        let mut current_idx = 0usize;
        let mut current_cum = 0.0f64;
        out.clear();

        for i in 0..n {
            let target = (i as f64 + u) / n as f64;

            while current_cum < target && current_idx < n {
                current_cum += weights[current_idx];
                current_idx += 1;
            }

            let ancestor = if current_idx == 0 { 0 } else { current_idx - 1 };
            out.push(ancestor);
        }
    }
}

/// Multinomial resampling strategy.
///
/// Standard multinomial resampling where each particle is independently
/// sampled according to its weight.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialResampling;

impl ResamplingStrategy for MultinomialResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        out.clear();

        // Build CDF (reuses allocation if called repeatedly via resample() wrapper,
        // but this internal vec is unavoidable without changing the trait further)
        let mut cdf = Vec::with_capacity(n);
        let mut cumsum = 0.0;
        for &w in weights {
            cumsum += w;
            cdf.push(cumsum);
        }

        for _ in 0..n {
            let u: f64 = rng.random();
            let idx = cdf.partition_point(|&c| c < u).min(n - 1);
            out.push(idx);
        }
    }
}

/// Stratified resampling strategy.
///
/// Uses independent random numbers for each stratum, providing a middle
/// ground between systematic and multinomial.
#[derive(Clone, Copy, Debug)]
pub struct StratifiedResampling;

impl ResamplingStrategy for StratifiedResampling {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        let n = weights.len();
        let mut current_idx = 0usize;
        let mut current_cum = 0.0f64;
        out.clear();

        for i in 0..n {
            let u: f64 = rng.random();
            let target = (i as f64 + u) / n as f64;

            while current_cum < target && current_idx < n {
                current_cum += weights[current_idx];
                current_idx += 1;
            }

            let ancestor = if current_idx == 0 { 0 } else { current_idx - 1 };
            out.push(ancestor);
        }
    }
}

/// Enum for dynamic dispatch over resampling strategies.
#[derive(Clone, Copy, Debug)]
pub enum ResamplingStrategies {
    Systematic(SystematicResampling),
    Multinomial(MultinomialResampling),
    Stratified(StratifiedResampling),
}

impl ResamplingStrategies {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "systematic" => Ok(ResamplingStrategies::Systematic(SystematicResampling)),
            "multinomial" => Ok(ResamplingStrategies::Multinomial(MultinomialResampling)),
            "stratified" => Ok(ResamplingStrategies::Stratified(StratifiedResampling)),
            _ => Err(format!("Unknown resampling strategy: '{}'", name)),
        }
    }
}

impl ResamplingStrategy for ResamplingStrategies {
    fn resample_into(&self, rng: &mut impl Rng, weights: &[f64], out: &mut Vec<usize>) {
        match self {
            ResamplingStrategies::Systematic(s) => s.resample_into(rng, weights, out),
            ResamplingStrategies::Multinomial(s) => s.resample_into(rng, weights, out),
            ResamplingStrategies::Stratified(s) => s.resample_into(rng, weights, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const WEIGHTS: [f64; 3] = [0.1, 0.3, 0.6];

    fn assert_shape_and_bounds(strategy: ResamplingStrategies, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::new();

        strategy.resample_into(&mut rng, &WEIGHTS, &mut out);

        assert_eq!(out.len(), WEIGHTS.len());
        assert!(out.iter().all(|&index| index < WEIGHTS.len()));
    }

    fn assert_degenerate_weights(strategy: ResamplingStrategies, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = [1.0, 0.0, 0.0];

        assert_eq!(
            strategy.resample(&mut rng, &weights),
            vec![0; weights.len()]
        );
    }

    fn assert_seeded_frequencies(strategy: ResamplingStrategies, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(WEIGHTS.len());
        let mut counts = [0usize; WEIGHTS.len()];
        let repetitions = 20_000;

        for _ in 0..repetitions {
            strategy.resample_into(&mut rng, &WEIGHTS, &mut out);
            for &index in &out {
                counts[index] += 1;
            }
        }

        let sample_count = (repetitions * WEIGHTS.len()) as f64;
        for (index, (&count, &expected)) in counts.iter().zip(&WEIGHTS).enumerate() {
            let observed = count as f64 / sample_count;
            assert!(
                (observed - expected).abs() < 0.01,
                "index {index}: expected frequency {expected}, observed {observed}",
            );
        }
    }

    fn assert_reuses_output_buffer(strategy: ResamplingStrategies, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(8);
        out.extend([usize::MAX; 5]);
        let pointer = out.as_ptr();
        let capacity = out.capacity();

        strategy.resample_into(&mut rng, &WEIGHTS, &mut out);

        assert_eq!(out.as_ptr(), pointer);
        assert_eq!(out.capacity(), capacity);
        assert_eq!(out.len(), WEIGHTS.len());
    }

    #[test]
    fn test_systematic_resample() {
        let mut weights = &[0.0, 0.25, 0.75];
        let mut rng = rand::rng();
        let mut out = Vec::new();

        ResamplingStrategies::Systematic(SystematicResampling)
            .resample_into(&mut rng, weights, &mut out);
        assert!(out.iter().all(|&index| index >= 1 && index < weights.len()));

        weights = &[0.5, 0.3, 0.2];
        out = Vec::new();
        ResamplingStrategies::Systematic(SystematicResampling)
            .resample_into(&mut rng, weights, &mut out);
        assert!(out.iter().all(|&index| index < weights.len()));
    }

    #[test]
    fn test_multinomial_shape_and_bounds() {
        assert_shape_and_bounds(ResamplingStrategies::Multinomial(MultinomialResampling), 17);
    }

    #[test]
    fn test_stratified_shape_and_bounds() {
        assert_shape_and_bounds(ResamplingStrategies::Stratified(StratifiedResampling), 23);
    }

    #[test]
    fn test_multinomial_degenerate_weights() {
        assert_degenerate_weights(ResamplingStrategies::Multinomial(MultinomialResampling), 29);
    }

    #[test]
    fn test_stratified_degenerate_weights() {
        assert_degenerate_weights(ResamplingStrategies::Stratified(StratifiedResampling), 31);
    }

    #[test]
    fn test_multinomial_reuses_output_buffer() {
        assert_reuses_output_buffer(ResamplingStrategies::Multinomial(MultinomialResampling), 37);
    }

    #[test]
    fn test_stratified_reuses_output_buffer() {
        assert_reuses_output_buffer(ResamplingStrategies::Stratified(StratifiedResampling), 39);
    }

    #[test]
    fn test_multinomial_seeded_frequencies() {
        assert_seeded_frequencies(ResamplingStrategies::Multinomial(MultinomialResampling), 41);
    }

    #[test]
    fn test_stratified_seeded_frequencies() {
        assert_seeded_frequencies(ResamplingStrategies::Stratified(StratifiedResampling), 43);
    }
}
