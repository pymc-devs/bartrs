//! Response strategy implementations for computing leaf (terminal) node values.

use numpy::ndarray::{Array, Array1, ArrayView1, Ix2};
use rand::RngCore;
use rand_distr::{Distribution, Normal};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    Gaussian = 0,
    Linear = 1,
    // Monotone, etc.
}

impl LeafKind {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => LeafKind::Linear,
            _ => LeafKind::Gaussian,
        }
    }
}

#[derive(Clone, Debug)]
pub enum LeafPayload {
    Gaussian { value: Vec<f64> },
    Linear {intercept: Vec<f64>, slope: Vec<f64>, var: u32 },

    // TODO: Monotone. And in the future GP, etc.
}

#[derive(Clone, Debug)]
pub struct LeafProposal {
    pub node_idx: usize,
    pub split_var: u32,
    pub split_val: f64,
    pub left: LeafPayload,
    pub right: LeafPayload,
}

/// Response method interface for computing leaf values.
pub trait ResponseStrategy {
    fn sample_leaf_proposal(
        &self,
        rng: &mut dyn RngCore,
        node_samples: &[u32],
        col: &ArrayView1<f64>,
        sum_trees: &Array<f64, Ix2>,
        leaf_sd: &Array1<f64>,
        n_trees: usize,
        n_outputs: usize,
        split_val: f64,
        split_var: u32,
        node_idx: usize,
    ) -> LeafProposal;
}

/// Gaussian response strategy.
///
/// Samples from a normal distribution centered at the empirical mean
/// of the residuals in the leaf, with fixed variance.
#[derive(Clone, Copy, Debug)]
pub struct GaussianResponseStrategy;

impl ResponseStrategy for GaussianResponseStrategy {
    fn sample_leaf_proposal(
        &self,
        rng: &mut dyn RngCore,
        node_samples: &[u32],
        col: &ArrayView1<f64>,
        sum_trees: &Array<f64, Ix2>,
        leaf_sd: &Array1<f64>,
        n_trees: usize,
        n_outputs: usize,
        split_val: f64,
        split_var: u32,
        node_idx: usize,
    ) -> LeafProposal {

        // Initialize accumulators per output
        let mut left_sum = vec![0.0f64; n_outputs];
        let mut left_n = vec![0usize; n_outputs];
        let mut right_sum = vec![0.0f64; n_outputs];
        let mut right_n = vec![0usize; n_outputs];

        for &s in node_samples.iter() {
            let idx = s as usize;
            let v = unsafe { *col.uget(idx) };
            for o in 0..n_outputs {
                let p = unsafe { *sum_trees.uget([o, idx]) };
                if v <= split_val {
                    left_sum[o] += p;
                    left_n[o] += 1;
                } else {
                    right_sum[o] += p;
                    right_n[o] += 1;
                } 
            }
        }

        let dist = Normal::new(0.0, 1.0).unwrap();

        let mut left_value = vec![0.0f64; n_outputs];
        let mut right_value = vec![0.0f64; n_outputs];

        for o in 0..n_outputs {
            let noise = dist.sample(rng) * leaf_sd[o]; // * config.sigma;
            left_value[o] = if left_n[o] == 0 {
                noise
            } else {
                left_sum[o] / left_n[o] as f64 / n_trees as f64 + noise
            };


            let noise_r = dist.sample(rng) * leaf_sd[o]; // *config.sigma;
            right_value[o] = if right_n[o] == 0 {
                noise_r
            } else {
                right_sum[o] / right_n[o] as f64 / n_trees as f64 + noise_r
            };
        }

        LeafProposal {
            node_idx,
            split_var,
            split_val,
            left: LeafPayload::Gaussian { value: left_value },
            right: LeafPayload::Gaussian { value: right_value },
        }
    }
}

/// MOTR-BART response strategy (placeholder).
#[derive(Clone, Copy, Debug)]
pub struct MotrStrategy;

impl ResponseStrategy for MotrStrategy {
    fn sample_leaf_proposal(
        &self,
        _rng: &mut dyn RngCore,
        _node_samples: &[u32],
        _col: &ArrayView1<f64>,
        _sum_trees: &Array<f64, Ix2>,
        _leaf_sd: &Array1<f64>,
        _n_trees: usize,
        _n_outputs: usize,
        _split_val: f64,
        _split_var: u32,
        _node_idx: usize,
    ) -> LeafProposal {
        todo!("MotrStrategy not yet implemented")
    }
}


#[derive(Clone, Copy, Debug)]
pub struct LinearStrategy;

impl LinearStrategy {
    fn fit_linear_1d(x: &[f64], y: &[f64], noise: f64, n_trees: usize) -> Option<(f64, f64)> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }

        if y.len() < 3 {
            let a = y.iter().sum::<f64>() / y.len() as f64 / n_trees as f64 + noise;
            return Some((a, 0.0));
        }

        let y_scaled: Vec<f64> = y.iter().map(|&v| v / n_trees as f64 + noise).collect();

        let n = x.len() as f64;
        let xbar = x.iter().sum::<f64>() / n;
        let ybar = y_scaled.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..x.len() {
            let xd = x[i] - xbar;
            num += xd * (y_scaled[i] - ybar);
            den += xd * xd;
        }

        let b = if den == 0.0 { return Some((ybar / n_trees as f64 + noise, 0.0)) } else { num / den };
        let a = ybar - b * xbar;
        Some((a, b))
    }
}

impl ResponseStrategy for LinearStrategy {
    fn sample_leaf_proposal(
        &self,
        rng: &mut dyn RngCore,
        node_samples: &[u32],
        col: &ArrayView1<f64>,
        sum_trees: &Array<f64, Ix2>,
        leaf_sd: &Array1<f64>,
        n_trees: usize,
        n_outputs: usize,
        split_val: f64,
        split_var: u32,
        node_idx: usize,
    ) -> LeafProposal {
        let mut left_idx: Vec<usize> = Vec::new();
        let mut right_idx: Vec<usize> = Vec::new();

        for &s in node_samples {
            let idx = s as usize;
            let v = unsafe { *col.uget(idx) };
            if v <= split_val {
                left_idx.push(idx);
            } else {
                right_idx.push(idx);
            }
        }

        let x_left: Vec<f64> = left_idx.iter().map(|&i| unsafe { *col.uget(i) }).collect();
        let x_right: Vec<f64> = right_idx.iter().map(|&i| unsafe { *col.uget(i) }).collect();

        let mut left_intercepts = Vec::with_capacity(n_outputs);
        let mut left_slopes = Vec::with_capacity(n_outputs);
        let mut right_intercepts = Vec::with_capacity(n_outputs);
        let mut right_slopes = Vec::with_capacity(n_outputs);

        let normal = Normal::new(0.0, 1.0).unwrap();

        for o in 0..n_outputs {
            let noise_l = normal.sample(rng) * leaf_sd[o];
            let noise_r = normal.sample(rng) * leaf_sd[o];

            let y_left: Vec<f64> = left_idx
                .iter()
                .map(|&i| unsafe { *sum_trees.uget([o, i]) })
                .collect();
            let y_right: Vec<f64> = right_idx
                .iter()
                .map(|&i| unsafe { *sum_trees.uget([o, i]) })
                .collect();

            // empty leaf -> zero
            // 1 point or <3 points, constant mean
            // otherwise linear fit
            let (a_l, b_l) = if y_left.is_empty() {
                (0.0, 0.0)
            } else if y_left.len() < 3 {
                Self::fit_linear_1d(&x_left, &y_left, noise_l, n_trees).unwrap_or_else(|| (y_left.iter().sum::<f64>() / y_left.len() as f64 / n_trees as f64 + noise_l, 0.0))
            } else {
                let (a, b) = Self::fit_linear_1d(&x_left, &y_left, noise_l, n_trees)
                    .unwrap_or_else(|| (y_left.iter().sum::<f64>() / y_left.len() as f64 / n_trees as f64 + noise_l, 0.0));
                (a, b)
            };

            let (a_r, b_r) = if y_right.is_empty() {
                (0.0, 0.0)
            } else if y_right.len() < 3 {
                Self::fit_linear_1d(&x_right, &y_right, noise_r, n_trees).unwrap_or_else(|| (y_right.iter().sum::<f64>() / y_right.len() as f64 / n_trees as f64 + noise_r, 0.0))
            } else {
                let (a, b) = Self::fit_linear_1d(&x_right, &y_right, noise_r, n_trees)
                    .unwrap_or_else(|| (y_right.iter().sum::<f64>() / y_right.len() as f64 / n_trees as f64 + noise_r, 0.0));
                (a, b)
            };

            left_intercepts.push(a_l);
            left_slopes.push(b_l);
            right_intercepts.push(a_r);
            right_slopes.push(b_r);
        }

        LeafProposal {
            node_idx,
            split_var,
            split_val,
            left: LeafPayload::Linear {
                intercept: left_intercepts,
                slope: left_slopes,
                var: split_var,
            },
            right: LeafPayload::Linear {
                intercept: right_intercepts,
                slope: right_slopes,
                var: split_var,
            },
        }
    }
}

/// Enum for dynamic dispatch over response strategies.
#[derive(Clone, Debug)]
pub enum ResponseStrategies {
    Motr(MotrStrategy),
    Gaussian(GaussianResponseStrategy),
    Linear(LinearStrategy),
}

impl ResponseStrategies {
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name.to_lowercase().as_str() {
            "gaussian" => Ok(ResponseStrategies::Gaussian(GaussianResponseStrategy)),
            "linear" => Ok(ResponseStrategies::Linear(LinearStrategy)),
            "motr" => Ok(ResponseStrategies::Motr(MotrStrategy)),
            _ => Err(format!(
                "Unknown response strategy: '{}'. Supported: 'gaussian', 'motr', 'linear'.",
                name
            )),
        }
    }
}

impl ResponseStrategy for ResponseStrategies {
    fn sample_leaf_proposal(
        &self,
        rng: &mut dyn RngCore,
        node_samples: &[u32],
        col: &ArrayView1<f64>,
        sum_trees: &Array<f64, Ix2>,
        leaf_sd: &Array1<f64>,
        n_trees: usize,
        n_outputs: usize,
        split_val: f64,
        split_var: u32,
        node_idx: usize,
    ) -> LeafProposal {
        match self {
            ResponseStrategies::Motr(s) => s.sample_leaf_proposal(
                rng,
                node_samples,
                col,
                sum_trees,
                leaf_sd,
                n_trees,
                n_outputs,
                split_val,
                split_var,
                node_idx,
            ),
            ResponseStrategies::Gaussian(s) => s.sample_leaf_proposal(
                rng,
                node_samples,
                col,
                sum_trees,
                leaf_sd,
                n_trees,
                n_outputs,
                split_val,
                split_var,
                node_idx,
            ),
            ResponseStrategies::Linear(s) => s.sample_leaf_proposal(
                rng,
                node_samples,
                col,
                sum_trees,
                leaf_sd,
                n_trees,
                n_outputs,
                split_val,
                split_var,
                node_idx,
            ),

        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_linear_1d_intercept() {
        let x = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let y = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let noise = 0.0f64;
        let n_trees = 1;

        let Some((intercept, slope)) = LinearStrategy::fit_linear_1d(x, y, noise, n_trees) else { 
            panic!("Got None when was expecting intercept and slope");
        };

        let a = intercept;
        let _b = slope;
        assert!((a - 0.0).abs() < 1e-6, "Expected intercept ~0.0, got {}", a);
    }

    #[test]
    fn test_fit_linear_1d_slope() {
        let x = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let y = &[2.0, 4.0, 6.0, 8.0, 10.0];

        let noise = 0.0f64;
        let n_trees = 1;

        let Some((intercept, slope)) = LinearStrategy::fit_linear_1d(x, y, noise, n_trees) else {
            panic!("Got None when was expecting intercept and slope");
        };

        let _a = intercept;
        let b = slope;
        assert!((b - 2.0).abs() < 1e-6, "Expected intercept ~2.0, got {}", b);

    }
    
}