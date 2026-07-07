use numpy::ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand::rngs::SmallRng;

use crate::config::BartConfig;
use crate::data::OwnedData;
use crate::resampling::ResamplingStrategy;
use crate::smc::smc_step;
use crate::splitting::SplitRules;
use crate::state::{BartInfo, BartState};
use crate::tree::{TreeArrays, LEAF_SENTINEL};
use crate::weight::WeightFn;
use crate::response::ResponseStrategies;


/// Welford's online algorithm for compute variance
#[derive(Clone)]
pub struct RunningSd {
    /// Number of data points used in computing std. Might be different from number of rows in X data.
    pub count: usize,
    /// The shape of the leaves
    pub shape: usize,
    /// Running mean
    pub mean: Array2<f64>,
    /// Running second moment
    pub m2: Array2<f64>,
}

impl RunningSd {
    pub fn new(shape: usize, num_samples: usize) -> Self {
        Self { count: 0, shape: shape, mean: Array2::zeros((shape, num_samples)), m2: Array2::zeros((shape, num_samples)) }
    }

    pub fn update(&mut self, new_value: &Array2<f64>) -> Array1<f64> {
        self.count += 1;
        let delta = new_value - &self.mean;
        self.mean += &(delta.clone() / self.count as f64);
        let delta2 = new_value - &self.mean;
        self.m2 += &(delta * delta2);

        let std = self.m2.mapv(|v| (v / self.count as f64).sqrt());
        std.mean_axis(Axis(1)).unwrap()
    }
}


/// BlackJAX-style sampling algorithm trait.
pub trait SamplingAlgorithm {
    type State;
    type Info;

    fn init(&self, rng: &mut impl Rng) -> Self::State;
    fn step(&self, rng: &mut impl Rng, state: Self::State) -> (Self::State, Self::Info);
}

/// Concrete BART kernel parameterized by strategy types.
pub struct BartKernel<R, W> {
    pub split_rules: Vec<SplitRules>,
    pub resampling: R,
    pub weight_fn: W,
    pub config: BartConfig,
    pub data: OwnedData,
}

impl<R, W> SamplingAlgorithm for BartKernel<R, W>
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    type State = BartState;
    type Info = BartInfo;

    fn init(&self, _rng: &mut impl Rng) -> BartState {
        let n_samples = self.data.n_samples();
        let y_mean = self.data.y.mean().unwrap_or(0.0);
        let init_leaf_value = y_mean / self.config.n_trees as f64;
        let data_view = self.data.view();
        let n_trees = self.config.n_trees;
        let n_outputs = self.config.n_outputs;
        
        let forest: Vec<TreeArrays> = (0..self.config.n_trees)
            .map(|_| TreeArrays::new(init_leaf_value, n_samples, self.config.max_depth, n_outputs))
            .collect();

        let predictions = Array2::from_elem((n_outputs, n_samples), y_mean);
        let variable_inclusion = vec![0u32; self.data.n_features()];

        let mut leaf_sd = Array1::ones(n_outputs);
        let running_sd = RunningSd::new(n_outputs, n_samples);

        let is_binary = data_view.y.iter().copied().all(|v| v == 0.0 || v == 1.0);

        if is_binary {
            leaf_sd = Array1::from_elem(n_outputs, 3.0 / (n_trees as f64).sqrt());
        } else {
            let scale = (n_trees as f64).sqrt();
            let y_std = self.data.y_std_as_scalar();
            leaf_sd = leaf_sd.mapv(|_| y_std / scale);
        }

        BartState {
            forest,
            predictions,
            variable_inclusion,
            next_tree_idx: 0,
            tune: true,
            running_sd: running_sd,
            leaf_sd: leaf_sd,
            iter: 0,
            stump_count: 0,
        }
    }

    fn step(&self, rng: &mut impl Rng, mut state: BartState) -> (BartState, BartInfo) {
        let data_view = self.data.view();
        let n_samples = self.data.n_samples();
        let n_trees = self.config.n_trees;

        let response = ResponseStrategies::from_name(&self.config.response)
            .expect("Unknown response strategy");

        let batch_frac = if state.tune {
            self.config.batch_tune
        } else {
            self.config.batch_post
        };
        let batch_size = ((batch_frac * n_trees as f64).round() as usize)
            .max(1)
            .min(n_trees);
        let batch_start = state.next_tree_idx;

        let mut acceptance_count = 0;
        let mut tree_depths = Vec::with_capacity(batch_size);
        let mut total_log_likelihood = 0.0;

        let mut variable_inclusion = vec![0u32; self.data.n_features()];

        let mut others_pred_buf = Array2::zeros((self.config.n_outputs, n_samples));
        let mut tree_pred_buf = Array2::zeros((self.config.n_outputs, n_samples));

        for k in 0..batch_size {
            let tree_idx = (state.next_tree_idx + k) % n_trees;

            // residuals = sum of all OTHER trees = predictions - old_tree.predict()
            state.forest[tree_idx].predict_training_into_multi(&mut tree_pred_buf, Some(data_view.x));
            others_pred_buf.assign(&state.predictions);
            others_pred_buf -= &tree_pred_buf;

            let (new_tree, step_info) = smc_step(
                rng,
                &state.predictions,
                &self.config,
                &data_view,
                &self.split_rules,
                &self.resampling,
                &self.weight_fn,
                state.forest[tree_idx].clone(),
                &state.leaf_sd,
                &response,
            );


            // predictions = residuals + new_tree.predict()
            new_tree.predict_training_into_multi(&mut tree_pred_buf, Some(data_view.x));
            state.predictions.assign(&others_pred_buf);
            state.predictions += &tree_pred_buf;

            total_log_likelihood += step_info.log_likelihood;
            acceptance_count += step_info.acceptance_count;

            let depth = (0..new_tree.size)
                .filter(|&i| new_tree.is_leaf(i))
                .map(|i| new_tree.get_depth(i) as u8)
                .max()
                .unwrap_or(0);
            tree_depths.push(depth);

            for sv in new_tree.split_var.iter().take(new_tree.size) {
                if *sv != LEAF_SENTINEL && !state.tune {
                    variable_inclusion[*sv as usize] += 1;
                }
            }

            state.iter += 1;
            if state.tune {
                if state.iter > 2 {
                    state.leaf_sd = state.running_sd.update(&tree_pred_buf);
                } else {
                    let _ = state.running_sd.update(&tree_pred_buf);
                }

            }

            state.forest[tree_idx] = new_tree;
        }
        state.variable_inclusion = variable_inclusion;

        state.next_tree_idx = (state.next_tree_idx + batch_size) % n_trees;

        let info = BartInfo {
            log_likelihood: total_log_likelihood,
            acceptance_count,
            tree_depths,
            batch_start,
            batch_size,

        };

        (state, info)
    }
}

/// Type-erased kernel for Python bindings.
pub trait ErasedKernel {
    fn init(&self, rng: &mut SmallRng) -> BartState;
    fn step(&self, rng: &mut SmallRng, state: BartState) -> (BartState, BartInfo);
}

impl<R, W> ErasedKernel for BartKernel<R, W>
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    fn init(&self, rng: &mut SmallRng) -> BartState {
        SamplingAlgorithm::init(self, rng)
    }

    fn step(&self, rng: &mut SmallRng, state: BartState) -> (BartState, BartInfo) {
        SamplingAlgorithm::step(self, rng, state)
    }
}
