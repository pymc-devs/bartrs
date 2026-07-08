use numpy::ndarray::{Array2, Array1};

use crate::tree::TreeArrays;
use crate::kernel::RunningSd;

/// Complete state of the BART sampler, consumed and reproduced by each step.
#[derive(Clone)]
pub struct BartState {
    pub forest: Vec<TreeArrays>,
    pub predictions: Array2<f64>,
    pub variable_inclusion: Vec<u32>,
    /// Round-robin index of the next tree to update.
    pub next_tree_idx: usize,
    /// Whether the sampler is in tune mode (selects which batch fraction to use and also whether to update running_sd).
    pub tune: bool,
    /// Used to compute the std's of the leaves
    pub running_sd: RunningSd,
    /// Vector to store independent standard deviations of leaves. If single-output, will be vector of length one.
    pub leaf_sd: Array1<f64>, 
    /// Number of iterations. Used to check when to start updating the leaf_sd vector during tuning in kernel.rs
    pub iter: usize,
    /// Count of times a stump (no splits) was selected as final tree
    pub stump_count: usize,
}

impl BartState { 
    pub fn get_variable_inclusion(&self) -> Vec<u32> {
        self.variable_inclusion.clone()
    }
}

/// Diagnostic information from a single sampling step.
pub struct BartInfo {
    pub log_likelihood: f64,
    pub acceptance_count: usize,
    pub tree_depths: Vec<u8>,
    /// Index of the start of trees updated this step
    pub batch_start: usize,
    /// Number of trees updated per step
    pub batch_size: usize,
}
