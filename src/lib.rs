use std::ffi::c_double;

pub mod config;
pub mod data;
pub mod forest;
pub mod kernel;
pub mod particle;
pub mod resampling;
pub mod response;
pub mod smc;
pub mod splitting;
pub mod state;
pub mod tree;
pub mod update;
pub mod weight;

use crate::tree::{TreeArrays, TreeBatch};
use crate::config::BartConfig;
use crate::data::OwnedData;
use crate::kernel::{BartKernel, ErasedKernel, SamplingAlgorithm};
use crate::resampling::SystematicResampling;
use crate::splitting::{ContinuousSplit, SplitRules};
use crate::weight::PyMCWeightFn;

use numpy::{
    PyReadonlyArray, PyReadonlyArrayDyn, PyArray2, PyArray3, PyUntypedArrayMethods, PyReadonlyArray2,
    ToPyArray,
    ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayViewMut1, Ix2, Ix3, s},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;

type LogpFunc = unsafe extern "C" fn(*const f64, usize) -> c_double;

#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PyBartSettings {
    n_trees: usize,
    n_particles: usize,
    max_depth: u8,
    alpha: f64,
    beta: f64,
    sigma: f64,
    n_outputs: usize,
    split_prior: Vec<f64>,
    split_rules: Vec<String>,
    response_rule: String,
    resampling_rule: String,
    batch_tune: f64,
    batch_post: f64,
    seed: u64,
    n_draws: usize,
}

#[pymethods]
impl PyBartSettings {
    #[new]
    #[pyo3(signature = (
        n_trees,
        n_particles,
        max_depth,
        alpha,
        beta,
        sigma,
        n_outputs,
        split_prior,
        split_rules,
        response_rule,
        resampling_rule,
        batch_tune = 0.1,
        batch_post = 0.1,
        seed = 0,
        n_draws = 0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_trees: usize,
        n_particles: usize,
        max_depth: u8,
        alpha: f64,
        beta: f64,
        sigma: f64,
        n_outputs: usize,
        split_prior: Vec<f64>,
        split_rules: Vec<String>,
        response_rule: String,
        resampling_rule: String,
        batch_tune: f64,
        batch_post: f64,
        seed: u64,
        n_draws: usize,
    ) -> Self {
        Self {
            n_trees,
            n_particles,
            max_depth,
            alpha,
            beta,
            sigma,
            n_outputs,
            split_prior,
            split_rules,
            response_rule,
            resampling_rule,
            batch_tune,
            batch_post,
            seed,
            n_draws,
        }
    }
}

#[pyclass(unsendable)]
struct PySampler {
    kernel: Box<dyn ErasedKernel>,
    state: Option<crate::state::BartState>,
    rng: SmallRng,
    all_trees: Vec<TreeBatch>,
    baseline_forest: Option<Vec<TreeArrays>>,
    n_trees: usize,
    n_outputs: usize,
}

#[pymethods]
impl PySampler {
    #[staticmethod]
    fn init(
        x: PyReadonlyArray<f64, Ix2>,
        y: PyReadonlyArrayDyn<f64>,
        model: usize,
        settings: PyBartSettings,
    ) -> PyResult<PySampler> {
        let mut x_data = x.as_array().to_owned();
        // Accept either 1D or 2D `y`. If 1D, reshape to (1, n_samples).
        let y_array = match y.ndim() {
            1 => {
                let a = y.as_array();
                let n = a.len();
                let mut out = Array2::zeros((1, n));
                for i in 0..n {
                    out[[0, i]] = a[i];
                }
                out
            }
            2 => y.as_array().to_owned().into_dimensionality::<Ix2>().unwrap(),
            _ => return Err(PyValueError::new_err("y must be 1D or 2D array")),
        };
        let y_data = y_array;

        let logp_func: LogpFunc = unsafe { std::mem::transmute(model as *const ()) };
        let weight_fn = unsafe { PyMCWeightFn::from_raw(logp_func) };

        // Parse split rules
        let split_rules: Vec<SplitRules> = settings
            .split_rules
            .iter()
            .map(|rule| SplitRules::from_name(rule).map_err(PyValueError::new_err))
            .collect::<PyResult<Vec<SplitRules>>>()?;

        // Fill with continuous splits if not enough rules provided
        let n_features = x_data.ncols();
        let split_rules = if split_rules.len() < n_features {
            let mut rules = split_rules;
            rules.resize(n_features, SplitRules::Continuous(ContinuousSplit));
            rules
        } else {
            split_rules
        };

        let mut rng = SmallRng::seed_from_u64(settings.seed);

        for (idx, rule) in split_rules.iter().enumerate() {
            if matches!(rule, SplitRules::Continuous(_)) {
                let std = nanstd(x_data.column(idx));
                let col = x_data.column_mut(idx);
                jitter_duplicated(col, std, &mut rng);
            }
        }

        let config = BartConfig {
            n_trees: settings.n_trees,
            n_particles: settings.n_particles,
            max_depth: settings.max_depth,
            alpha: settings.alpha,
            beta: settings.beta,
            sigma: settings.sigma,
            n_outputs: settings.n_outputs,
            min_samples_leaf: 5, // try out min_samples_leaf=5
            splitting_probs: if settings.split_prior.is_empty() {
                None
            } else {
                Some(Array1::from_vec(settings.split_prior))
            },
            batch_tune: settings.batch_tune,
            batch_post: settings.batch_post,
            response: settings.response_rule,
        };

        let data = OwnedData::new(x_data, y_data);

        let all_trees: Vec<TreeBatch> = if settings.n_draws == 0 {
            Vec::new()
        } else {
            Vec::with_capacity(settings.n_draws)
        };

        let kernel = BartKernel {
            split_rules,
            resampling: SystematicResampling,
            weight_fn,
            config,
            data,
        };

        let state = SamplingAlgorithm::init(&kernel, &mut rng);

        Ok(PySampler {
            kernel: Box::new(kernel),
            state: Some(state),
            rng,
            all_trees,
            baseline_forest: None,
            n_trees: settings.n_trees,
            n_outputs: settings.n_outputs,
        })
    }

    fn reserve_draws(&mut self, draws: usize) {
        self.all_trees = Vec::with_capacity(draws);
    }

    #[pyo3(signature = (tune = None))]
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        tune: Option<bool>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Vec<u32>,
        Option<(usize, Vec<TreeArrays>)>,
        Option<Vec<TreeArrays>>,
    )> {

        let mut state = self
            .state
            .take()
            .ok_or_else(|| PyValueError::new_err("Sampler state is missing (internal error)"))?;

        if let Some(t) = tune {
            state.tune = t;
        }
        let was_tune = state.tune;

        // `new_baseline` mirrors what gets stored in `self.baseline_forest`, but only on the
        // step where it's first computed, so callers can stream it back to Python exactly once.
        let mut new_baseline: Option<Vec<TreeArrays>> = None;
        if !was_tune && self.baseline_forest.is_none() {
            let mut forest = state.forest.clone();
            for t in &mut forest { t.leaf_indices = Vec::new(); }
            self.baseline_forest = Some(forest.clone());
            new_baseline = Some(forest);
        }


        let (new_state, info) = self.kernel.step(&mut self.rng, state);

        let mut new_batch: Option<(usize, Vec<TreeArrays>)> = None;
        if !was_tune {
            let mut trees = Vec::with_capacity(info.batch_size);
            for k in 0..info.batch_size {
                let idx = (info.batch_start + k) % self.n_trees;
                let mut t = new_state.forest[idx].clone();
                t.leaf_indices = Vec::new();
                trees.push(t);
            }
            self.all_trees.push(TreeBatch { start: info.batch_start, trees: trees.clone() });
            new_batch = Some((info.batch_start, trees));
        }


        // Return predictions as a 2-D array with shape (n_outputs, n_samples)
        let result = numpy::PyArray2::from_owned_array(py, new_state.predictions.clone());

        self.state = Some(new_state);

        let variable_inclusion = self.state.as_ref().unwrap().get_variable_inclusion();

        Ok((result, variable_inclusion, new_batch, new_baseline))
    }

    fn sample_posterior<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>, samples: usize, excluded: Option<&Bound<'py, PyList>>) -> PyResult<Py<PyArray3<f64>>>{

        let data = x.as_array().to_owned();

        let excl = match excluded {
            Some(list) => list.iter()
                .map(|item| item.extract::<usize>())
                .collect::<PyResult<Vec<usize>>>()?,
            None => Vec::new(),
        };

        let preds = self._sample_posterior(&data, samples, &excl).to_pyarray(py).unbind();
        return Ok(preds)
    }
}

impl PySampler {
    fn _sample_posterior(&mut self, x: &Array<f64, Ix2>, samples: usize, excluded: &Vec<usize>) -> Array<f64, Ix3> {
        predict_from_history(
            &self.all_trees,
            self.baseline_forest.as_deref(),
            self.n_trees,
            self.n_outputs,
            x,
            samples,
            excluded,
            &mut self.rng,
        )
    }

}

#[allow(clippy::too_many_arguments)]
fn predict_from_history(
    batches: &[TreeBatch],
    baseline_forest: Option<&[TreeArrays]>,
    n_trees: usize,
    n_outputs: usize,
    x: &Array<f64, Ix2>,
    samples: usize,
    excluded: &Vec<usize>,
    rng: &mut SmallRng,
) -> Array<f64, Ix3> {
    let n_data_samples: usize = x.nrows();
    let n_draws = batches.len();

    assert!(n_draws > 0, "No posterior draws stored yet");

    let random_samples: Vec<usize> = (0..samples)
        .map(|_| rng.random_range(0..n_draws as u32) as usize)
        .collect();

    let mut predictions = Array3::zeros((samples, n_outputs, n_data_samples));

    for posterior_sample_idx in 0..samples {
        let draw_idx = random_samples[posterior_sample_idx];
        let mut sample = predictions.slice_mut(s![posterior_sample_idx, .., ..]);

        let mut covered = vec![false; n_trees];
        let mut remaining = n_trees;

        let mut b = draw_idx as isize;
        while remaining > 0 && b >= 0 {
            let batch = &batches[b as usize];
            for (k, tree) in batch.trees.iter().enumerate() {
                let tree_idx = (batch.start + k) % n_trees;
                if !covered[tree_idx] {
                    covered[tree_idx] = true;
                    remaining -= 1;
                    sample += &tree.predict_batch_test(x, excluded);
                }
            }
            b -= 1;
        }

        if remaining > 0 {
            if let Some(baseline) = baseline_forest {
                for tree_idx in 0..n_trees {
                    if !covered[tree_idx] {
                        sample += &baseline[tree_idx].predict_batch_test(x, excluded);
                    }
                }
            }
        }
    }

    predictions
}


#[pyclass(module = "bartrs.bartrs", weakref)]
struct PosteriorSampler {
    batches: Vec<TreeBatch>,
    baseline_forest: Option<Vec<TreeArrays>>,
    n_trees: usize,
    n_outputs: usize,
    rng: SmallRng,
}

#[pymethods]
impl PosteriorSampler {
    #[staticmethod]
    fn from_history(
        batches: Vec<(usize, Vec<TreeArrays>)>,
        baseline_forest: Option<Vec<TreeArrays>>,
        n_trees: usize,
        n_outputs: usize,
        seed: u64,
    ) -> Self {
        let batches = batches
            .into_iter()
            .map(|(start, trees)| TreeBatch { start, trees })
            .collect();
        Self {
            batches,
            baseline_forest,
            n_trees,
            n_outputs,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    #[getter]
    fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    fn sample_posterior<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>, samples: usize, excluded: Option<&Bound<'py, PyList>>) -> PyResult<Py<PyArray3<f64>>> {
        let data = x.as_array().to_owned();

        let excl = match excluded {
            Some(list) => list.iter()
                .map(|item| item.extract::<usize>())
                .collect::<PyResult<Vec<usize>>>()?,
            None => Vec::new(),
        };

        let preds = predict_from_history(
            &self.batches,
            self.baseline_forest.as_deref(),
            self.n_trees,
            self.n_outputs,
            &data,
            samples,
            &excl,
            &mut self.rng,
        ).to_pyarray(py).unbind();

        Ok(preds)
    }
}


#[pymodule]
fn bartrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBartSettings>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PosteriorSampler>()?;
    m.add_class::<TreeArrays>()?;
    Ok(())
}

fn jitter_duplicated(mut col: ArrayViewMut1<'_, f64>, std: f64, rng: &mut impl Rng) {
    if !are_whole_number(col.view()) {
        return;
    }

    let mut seen: Vec<f64> = Vec::new();
    let scale = std / 12.0;
    for value in col.iter_mut() {
        let num = *value;
        if seen.contains(&num) && !num.is_nan() {
            let z: f64 = rng.sample(StandardNormal);
            *value = num + scale * z;
        } else {
            seen.push(num);
        }
    }
}

fn are_whole_number(col: ArrayView1<'_, f64>) -> bool {
    for &value in col.iter() {
        if value.is_nan() {
            continue;
        }
        if value % 1.0 != 0.0 {
            return false;
        }
    }
    true
}

fn nanstd(col: ArrayView1<'_, f64>) -> f64 {
    let mut count: usize = 0usize;
    let mut mean: f64 = 0.0f64;
    let mut m2: f64 = 0.0f64;

    for &value in col.iter() {
        if value.is_nan() {
            continue;
        }
        count += 1;
        let delta: f64 = value - mean;
        mean += delta / count as f64;
        let delta2: f64 = value - mean;
        m2 += delta * delta2;
    }

    if count == 0 {
        return f64::NAN;
    }

    (m2 / count as f64).sqrt()
}