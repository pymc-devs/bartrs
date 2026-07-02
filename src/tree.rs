use numpy::ndarray::{Array, Array1, Array2, ArrayView2, Ix1, Ix2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::response::{LeafKind, LeafPayload, LeafProposal};
use crate::data::NotNan;

/// Bartz-style heap-indexed tree with separate internal/leaf arrays.
///
/// Uses heap convention: root=0, left=2i+1, right=2i+2.
/// Internal nodes store split variable and threshold. Leaf nodes store
/// predicted values. The `leaf_indices` vector maps each training sample
/// to its assigned leaf node.
#[pyclass(module = "bartrs.bartrs", get_all, from_py_object)]
#[derive(Clone, Debug)]
pub struct TreeArrays {
    /// Split variable per node (u32::MAX = leaf sentinel)
    pub split_var: Vec<u32>,
    /// Split threshold per node (NaN for leaves)
    pub split_val: Vec<f64>,
    /// Flattened leaf values per node
    pub leaf_val: Vec<f64>,
    /// Sample -> leaf node mapping for training data
    pub leaf_indices: Vec<u32>,
    /// Number of allocated nodes in the tree
    pub size: usize,
    /// Maximum allowed depth
    pub max_depth: u8,
    /// Number of outputs (for multi-output tasks). If 1, behaviour is unchanged.
    pub n_outputs: usize,
    /// Counts of samples in each node
    pub node_nvalue: Vec<f64>,
    /// Leaf type per node (0 = Gaussian, 1 = Linear, etc.)
    pub leaf_kind: Vec<u8>,
    /// Index into linear parameter arrays for linear leaf nodes, or usize::MAX if not applicable
    pub leaf_param_idx: Vec<usize>,
    /// Separate storage for linear leaf intercept
    pub linear_intercept: Vec<Vec<f64>>,
    /// Separate storage for linear leaf slope
    pub linear_slope: Vec<Vec<f64>>,
    /// Tells which variable is used in the linear leaf
    pub linear_var: Vec<u32>,
}

pub const LEAF_SENTINEL: u32 = u32::MAX;
const LEAF_PARAM_NONE: usize = usize::MAX;
 
#[pymethods]
impl TreeArrays {
    /// Create a new tree with just a root leaf node.
    #[new]
    pub fn new(init_leaf_value: f64, n_samples: usize, max_depth: u8, n_outputs: usize) -> Self {
        let max_nodes = max_nodes_for_depth(max_depth);

        let mut split_var = Vec::with_capacity(max_nodes);
        let mut split_val = Vec::with_capacity(max_nodes);
        let mut leaf_val = Vec::with_capacity(max_nodes * n_outputs.max(1));

        let mut leaf_kind = Vec::with_capacity(max_nodes);
        let mut leaf_param_idx = Vec::with_capacity(max_nodes);

        // Root is a leaf
        split_var.push(LEAF_SENTINEL);
        split_val.push(f64::NAN);
        for _ in 0..n_outputs.max(1) {
            leaf_val.push(init_leaf_value);
        }
        leaf_kind.push(LeafKind::Gaussian.as_u8());
        leaf_param_idx.push(LEAF_PARAM_NONE);

        let mut new_self = Self {
            split_var,
            split_val,
            leaf_val,
            leaf_indices: vec![0; n_samples],
            size: 1,
            max_depth,
            n_outputs: n_outputs.max(1),
            node_nvalue: vec![0.0; max_nodes],
            leaf_kind,
            leaf_param_idx,
            linear_intercept: Vec::new(),
            linear_slope: Vec::new(),
            linear_var: Vec::new(),
        };

        new_self.node_nvalue[0] = n_samples as f64;

        new_self
    }


    pub fn to_state<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("split_var", self.split_var.clone())?;
        dict.set_item("split_val", self.split_val.clone())?;
        dict.set_item("leaf_val", self.leaf_val.clone())?;
        dict.set_item("size", self.size)?;
        dict.set_item("max_depth", self.max_depth)?;
        dict.set_item("n_outputs", self.n_outputs)?;
        dict.set_item("node_nvalue", self.node_nvalue.clone())?;
        dict.set_item("leaf_kind", self.leaf_kind.clone())?;
        dict.set_item("leaf_param_idx", self.leaf_param_idx.clone())?;
        dict.set_item("linear_intercept", self.linear_intercept.clone())?;
        dict.set_item("linear_slope", self.linear_slope.clone())?;
        dict.set_item("linear_var", self.linear_var.clone())?;
        Ok(dict.into())
    }

    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        self.to_state(py)
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let dict: &Bound<'_, PyDict> = state.cast::<PyDict>()?;
        self.split_var = dict.get_item("split_var").unwrap().expect("split_var not found").extract()?;
        self.split_val = dict.get_item("split_val").unwrap().expect("split_val not found").extract()?;
        self.leaf_val = dict.get_item("leaf_val").unwrap().expect("leaf_val not found").extract()?;
        self.size = dict.get_item("size").unwrap().expect("size not found").extract()?;
        self.max_depth = dict.get_item("max_depth").unwrap().expect("max_depth not found").extract()?;
        self.n_outputs = dict.get_item("n_outputs").unwrap().expect("n_outputs not found").extract()?;
        self.node_nvalue = dict.get_item("node_nvalue").unwrap().expect("node_nvalue not found").extract()?;
        self.leaf_kind = dict.get_item("leaf_kind").unwrap().expect("leaf_kind not found").extract()?;
        self.leaf_param_idx = dict.get_item("leaf_param_idx").unwrap().expect("leaf_param_idx not found").extract()?;
        self.linear_intercept = dict.get_item("linear_intercept").unwrap().expect("linear_intercept not found").extract()?;
        self.linear_slope = dict.get_item("linear_slope").unwrap().expect("linear_slope not found").extract()?;
        self.linear_var = dict.get_item("linear_var").unwrap().expect("linear_var not found").extract()?;
        Ok(())
    } 

    pub fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Py<PyAny>, (f64, usize, u8, usize), Py<PyDict>)> {
        let cls = py.get_type::<TreeArrays>();
        // args are what `new(init_leaf_value, n_samples, max_depth, n_outputs)` expects
        let args = (self.leaf_val.get(0).copied().unwrap_or(0.0), self.leaf_indices.len(), self.max_depth, self.n_outputs);
        let state = self.to_state(py)?;
        Ok((cls.as_any().clone().unbind(), args, state))
    }


    pub fn predict<'py>(&self, x: PyReadonlyArray2<'py, f64>, py: Python<'py>, excluded: Option<&Bound<'py, PyList>>) -> PyResult<Py<PyArray2<f64>>> {
        let data = x.as_array().to_owned();
        let excl = match excluded {
            Some(list) => list.iter()
                .map(|item| item.extract::<usize>())
                .collect::<PyResult<Vec<usize>>>()?,
            None => Vec::new(),
        };
        let preds = self.predict_batch_test(&data, &excl).into_pyarray(py).unbind();
        
        Ok(preds)
    }

    #[getter]
    pub fn n_outputs(&self) -> PyResult<usize> {
        Ok(self.n_outputs)
    }

}

impl TreeArrays {
    /// Get the depth of a node in the binary tree.
    ///
    /// Heap layout: depth(i) = floor(log2(i + 1)).
    #[inline]
    pub fn get_depth(&self, node_idx: usize) -> usize {
        63 - ((node_idx + 1) as u64).leading_zeros() as usize
    }

    /// Check if a node is a leaf (no split variable assigned).
    pub fn is_leaf(&self, node_idx: usize) -> bool {
        node_idx < self.split_var.len() && self.split_var[node_idx] == LEAF_SENTINEL
    }

    /// Get all leaf node indices.
    pub fn get_leaf_indices(&self) -> Vec<usize> {
        (0..self.size).filter(|&i| self.is_leaf(i)).collect()
    }

    /// Get data (sample) indices for a leaf node.
    pub fn get_leaf_samples(&self, leaf_idx: usize) -> impl Iterator<Item = usize> + '_ {
        debug_assert!(self.is_leaf(leaf_idx), "Node {} is not a leaf", leaf_idx);
        let leaf_idx = leaf_idx as u32;
        self.leaf_indices
            .iter()
            .enumerate()
            .filter_map(move |(sample_idx, &assigned_leaf)| {
                if assigned_leaf == leaf_idx {
                    Some(sample_idx)
                } else {
                    None
                }
            })
    }

    /// Split a leaf node into an internal node with two new child leaves.
    pub fn split_node(&mut self, leaf_idx: usize, split_var: u32, split_val: f64, leaf_proposal: LeafProposal) {
        let left_child = 2 * leaf_idx + 1;
        let right_child = 2 * leaf_idx + 2;

        let max_nodes = max_nodes_for_depth(self.max_depth);
        assert!(
            right_child < max_nodes,
            "Tree mutation would exceed maximum capacity of {}",
            max_nodes
        );

        // Extend vectors if needed (push per-node entries)
        let required_size = right_child + 1;
        while self.split_var.len() < required_size {
            self.split_var.push(LEAF_SENTINEL);
            self.split_val.push(f64::NAN);
            for _ in 0..self.n_outputs {
                self.leaf_val.push(0.0);
            }
            self.leaf_kind.push(LeafKind::Gaussian.as_u8());
            self.leaf_param_idx.push(LEAF_PARAM_NONE);
        }

        // Convert leaf to internal node
        self.split_var[leaf_idx] = split_var;
        self.split_val[leaf_idx] = split_val;
        for o in 0..self.n_outputs {
            let idx = leaf_idx * self.n_outputs + o;
            self.leaf_val[idx] = f64::NAN;
        }

        // Set left child as leaf
        self.split_var[left_child] = LEAF_SENTINEL;
        self.split_val[left_child] = f64::NAN;
        self.apply_leaf_payload(left_child, leaf_proposal.left);

        self.split_var[right_child] = LEAF_SENTINEL;
        self.split_val[right_child] = f64::NAN;
        self.apply_leaf_payload(right_child, leaf_proposal.right);

        self.size = self.size.max(right_child + 1);
    }

    /// Update leaf assignments after a split using branchless bit trick.
    pub fn update_leaf_assignments(&mut self, split_node_idx: usize, split_var: u32, split_val: f64, affected_samples: &[usize], x_data: ArrayView2<f64>) {
        let base_child = (2 * split_node_idx + 1) as u32;

        for &sample_idx in affected_samples {
            let sample_val = x_data[[sample_idx, split_var as usize]];
            let child_offset = (sample_val > split_val) as u32;
            self.leaf_indices[sample_idx] = base_child + child_offset;
        }
    }

    /// Predict training data into a new 2-D array of shape (n_outputs, n_samples).
    pub fn predict_training(&self) -> Array<f64, Ix2> {
        let n_samples = self.leaf_indices.len();
        let mut out = Array2::zeros((self.n_outputs, n_samples));
        for s in 0..n_samples {
            let leaf = self.leaf_indices[s] as usize;
            for o in 0..self.n_outputs {
                out[[o, s]] = self.leaf_val[leaf * self.n_outputs + o];
            }
        }
        out
    }

    /// Predict training data into a pre-allocated single-output buffer (first output).
    pub fn predict_training_into(&self, out: &mut Array<f64, Ix1>) {
        let out_slice = out.as_slice_mut().expect("predictions buffer must be contiguous");
        for (dst, &leaf_idx) in out_slice.iter_mut().zip(self.leaf_indices.iter()) {
            let val = self.leaf_val[leaf_idx as usize * self.n_outputs];
            *dst = val;
        }
    }

    /// Multi-output prediction into a pre-allocated buffer with shape (n_outputs, n_samples).
    pub fn predict_training_into_multi(&self, out: &mut Array<f64, Ix2>, x_data: Option<ArrayView2<f64>>) {
        let n_outputs = out.nrows();
        let n_samples = out.ncols();
        debug_assert_eq!(n_samples, self.leaf_indices.len());

        for sample_idx in 0..n_samples {
            let leaf_idx = self.leaf_indices[sample_idx] as usize;
            self.fill_training_leaf_value(leaf_idx, sample_idx, n_outputs, x_data, out);
        }
    }

    /// Predict on test data by traversing the tree. Returns shape (n_outputs, n_test_samples).
    pub fn predict_batch_test(&self, data: &Array<f64, Ix2>, excluded: &Vec<usize>) -> Array<f64, Ix2> {
        let n_samples = data.nrows();
        let mut predictions = Array2::zeros((self.n_outputs, n_samples));

        let mut stack: Vec<(usize, Array1<f64>, usize)> = vec![(0usize, Array1::ones(n_samples), 0usize)];

        while !stack.is_empty() {
            let (node_idx, weights, _idx_split) = stack.pop().unwrap();


            if self.is_leaf(node_idx) {
                self.add_leaf_prediction(node_idx, &weights, data, &mut predictions);
            } else { 
                
                let sv = self.split_var[node_idx] as usize; // split var
                let st = self.split_val[node_idx];          // split threshold

                let left = 2*node_idx + 1;
                let right = 2*node_idx + 2;

                if (!excluded.is_empty()) && excluded.contains(&sv) {
                    let node_nvalue = self.node_nvalue[node_idx];

                    let left_nvalue = self.node_nvalue[left];

                    let prop_nvalue_left = if node_nvalue > 0.0 { left_nvalue / node_nvalue } else { 0.0 };

                    stack.push((left, weights.clone()*prop_nvalue_left, sv));
                    stack.push((right, weights.clone()*(1.0 - prop_nvalue_left), sv));
                } else { 
                    let to_left = data.column(sv).map(|i| {
                        if *i <= st {
                            1.0
                        } else {
                            0.0
                        }
                    });
                    
                    stack.push((left, weights.clone()*to_left.clone(), sv));
                    stack.push((right, weights.clone()*(1.0 - to_left.clone()), sv));
                }
            }
        }

        predictions 
    }       

    fn apply_leaf_payload(&mut self, node_idx: usize, payload: LeafPayload) {
        match payload {
            LeafPayload::Gaussian { value } => {
                self.leaf_kind[node_idx] = LeafKind::Gaussian.as_u8();
                self.leaf_param_idx[node_idx] = LEAF_PARAM_NONE;
                for o in 0..self.n_outputs {
                    let dst = node_idx * self.n_outputs + o;
                    self.leaf_val[dst] = value.get(o).copied().unwrap_or(0.0);
                }
            }
            LeafPayload::Linear { intercept, slope, var } => {
                let idx = self.linear_intercept.len();
                self.linear_intercept.push(intercept);
                self.linear_slope.push(slope);
                self.linear_var.push(var);
                self.leaf_kind[node_idx] = LeafKind::Linear.as_u8();
                self.leaf_param_idx[node_idx] = idx;
                for o in 0..self.n_outputs {
                    let dst = node_idx * self.n_outputs + o;
                    self.leaf_val[dst] = f64::NAN;
                }
            }
        }
    }

    fn add_leaf_prediction(
        &self,
        node_idx: usize,
        weights: &Array1<f64>,
        data: &Array<f64, Ix2>,
        predictions: &mut Array2<f64>,
    ) {
        match LeafKind::from_u8(self.leaf_kind[node_idx]) {
            LeafKind::Gaussian => {
                for out_idx in 0..self.n_outputs {
                    let mut row = predictions.row_mut(out_idx);
                    row += &(weights.clone() * self.leaf_val[node_idx * self.n_outputs + out_idx]);
                }
            }
            LeafKind::Linear => {
                let param_idx = self.leaf_param_idx[node_idx];
                if param_idx == LEAF_PARAM_NONE {
                    return;
                }
                let var = self.linear_var[param_idx] as usize;
                for out_idx in 0..self.n_outputs {
                    let mut row = predictions.row_mut(out_idx);
                    let intercept = self.linear_intercept[param_idx].get(out_idx).copied().unwrap_or(0.0);
                    let slope = self.linear_slope[param_idx].get(out_idx).copied().unwrap_or(0.0);
                    let mut contrib = data.column(var).to_owned();
                    contrib.mapv_inplace(|x| {
                        if x.is_valid() {
                            intercept + slope * x
                        } else {
                            0.0
                        }
                    });
                    row += &(weights.clone() * contrib);
                }
            }
        }
    }

    fn fill_training_leaf_value(
        &self,
        leaf_idx: usize,
        sample_idx: usize,
        n_outputs: usize,
        x_data: Option<ArrayView2<f64>>,
        out: &mut Array<f64, Ix2>,
    ) {
        match LeafKind::from_u8(self.leaf_kind[leaf_idx]) {
            LeafKind::Gaussian => {
                for out_idx in 0..n_outputs {
                    let val = self.leaf_val[leaf_idx * self.n_outputs + out_idx];
                    out[[out_idx, sample_idx]] = val;
                }
            }
            LeafKind::Linear => {
                let Some(x_data) = x_data else {
                    return;
                };
                let param_idx = self.leaf_param_idx[leaf_idx];
                if param_idx == LEAF_PARAM_NONE {
                    return;
                }
                let var = self.linear_var[param_idx] as usize;
                let x = x_data[[sample_idx, var]];
                for out_idx in 0..n_outputs {
                    let intercept = self.linear_intercept[param_idx].get(out_idx).copied().unwrap_or(0.0);
                    let slope = self.linear_slope[param_idx].get(out_idx).copied().unwrap_or(0.0);
                    out[[out_idx, sample_idx]] = if x.is_nan() {
                        intercept
                    } else {
                        intercept + slope * x
                    };
                }
            }
        }
    }
}

/// Calculate maximum number of nodes for a given depth.
pub fn max_nodes_for_depth(depth: u8) -> usize {
    (1usize << (depth as usize + 1)) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree_is_single_leaf() {
        let tree = TreeArrays::new(1.5, 10, 5, 1);
        assert_eq!(tree.size, 1);
        assert!(tree.is_leaf(0));
        assert_eq!(tree.leaf_val[0], 1.5);
        assert_eq!(tree.leaf_indices.len(), 10);
        assert!(tree.leaf_indices.iter().all(|&idx| idx == 0));
    }

    #[test]
    fn test_split_node_creates_children() {
        let mut tree = TreeArrays::new(0.0, 5, 5, 1);
        tree.split_node(
            0,
            0,
            2.5,
            LeafProposal {
                node_idx: 0,
                split_var: 0,
                split_val: 2.5,
                left: LeafPayload::Gaussian { value: vec![-1.0] },
                right: LeafPayload::Gaussian { value: vec![1.0] },
            },
        );

        assert!(!tree.is_leaf(0));
        assert!(tree.is_leaf(1));
        assert!(tree.is_leaf(2));
        assert_eq!(tree.leaf_val[1], -1.0);
        assert_eq!(tree.leaf_val[2], 1.0);
        assert_eq!(tree.size, 3);
    }

    #[test]
    fn test_get_depth() {
        let tree = TreeArrays::new(0.0, 1, 5, 1);
        assert_eq!(tree.get_depth(0), 0);
        assert_eq!(tree.get_depth(1), 1);
        assert_eq!(tree.get_depth(2), 1);
        assert_eq!(tree.get_depth(3), 2);
        assert_eq!(tree.get_depth(6), 2);
    }

    #[test]
    fn test_max_nodes_for_depth() {
        assert_eq!(max_nodes_for_depth(0), 1);
        assert_eq!(max_nodes_for_depth(1), 3);
        assert_eq!(max_nodes_for_depth(2), 7);
        assert_eq!(max_nodes_for_depth(5), 63);
        assert_eq!(max_nodes_for_depth(6), 127);
        assert_eq!(max_nodes_for_depth(9), 1023);
    }

    #[test]
    fn test_predict_training_root_only() {
        let tree = TreeArrays::new(3.14, 4, 5, 1);
        let preds = tree.predict_training();
        assert_eq!(preds.shape(), &[1,4]);
        assert!(preds.iter().all(|&v| (v - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_get_leaf_samples() {
        let mut tree = TreeArrays::new(0.0, 4, 5, 1);
        // Manually assign samples to different leaves
        tree.split_node(
            0,
            0,
            0.5,
            LeafProposal {
                node_idx: 0,
                split_var: 0,
                split_val: 0.5,
                left: LeafPayload::Gaussian { value: vec![-1.0] },
                right: LeafPayload::Gaussian { value: vec![1.0] },
            },
        );
        tree.leaf_indices = vec![1, 1, 2, 2]; // samples 0,1 -> left; 2,3 -> right

        let left_samples: Vec<usize> = tree.get_leaf_samples(1).collect();
        let right_samples: Vec<usize> = tree.get_leaf_samples(2).collect();

        assert_eq!(left_samples, vec![0, 1]);
        assert_eq!(right_samples, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "Tree mutation would exceed maximum capacity")]
    fn test_split_beyond_max_depth_panics() {
        let mut tree = TreeArrays::new(0.0, 1, 1, 1); // max_depth=1, max_nodes=3
        tree.split_node(
            0,
            0,
            0.5,
            LeafProposal {
                node_idx: 0,
                split_var: 0,
                split_val: 0.5,
                left: LeafPayload::Gaussian { value: vec![-1.0] },
                right: LeafPayload::Gaussian { value: vec![1.0] },
            },
        ); // OK: creates nodes 1,2
        tree.split_node(
            1,
            0,
            0.3,
            LeafProposal {
                node_idx: 1,
                split_var: 0,
                split_val: 0.3,
                left: LeafPayload::Gaussian { value: vec![-2.0] },
                right: LeafPayload::Gaussian { value: vec![2.0] },
            },
        ); // Panic: would need nodes 3,4 but max=3
    }

    #[test]
    fn test_multioutput_leaf_values() {
        let mut tree = TreeArrays::new(0.0, 3, 3, 2);
        // root leaf has two outputs both init 0.0
        assert_eq!(tree.leaf_val.len(), 2);
        // split root, set left=[1,2], right=[3,4]
        tree.split_node(
            0,
            0,
            0.5,
            LeafProposal {
                node_idx: 0,
                split_var: 0,
                split_val: 0.5,
                left: LeafPayload::Gaussian { value: vec![1.0, 2.0] },
                right: LeafPayload::Gaussian { value: vec![3.0, 4.0] },
            },
        );
        assert_eq!(tree.leaf_val[1 * 2 + 0], 1.0);
        assert_eq!(tree.leaf_val[1 * 2 + 1], 2.0);
        assert_eq!(tree.leaf_val[2 * 2 + 0], 3.0);
        assert_eq!(tree.leaf_val[2 * 2 + 1], 4.0);

        // assign samples and test predict_training
        tree.leaf_indices = vec![1,1,2];
        let preds = tree.predict_training();
        assert_eq!(preds.shape(), &[2,3]);
        assert_eq!(preds[[0,0]], 1.0);
        assert_eq!(preds[[1,2]], 4.0);
    }
}
