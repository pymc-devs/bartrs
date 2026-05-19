//! Pure SMC step function for BART tree proposals.

use std::sync::Arc;

use numpy::ndarray::{Array, ArrayView1, Ix2};
use rand::Rng;
use rand::distr::weighted::WeightedIndex;
use rand_distr::{Distribution, Normal};

use crate::config::BartConfig;
use crate::data::DataView;
use crate::particle::{Particle};
use crate::resampling::ResamplingStrategy;
use crate::splitting::SplitRules;
use crate::tree::TreeArrays;
use crate::update::{MutationDecision, TreeProposal};
use crate::weight::WeightFn;

/// Diagnostics from a single SMC tree step.
pub struct SmcStepInfo {
    pub log_likelihood: f64,
    pub acceptance_count: usize,
}

/// Run one SMC step to produce a new tree.
pub fn smc_step<R, W>(
    rng: &mut impl Rng,
    sum_trees: &Array<f64, Ix2>,
    config: &BartConfig,
    data: &DataView,
    split_rules: &[SplitRules],
    resampling: &R,
    weight_fn: &W,
    current_tree: TreeArrays,
) -> (TreeArrays, SmcStepInfo)
where
    R: ResamplingStrategy,
    W: WeightFn,
{
    let n_samples = data.n_samples();
    let init_leaf_value = data.y.mean().unwrap_or(0.0) / config.n_trees as f64;

    let mut particles: Vec<Particle> = (0..config.n_particles)
        .map(|i| {
            if i == 0 {
                // Particle::new_reference(init_leaf_value, n_samples, config.max_depth, config.n_outputs)

                Particle::from_reference(current_tree.clone(), n_samples, current_tree.max_depth)

            } else {
                Particle::new(init_leaf_value, n_samples, config.max_depth, config.n_outputs)
            }
        })
        .collect();

    let n_non_ref = config.n_particles - 1;
    let mut inner_weights = vec![0.0f64; n_non_ref];
    let mut acceptance_count = 0;


    let mut sum_trees_noi = Array::zeros((config.n_outputs, n_samples));
    let mut current_tree_pred = Array::zeros((config.n_outputs, n_samples));
    current_tree.predict_training_into_multi(&mut current_tree_pred);
    sum_trees_noi.assign(sum_trees);
    sum_trees_noi -= &current_tree_pred;

    let mut predictions_buf = Array::zeros((config.n_outputs, n_samples));
    let mut ancestors_buf: Vec<usize> = Vec::with_capacity(n_non_ref);
    let mut scratch_particles: Vec<Particle> = Vec::with_capacity(n_non_ref);
    let mut mutated = vec![false; n_non_ref];

    while particles[1..].iter().any(|p| p.has_expandable_nodes()) {
        mutated.iter_mut().for_each(|m| *m = false);
        for (i, particle) in particles[1..].iter_mut().enumerate() {
            if let Some(node_idx) = particle.peek_next_expandable() {
                let node_idx = node_idx as usize;

                match propose_mutation(
                    rng,
                    particle,
                    node_idx,
                    sum_trees,
                    config,
                    data,
                    split_rules,
                ) {
                    MutationDecision::Accept(proposal) => {
                        particle.pop_next_expandable();
                        particle.apply_mutation(&proposal, data.x);
                        acceptance_count += 1;
                        mutated[i] = true;
                    }
                    MutationDecision::Reject => {
                        particle.pop_next_expandable();
                    }
                }
            }
        }

        for (i, particle) in particles[1..].iter().enumerate() {
            if mutated[i] {
                        predictions_buf.fill(0.0);
                        particle.tree.predict_training_into_multi(&mut predictions_buf);
                        predictions_buf += &sum_trees_noi;
                    let flat: Vec<f64> = predictions_buf.iter().copied().collect();
                    inner_weights[i] = weight_fn.log_weight(&flat);
            }
        }
        normalize_weights_inplace(&mut inner_weights);

        resampling.resample_into(rng, &inner_weights, &mut ancestors_buf);
        scratch_particles.clear();
        scratch_particles.extend(ancestors_buf.iter().map(|&idx| particles[1 + idx].clone()));
        particles.truncate(1);
        particles.append(&mut scratch_particles);
    }

    let mut log_weights = vec![0.0f64; config.n_particles];
    for (i, particle) in particles.iter().enumerate() {
        predictions_buf.fill(0.0);
        particle.tree.predict_training_into_multi(&mut predictions_buf);
        predictions_buf += &sum_trees_noi;
        let flat: Vec<f64> = predictions_buf.iter().copied().collect();
        log_weights[i] = weight_fn.log_weight(&flat);
    }

    let mut weights = log_weights.clone();


    normalize_weights_inplace(&mut weights);

    let dist = WeightedIndex::new(&weights).unwrap();
    let selected_idx = dist.sample(rng);

    let selected_log_like = log_weights[selected_idx];

    let selected_particle = particles.swap_remove(selected_idx);
    // Drop remaining particles now so their Arc refs are released before try_unwrap.
    drop(particles);
    let final_tree = match Arc::try_unwrap(selected_particle.tree) {
        Ok(tree) => tree,
        Err(arc) => (*arc).clone(),
    };

    let info = SmcStepInfo {
        log_likelihood: selected_log_like,
        acceptance_count,
    };

    (final_tree, info)
}

/// Propose a mutation for a particle at a given node.
fn propose_mutation(
    rng: &mut impl Rng,
    particle: &Particle,
    node_idx: usize,
    sum_trees: &Array<f64, Ix2>,
    config: &BartConfig,
    data: &DataView,
    split_rules: &[SplitRules],
) -> MutationDecision {
    let depth = particle.tree.get_depth(node_idx);
    if depth == 0 {
        // continue;
    } else {
        let prob_not_expanding = 1.0 - (config.alpha * (depth as f64).powf(-config.beta));
        if prob_not_expanding > rng.random::<f64>() {
            return MutationDecision::Reject;
        }
    }

    // let prob_not_expanding = 1.0 - (config.alpha * (1.0 + depth as f64).powf(-config.beta));
    // println!("For depth {}, the probability of not expanding {}", depth, prob_not_expanding);

    // if prob_not_expanding > rng.random::<f64>() {
    //     return MutationDecision::Reject;
    // }

    let node_samples = particle.leaf_samples(node_idx);
    if node_samples.is_empty() {
        return MutationDecision::Reject;
    }

    let split_var = if let Some(ref probs) = config.splitting_probs {
        sample_feature_from_probs(rng, probs.as_slice().unwrap())
    } else {
        rng.random_range(0..data.n_features())
    };

    let col = data.x.column(split_var);
    let feature_values = node_samples
        .iter()
        .map(|&s| unsafe { *col.uget(s as usize) });

    let split_strategy = &split_rules[split_var];
    let split_val = match split_strategy.sample_split_value(rng, feature_values) {
        Some(v) => v,
        None => return MutationDecision::Reject,
    };

    let (left_value, right_value) = propose_leaf_values(
        rng,
        node_samples,
        &col,
        split_val,
        sum_trees,
        config,
    );

    MutationDecision::Accept(TreeProposal {
        node_idx,
        split_var: split_var as u32,
        split_val,
        left_value,
        right_value,
    })
}

fn propose_leaf_values(
    rng: &mut impl Rng,
    node_samples: &[u32],
    col: &ArrayView1<f64>,
    split_val: f64,
    sum_trees: &Array<f64, Ix2>,
    config: &BartConfig,
) -> (Vec<f64>, Vec<f64>) {
    let n_outputs = config.n_outputs;

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
            if v < split_val {
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
        let noise = dist.sample(rng) * config.sigma;
        left_value[o] = if left_n[o] == 0 {
            noise
        } else {
            left_sum[o] / left_n[o] as f64 / config.n_trees as f64 + noise
        };


        let noise_r = dist.sample(rng) * config.sigma;
        right_value[o] = if right_n[o] == 0 {
            noise_r
        } else {
            right_sum[o] / right_n[o] as f64 / config.n_trees as f64 + noise_r
        };
    }

    (left_value, right_value)
}

fn sample_feature_from_probs(rng: &mut impl Rng, probs: &[f64]) -> usize {
    let total: f64 = probs.iter().sum();
    let mut target = rng.random::<f64>() * total;

    for (idx, &prob) in probs.iter().enumerate() {
        target -= prob;
        if target <= 0.0 {
            return idx;
        }
    }

    probs.len() - 1
}

/// Normalize log-weights in-place using log-sum-exp for numerical stability.
pub fn normalize_weights_inplace(weights: &mut [f64]) {
    let max_log_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    for w in weights.iter_mut() {
        *w = (*w - max_log_weight).exp();
    }

    let sum: f64 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}
