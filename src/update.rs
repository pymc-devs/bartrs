/// Represents the decision outcome for mutation proposals.
#[derive(Clone, Debug)]
pub enum MutationDecision {
    Accept(TreeProposal),
    Reject,
}

use crate::response::LeafProposal;

/// BART tree proposal containing all information needed for a growth mutation.
///
/// Sample partitioning is performed by the particle from its `leaf_to_samples`
/// cache, so the proposal does not carry the affected sample list.
#[derive(Clone, Debug)]
pub struct TreeProposal {
    pub leaf_proposal: LeafProposal,
}
