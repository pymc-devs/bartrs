from pymc.sampling import mcmc

from pymc_bartrs.bart import BART
from pymc_bartrs.compile_pymc import CompiledPyMCModel
from pymc_bartrs.pgbart import PGBART
from pymc_bartrs.utils import (
    compute_variable_importance,
    plot_convergence,
    plot_ice,
    plot_pdp,
    plot_scatter_submodels,
    plot_variable_importance,
    plot_variable_inclusion,
)

__all__ = [
    "BART",
    "PGBART",
    "CompiledPyMCModel",
    "compute_variable_importance",
    "plot_convergence",
    "plot_ice",
    "plot_pdp",
    "plot_scatter_submodels",
    "plot_variable_importance",
    "plot_variable_inclusion",
]

methods = mcmc.STEP_METHODS
if not any(method is PGBART for method in methods):
    methods.append(PGBART)
