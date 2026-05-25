from pymc.sampling import mcmc


from pymc_bartrs.compile_pymc import CompiledPyMCModel
from pymc_bartrs.pgbart import PGBART

__all__ = [
    "PGBART",
    "CompiledPyMCModel",
]

methods = mcmc.STEP_METHODS
if not any(method is PGBART for method in methods):
    methods.append(PGBART)
