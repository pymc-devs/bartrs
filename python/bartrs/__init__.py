from pymc.sampling import mcmc


from bartrs.compile_pymc import CompiledPyMCModel
from bartrs.pgbart import PGBART

__all__ = [
    "PGBART",
    "CompiledPyMCModel",
]

methods = mcmc.STEP_METHODS
if not any(method is PGBART for method in methods):
    methods.append(PGBART)
