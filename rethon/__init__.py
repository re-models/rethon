import logging
import json
import importlib.resources as pkg_resources
from . import config


# Todo: Add and test checking dependencies

from .base import (
    ReflectiveEquilibrium,
    StandardReflectiveEquilibrium,
    LocalReflectiveEquilibrium,
    GlobalReflectiveEquilibrium,
    REContainer,
    REState
)
from .core import (
    StandardGlobalReflectiveEquilibrium,
    StandardLocalReflectiveEquilibrium,
    FullBranchREContainer
)
from .ensemble_generation import (
    AbstractEnsembleGenerator,
    EnsembleGenerator,
    SimpleEnsembleGenerator,
    GlobalREEnsembleGenerator,
    LocalREEnsembleGenerator,
    SimpleMultiAgentREContainer,
    MultiAgentEnsemblesGenerator,
    SimpleMultiAgentEnsemblesGenerator
)
from .bitarray_implementation import GlobalBitarrayReflectiveEquilibrium
from .numpy_implementation import (
    NumpyReflectiveEquilibrium,
    GlobalNumpyReflectiveEquilibrium,
    LocalNumpyReflectiveEquilibrium
)
from .set_implementation import GlobalSetBasedReflectiveEquilibrium

from .model_variations import StandardGlobalReflectiveEquilibriumLinearG, StandardLocalReflectiveEquilibriumLinearG
# from .util import (
#     re_weight_variations,
#     re_from_text_file,
#     varied_alphas,
#     standard_model_params_varied_alphas,
#     local_re_model_params_varied_alphas
# )


# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "ReflectiveEquilibrium",
    "StandardReflectiveEquilibrium",
    "LocalReflectiveEquilibrium",
    "GlobalReflectiveEquilibrium",
    "REContainer",
    "REState",
    "StandardGlobalReflectiveEquilibrium",
    "StandardLocalReflectiveEquilibrium",
    "FullBranchREContainer",
    "SimpleMultiAgentREContainer",
    "AbstractEnsembleGenerator",
    "EnsembleGenerator",
    "SimpleEnsembleGenerator",
    "GlobalREEnsembleGenerator",
    "LocalREEnsembleGenerator",
    "MultiAgentEnsemblesGenerator",
    "SimpleMultiAgentEnsemblesGenerator",
    "GlobalBitarrayReflectiveEquilibrium",
    "NumpyReflectiveEquilibrium",
    "GlobalNumpyReflectiveEquilibrium",
    "LocalNumpyReflectiveEquilibrium",
    "GlobalSetBasedReflectiveEquilibrium",
    # "re_weight_variations",
    # "re_from_text_file",
    # "varied_alphas",
    # "standard_model_params_varied_alphas",
    # "local_re_model_params_varied_alphas"
    "StandardGlobalReflectiveEquilibriumLinearG",
    "StandardLocalReflectiveEquilibriumLinearG"
]

# Configure logging
with pkg_resources.path(config, "logging-config.json") as path:
    with open(path) as config_file:
        config_dict = json.load(config_file)
        logging.config.dictConfig(config_dict)
