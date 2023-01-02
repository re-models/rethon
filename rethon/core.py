# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations

from tau import Position
from .base import ReflectiveEquilibrium, REContainer, REState
from .numpy_implementation import GlobalNumpyReflectiveEquilibrium, LocalNumpyReflectiveEquilibrium

from typing import List, Iterator

from copy import copy


import logging

logging.basicConfig(filename='re_process.log', level=logging.INFO)
# logging.basicConfig(filename='re_process.log', level=logging.ERROR)


class StandardGlobalReflectiveEquilibrium(GlobalNumpyReflectiveEquilibrium):
    """
    Class that simply tags :py:class:`GlobalNumpyReflectiveEquilibrium` as the default implementation of
    :py:class:`GlobalReflectiveEquilibrium`.
    """
    pass

class StandardLocalReflectiveEquilibrium(LocalNumpyReflectiveEquilibrium):
    """
    Class that simply tags :py:class:`LocalNumpyReflectiveEquilibrium` as the default implementation of
    :py:class:`LocalReflectiveEquilibrium`
    """
    pass

class FullBranchREContainer(REContainer):
    """An REContainer generating all branches of model runs.

    This container will generate all branches of the given model (i.e. branches that occur when the choice of
    next theories and/or commitments is underdetermined by :py:func:`ReflectiveEquilibirium.theory_candidates`
    and/or  :py:func:`ReflectiveEquilibirium.commitments_candidates`).

    Attributes:
        max_re_length: The maximum allowed amount of steps for each individual branch (to avoid infinite loops for
            non-converging processes).
        max_branches: The maximum allow amount of branches (to avoid indefinite generation of branches).
    """
    def __init__(self, max_re_length: int = 50, max_branches: int = 50):
        self._max_re_length = max_re_length
        self._max_branches = max_branches

    def result_states(self, re_model: ReflectiveEquilibrium) -> List[REState]:
        return [re.state() for re in self.re_processes([re_model])]

    def re_processes(self, re_models: List[ReflectiveEquilibrium] = None) -> Iterator[ReflectiveEquilibrium]:
        if re_models:
            re = re_models[0]

            if re.dialectical_structure() == None:
                raise AttributeError("Before running an RE-process a dialectical structure must be set.")
            if re.state().initial_commitments() == None:
                raise AttributeError("Before running an RE-process initial commitments must be set.")
            # if the process already finished reset to initial state
            if re.state().finished:
                re.set_initial_state(re.state().initial_commitments())
            # if necessary update internal attributes
            re.update()

            self.active_branch_states = [re.state()]
            branch_counter = 0

            while self.active_branch_states:
                # we are reusing the same re instance and update the internal state
                # Idea: we want to avoid unnessary internal updates of re instances. Instead we create an additional
                # re instance later (to return an iterator with DIFFERENT re instances), which will avoid unnessary
                # updates (depending on when the implementation calls an update).
                re.set_state(self.active_branch_states.pop(0))
                branch_counter += 1
                step_counter = 0

                while (not re.state().finished):
                    step_counter += 1
                    if step_counter > self._max_re_length:
                        raise RuntimeWarning("Reached max loop count for re_process without finding a fixed point."
                                             f"Current state is: {re.state().as_dict()}")
                    if branch_counter > self._max_branches:
                        raise RuntimeWarning("Reached max amount of branches."
                                             f"Current state is: {re.state().as_dict()}")

                    re.next_step()
                    for alternative in re.state().last_alternatives():
                        alternatives_branch = copy(re.state().last_alternatives())
                        alternatives_branch.add(re.state().last_step())
                        alternatives_branch.remove(alternative)
                        additional_branch_state = re.state().past(-1)
                        additional_branch_state.add_step(alternative, alternatives_branch)
                        self.active_branch_states.append(additional_branch_state)

                yield copy(re)



