"""Implementing abstract base classes for RE on the basis of numpy."""

from __future__ import annotations

from .base import StandardReflectiveEquilibrium, GlobalReflectiveEquilibrium, LocalReflectiveEquilibrium
from .numba_re import numpy_hamming_distance2
from tau import Position, DialecticalStructure, NumpyPosition

import numpy as np
from typing import List


class NumpyReflectiveEquilibrium(StandardReflectiveEquilibrium):
    # not needed, since penalties are summed up in Hamming distance (see below)
    # ToDo: To discuss with @Andreas (either we provide an implementation or we remove the method from the interface).
    def penalty(self, pos1: Position, pos2: Position, sentence: int, penalties: List[float]) -> float:
        pass

    # overwrite Hamming distance in core.py

    def hamming_distance(self, position1: Position, position2: Position, penalties) -> float:

        if not isinstance(penalties, np.ndarray):
            penalties = np.array(penalties)
        return numpy_hamming_distance2(NumpyPosition.as_np_array(position1),
                                       NumpyPosition.as_np_array(position2),
                                       penalties)


class GlobalNumpyReflectiveEquilibrium(GlobalReflectiveEquilibrium, NumpyReflectiveEquilibrium):

    def __init__(self, dialectical_structure: DialecticalStructure = None, initial_commitments: Position = None,
                 model_name="GlobalNumpyReflectiveEquilibrium"):
        super().__init__(dialectical_structure, initial_commitments, model_name)


class LocalNumpyReflectiveEquilibrium(LocalReflectiveEquilibrium, NumpyReflectiveEquilibrium):
    """Numpy implementation of a locally searching RE process."""

    def first_theory(self) -> Position:
        """Choose an initial theory in the neighbourhood of the empty position."""

        empty_pos = NumpyPosition.from_set(set(), self.dialectical_structure().sentence_pool().size())

        neighbours = empty_pos.neighbours(self.model_parameters()["neighbourhood_depth"])

        max_achievement = 0
        initial_theories = {}

        for initial_theory_candidate in neighbours:

            if self.dialectical_structure().is_consistent(initial_theory_candidate):
                achievement = self.achievement(self.state().initial_commitments(),
					   initial_theory_candidate, self.state().initial_commitments())
                if achievement > max_achievement:
                    initial_theories = {initial_theory_candidate}
                    max_achievement = achievement
                elif achievement == max_achievement:
                    initial_theories.add(initial_theory_candidate)

        return self.pick_theory_candidate(initial_theories)