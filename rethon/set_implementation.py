# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations

from .base import GlobalReflectiveEquilibrium
from tau import Position

from typing import List

class GlobalSetBasedReflectiveEquilibrium(GlobalReflectiveEquilibrium):

    def penalty(self, position1: Position, position2: Position, sentence:int, penalties: List[float]) -> float:
        # contradiction
        if {sentence, -sentence}.issubset(position1.as_set().union(position2.as_set())):
            return penalties[3]
        # pos1 extends pos2
        elif len({sentence, -sentence}.intersection(position1.as_set())) != 0 and \
                len({sentence, -sentence}.intersection(position2.as_set())) == 0:
            return penalties[2]
        # pos2 extends pos1
        elif len({sentence, -sentence}.intersection(position1.as_set())) == 0 and \
                len({sentence, -sentence}.intersection(position2.as_set())) != 0:
            return penalties[1]
        # agreement
        else:
            return penalties[0]