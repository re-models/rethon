from .base import GlobalReflectiveEquilibrium
from tau import Position

from typing import List

import logging
logging.basicConfig(filename='bitarray_implementation.log', level=logging.INFO)


class GlobalBitarrayReflectiveEquilibrium(GlobalReflectiveEquilibrium):

    # not needed, since penalties are summed up in Hamming distance (see below)
    def penalty(self, pos1: Position, pos2: Position, sentence: int, penalties: List[float]) -> float:

        # adjust index
        i = sentence - 1

        # contradiction
        if all(pos1.as_bitarray()[2 * i:2 * i + 2] | pos2.as_bitarray()[2 * i:2 * i + 2]):
            return penalties[3]

        # pos1 extends pos2
        elif any(pos1.as_bitarray()[2 * i:2 * i + 2]) and all(~(pos2.as_bitarray()[2 * i:2 * i + 2])):
            return penalties[2]

        # pos2 extends pos1
        elif any(pos2.as_bitarray()[2 * i:2 * i + 2]) and all(~(pos1.as_bitarray()[2 * i:2 * i + 2])):
            return penalties[1]

        # agreement
        else:
            return penalties[0]

    # overwrite Hamming distance in core.py
    def hamming_distance(self, position1: Position, position2: Position, penalties: List[float]) -> float:
        d = 0
        p1 = position1.as_bitarray()
        p2 = position2.as_bitarray()

        pos1_or_pos2 = p1 | p2
        pos1_xor_pos2 = ~(p1 ^ p2)
        pos1_ext_pos2 = p1 & ~p2

        # todo: Unresolved attribute reference 'n' for class 'DialecticalStructure'
        for i in range(self.dialectical_structure().n):
            # contradiction
            if all(pos1_or_pos2[2*i:2*i+2]):
                d += penalties[3]
            # agreement
            elif all(pos1_xor_pos2[2*i:2*i+2]):
                d += penalties[0]
            # pos1 extends pos2
            elif any(pos1_ext_pos2[2*i:2*i+2]):
                d += penalties[2]
            # pos2 extends pos1
            else:
                d += penalties[1]

        return d
