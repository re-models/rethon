import pytest
from random import randint
from typing import Set, List
import logging
import json
from os import getcwd, path, remove
import importlib
import tarfile

from theodias import DAGSetBasedDialecticalStructure, SetBasedPosition
from theodias.util import create_random_arguments, random_positions
from rethon.util import rethon_loads, rethon_dumps

from .core import FullBranchREContainer
from rethon import REState
from .numpy_implementation import LocalNumpyReflectiveEquilibrium
from .set_implementation import GlobalSetBasedReflectiveEquilibrium


# simply call `pytest -vv` on the console from directory of this file to execute the test
# or **`pytest -vv --log-cli-level INFO`** to show life logs (you simply use pytest logging mechanism, no need to
# configure a logger, see https://stackoverflow.com/questions/4673373/logging-within-pytest-tests)
# Use **`pytest tests.py -k 'position'`** to test only test cases in 'tests.py' (i.e. functions having
# the string 'test' in their name) that have (additionally) the string 'position' in their name.


model_implementations = [{'tau_module_name': 'theodias',
                          'position_class_name': 'StandardPosition',
                          'dialectical_structure_class_name': 'DAGDialecticalStructure',
                          're_module_name': 'rethon',
                          're_class_name': 'StandardGlobalReflectiveEquilibrium'
                          },
                         # {'tau_module_name': 'theodias',
                         #  'position_class_name':'StandardPosition',
                         #  'dialectical_structure_class_name': 'BDDDialecticalStructure',
                         #  're_module_name': 'rethon',
                         #   're_class_name': 'StandardLocalReflectiveEquilibrium'
                         #  },
                         {'tau_module_name': 'theodias',
                          'position_class_name': 'SetBasedPosition',
                          'dialectical_structure_class_name': 'DAGSetBasedDialecticalStructure',
                          're_module_name': 'rethon',
                          're_class_name': 'GlobalSetBasedReflectiveEquilibrium'
                          },
                         {'tau_module_name': 'theodias',
                          'position_class_name': 'NumpyPosition',
                          'dialectical_structure_class_name': 'DAGNumpyDialecticalStructure',
                          're_module_name': 'rethon',
                          're_class_name': 'GlobalNumpyReflectiveEquilibrium'
                          },
                         # {'tau_module_name': 'theodias',
                         #  'position_class_name': 'NumpyPosition',
                         #  'dialectical_structure_class_name': 'BDDNumpyDialecticalStructure',
                         #  're_module_name': 'rethon',
                         #  're_class_name': 'LocalNumpyReflectiveEquilibrium'
                         #  },
                        {'tau_module_name': 'theodias',
                         'position_class_name': 'BitarrayPosition',
                         'dialectical_structure_class_name': 'DAGBitarrayDialecticalStructure',
                         're_module_name': 'rethon',
                         're_class_name': 'GlobalBitarrayReflectiveEquilibrium'
                         }
                         ]


# helper functions

# this function will return a Position of the desired implementation
def get_position(pos: Set[int], n_unnegated_sentence_pool: int, impl):
    position_class_ = getattr(importlib.import_module(impl['tau_module_name']),
                              impl['position_class_name'])
    return position_class_.from_set(pos, n_unnegated_sentence_pool)

def get_dia(args: List[List[int]], n_unnegated_sentence_pool: int, impl):
    dia_class_ = getattr(importlib.import_module(impl['tau_module_name']),
                              impl['dialectical_structure_class_name'])
    return dia_class_.from_arguments(args, n_unnegated_sentence_pool)

def get_re(args: List[List[int]], n_unnegated_sentence_pool: int, impl):
    dia = get_dia(args, n_unnegated_sentence_pool, impl)
    reflective_equilibrium_class_ = getattr(importlib.import_module(impl['re_module_name']),
                                            impl['re_class_name'])
    return reflective_equilibrium_class_(dia)

def compare_re_states(s, t):
    t = list(t)   # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t

# the actual tests
class TestRemodel:
    #logging.basicConfig(filename='/home/sebastian/python_unit_testing.log',
    #                    filemode='a',
    #                    level=logging.INFO)

    log = logging.getLogger("RE unit testing ")

    def test_basic_re_model_parameters(self):
        # ToDo: check whether updating and setting dirty works as expected
        re = GlobalSetBasedReflectiveEquilibrium(DAGSetBasedDialecticalStructure.from_arguments([], 2))
        #print(re.default_model_parameters())
        assert GlobalSetBasedReflectiveEquilibrium.default_model_parameters() == re.model_parameters()
        assert re.model_parameter("weights") == {'account': 0.35, 'systematicity': 0.55, 'faithfulness': 0.1}
        re.set_dirty(False)
        assert re.is_dirty() == False
        # overwriting parameters
        re.set_model_parameters(account_penalties=[1, 1, 1, 1])
        assert re.model_parameter("account_penalties") == [1, 1, 1, 1]
        # should set re to dirty
        assert re.is_dirty()
        # adding parameters by key-value pairs
        re.set_model_parameters(new_parameter="new", newer_parameter=1)
        assert re.model_parameter("new_parameter") == "new"
        assert re.model_parameter("newer_parameter") == 1
        # adding parameters by dict
        re.set_model_parameters({'param1': 'param_value', 'param2': 2})
        assert re.model_parameter("param1") == "param_value"
        assert re.model_parameter("param2") == 2
        assert re.model_parameter_names() == {'account_penalties', 'faithfulness_penalties', 'new_parameter',
                                              'newer_parameter', 'param1', 'param2', 'weights'}
        re.model_parameters_set_to_default()
        assert re.default_model_parameters() == re.model_parameters()
        with pytest.raises(KeyError):
            re.model_parameter("unknown-parameter")

        re = LocalNumpyReflectiveEquilibrium()
        assert LocalNumpyReflectiveEquilibrium.default_model_parameters() == re.model_parameters()
        assert GlobalSetBasedReflectiveEquilibrium.default_model_parameters() != re.model_parameters()
        re.set_model_parameters(neighbourhood_depth = 2)
        assert LocalNumpyReflectiveEquilibrium.default_model_parameters() != re.model_parameters()
        re.model_parameters_set_to_default()
        assert LocalNumpyReflectiveEquilibrium.default_model_parameters() == re.model_parameters()

    def test_basic_re_state(self):
        # checking permissible instanciation
        with pytest.raises(ValueError):
            REState(finished=False, evolution=[SetBasedPosition({-1, 2}, 4)], alternatives=[set(),set()], time_line=[0])
        with pytest.raises(ValueError):
            REState(finished=False, evolution=[SetBasedPosition({-1, 2}, 4)], alternatives=[set()], time_line=[0,3])
        with pytest.raises(ValueError):
            REState(finished=False, evolution=[SetBasedPosition({-1, 2}, 4)], alternatives=[set(), set()], time_line=[0,3])
        # instantiating with non-permissible timelines
        with pytest.raises(ValueError):
            REState(finished=False, evolution=[SetBasedPosition({-1, 2}, 4), SetBasedPosition({-1, 2}, 4)],
                    alternatives=[set(), set()], time_line=[4, 3])
        with pytest.raises(ValueError):
            REState(finished=False, evolution=[SetBasedPosition({-1, 2}, 4), SetBasedPosition({-1, 2}, 4)],
                    alternatives=[set(), set()], time_line=[4, 4])

        # checking past_theory and past_coms for non-permissible values
        re_state = REState(False, [], [], [])
        with pytest.raises(ValueError):
            re_state.past_commitments()
        with pytest.raises(ValueError):
            re_state.past_commitments(past_step=-2, time=2)
        with pytest.raises(ValueError):
            re_state.past_theory()
        with pytest.raises(ValueError):
            re_state.past_theory(past_step=-2, time=2)


        re_state = REState(False, [], [], [])
        assert re_state.next_step_is_theory() == False
        assert re_state.last_commitments() == None
        assert re_state.past_commitments(past_step=-1) == None
        assert re_state.past_commitments(time=1) == None
        assert re_state.last_theory() == None
        assert re_state.initial_commitments() == None

        re_state = REState(finished=False,
                            evolution=[SetBasedPosition({-1, 2}, 4)],
                            alternatives=[set()],
                            time_line = [3])
        assert re_state.next_step_is_theory() == True
        assert re_state.last_commitments() == SetBasedPosition({-1, 2}, 4)
        assert re_state.initial_commitments() == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(past_step=0) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(past_step=-1) == None
        assert re_state.past_commitments(time = 3) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time = 5) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time = 1) == None


        assert re_state.last_theory() == None
        assert re_state.past_theory(0) == None

        assert re_state.past_theory(time=3) == None
        assert re_state.past_theory(time=0) == None
        assert re_state.past_theory(time=100) == None


        re_state = REState(finished=False,
                           evolution=[SetBasedPosition({-1, 2}, 4),
                                      SetBasedPosition({-1, 2, 3}, 4)],
                           alternatives=[set(),set()],
                            time_line = [0, 1])
        assert re_state.next_step_is_theory() == False
        assert re_state.last_commitments() == SetBasedPosition({-1, 2}, 4)

        assert re_state.last_theory() == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.initial_commitments() == SetBasedPosition({-1, 2}, 4)

        assert re_state.past_theory(time=0) == None
        assert re_state.past_theory(time=1) == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(time=100) == SetBasedPosition({-1, 2, 3}, 4)


        re_state = REState(finished=True,
                           evolution=[SetBasedPosition({-1, 2}, 4), # init coms (time = 0)
                                      SetBasedPosition({-1, 2, 3}, 4), # first theory (time = 1)
                                      SetBasedPosition({-1, 2, 3, 4}, 4)], # second coms (time = 2)
                           alternatives=[[],[],[]],
                           time_line = [0,1,2])
        assert re_state.next_step_is_theory() == True
        assert re_state.last_commitments() == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(0) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(-1) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(-2) is None

        assert re_state.past_commitments(time=2) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=100) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=1) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=0) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=-1) == None

        assert re_state.last_theory() == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(time=0) == None
        assert re_state.past_theory(time=1) == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(time=2) == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(time=100) == SetBasedPosition({-1, 2, 3}, 4)


        assert re_state.initial_commitments() == SetBasedPosition({-1, 2}, 4)
        assert len(re_state) == 3

        assert re_state.past(0) == re_state
        #logging.info(re_state.past(-1).as_dict())
        assert re_state.past(-1) == REState(finished=False,
                                            evolution=[SetBasedPosition({-1, 2}, 4),
                                            SetBasedPosition({-1, 2, 3}, 4)],
                                            alternatives=[[],[]],
                                            time_line = [0,1])
        assert re_state.past(-2) == REState(finished=False,
                                            evolution=[SetBasedPosition({-1, 2}, 4)],
                                            alternatives=[[]],
                                            time_line = [0])
        with pytest.raises(ValueError):
            re_state.past(-3)
        with pytest.raises(ValueError):
            re_state.past(1)

        re_state = REState(finished=True,
                           evolution=[SetBasedPosition({-1, 2}, 4), # init coms (time = 0)
                                      SetBasedPosition({-1, 2, 3}, 4), # th (time = 1)
                                      SetBasedPosition({-1, 2, 3, 4}, 4), # com (time = 3)
                                      SetBasedPosition({-1, 2, 3, 4, 5}, 4), # th (time = 5)
                                      SetBasedPosition({-1, 2, 3, 4, 5, 6}, 4) #com (time = 7)
                                      ],
                           alternatives=[set(), set(), set(),set(),set()],
                           time_line = [0,1,3,5,7])

        assert re_state.last_commitments() == SetBasedPosition({-1, 2, 3, 4, 5, 6}, 4)
        assert re_state.past_commitments(0) == SetBasedPosition({-1, 2, 3, 4, 5, 6}, 4)
        assert re_state.past_commitments(-1) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(-2) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(-3) == None
        assert re_state.last_theory() == SetBasedPosition({-1, 2, 3, 4, 5}, 4)
        assert re_state.past_theory(0) == SetBasedPosition({-1, 2, 3, 4, 5}, 4)
        assert re_state.past_theory(-1) == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(-2) == None
        # testing with time parameter

        assert re_state.past_commitments(time=0) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=1) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=2) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=3) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=-1) == None

        re_state = REState(finished=True,
                           evolution=[SetBasedPosition({-1, 2}, 4),  # init coms (time = 0)
                                      SetBasedPosition({-1, 2, 3}, 4),  # th (time = 1)
                                      SetBasedPosition({-1, 2, 3, 4}, 4),  # com (time = 3)
                                      SetBasedPosition({-1, 2, 3, 4, 5}, 4),  # th (time = 5)
                                      ],
                           alternatives=[set(), set(), set(), set()],
                           time_line=[0, 1, 3, 5])

        assert re_state.last_commitments() == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(0) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(-1) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(-2) == None
        assert re_state.past_commitments(-3) == None
        assert re_state.last_theory() == SetBasedPosition({-1, 2, 3, 4, 5}, 4)
        assert re_state.past_theory(0) == SetBasedPosition({-1, 2, 3, 4, 5}, 4)
        assert re_state.past_theory(-1) == SetBasedPosition({-1, 2, 3}, 4)
        assert re_state.past_theory(-2) == None
        # testing with time parameter

        assert re_state.past_commitments(time=0) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=1) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=2) == SetBasedPosition({-1, 2}, 4)
        assert re_state.past_commitments(time=3) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=4) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=5) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=6) == SetBasedPosition({-1, 2, 3, 4}, 4)
        assert re_state.past_commitments(time=100) == SetBasedPosition({-1, 2, 3, 4}, 4)

        assert re_state.past_commitments(time=-1) == None

        # testing behaviour of adding step or initializing states with impermissible times
        re_state = REState(finished=False,
                            evolution=[SetBasedPosition({-1, 2}, 4)],
                            alternatives=[set()],
                            time_line = [4])

        with pytest.raises(ValueError):
            re_state.add_step(SetBasedPosition({-1, 2}, 4), [set()], 4)
        with pytest.raises(ValueError):
            re_state.add_step(SetBasedPosition({-1, 2}, 4), [set()], 3)


        # testing past_x
        re_state = REState(finished=True,
                           evolution=[SetBasedPosition({1}, 7), # com
                                      SetBasedPosition({2}, 7),
                                      SetBasedPosition({3}, 7), # com
                                      SetBasedPosition({4}, 7),
                                      SetBasedPosition({5}, 7), # com
                                      SetBasedPosition({6}, 7),
                                      SetBasedPosition({7}, 7)], # com
                           alternatives=[[], [], [], [], [], [], []],
                           time_line=[2, 3, 5, 6, 7, 10, 13])

        assert re_state.past_step(0) == None
        assert re_state.past_step(1) == None
        assert re_state.past_step(2) == SetBasedPosition({1}, 7)
        assert re_state.past_step(4) == SetBasedPosition({2}, 7)
        assert re_state.past_step(5) == SetBasedPosition({3}, 7)
        assert re_state.past_step(9) == SetBasedPosition({5}, 7)
        assert re_state.past_step(13) == SetBasedPosition({7}, 7)
        assert re_state.past_step(14) == SetBasedPosition({7}, 7)
        assert re_state.past_step(100) == SetBasedPosition({7}, 7)

        assert re_state.past_commitments(time=1) ==  None
        assert re_state.past_commitments(time=0) ==  None
        assert re_state.past_commitments(time=2) == SetBasedPosition({1}, 7)
        assert re_state.past_commitments(time=3) == SetBasedPosition({1}, 7)
        assert re_state.past_commitments(time=5) == SetBasedPosition({3}, 7)
        assert re_state.past_commitments(time=6) == SetBasedPosition({3}, 7)
        assert re_state.past_commitments(time=7) == SetBasedPosition({5}, 7)
        assert re_state.past_commitments(time=9) == SetBasedPosition({5}, 7)
        assert re_state.past_commitments(time=10) == SetBasedPosition({5}, 7)
        assert re_state.past_commitments(time=13) == SetBasedPosition({7}, 7)
        assert re_state.past_commitments(time=100) == SetBasedPosition({7}, 7)

        assert re_state.past_theory(time=1) ==  None
        assert re_state.past_theory(time=2) ==  None
        assert re_state.past_theory(time=3) ==  SetBasedPosition({2}, 7)
        assert re_state.past_theory(time=5) ==  SetBasedPosition({2}, 7)

        assert re_state.past_theory(time=10) ==  SetBasedPosition({6}, 7)
        assert re_state.past_theory(time=12) ==  SetBasedPosition({6}, 7)
        assert re_state.past_theory(time=13) ==  SetBasedPosition({6}, 7)
        assert re_state.past_theory(time=100) ==  SetBasedPosition({6}, 7)

    def test_basic_re_hamming_distance(self):
        for re_impl in model_implementations:
            self.log.info(f"Testing reflective equilibrum class of type: {re_impl['re_class_name']}")

            # DIALECTICAL STRUCTURES
            # hamming distance of the standard model is independent of dia and sentence pool
            n = 7
            re = get_re([], n , re_impl)

            for impl in model_implementations:

                assert (re.hamming_distance(get_position({1, 2, -1}, n, impl),
                                             get_position({-1}, n, impl), [0, 0.3, 1, 1]) == 2)
                assert (re.hamming_distance(get_position({1, 2}, n, impl),
                                             get_position({-1}, n, impl), [0, 0.3, 1, 1]) == 2)
                assert (re.hamming_distance(get_position({1}, n, impl),
                                             get_position({-1}, n, impl), [0, 0.3, 1, 1]) == 1)
                assert (re.hamming_distance(get_position({1}, n, impl),
                                             get_position(set(), n, impl), [0, 0.3, 1, 1]) == 1)
                assert (re.hamming_distance(get_position(set(), n, impl),
                                             get_position(set(), n, impl),[0, 0.3, 1, 1]) == 0)
                # tolerate very small differences due to floating point rounding errors
                assert (abs(re.hamming_distance(get_position({1, 2, 4}, n, impl),
                                             get_position({3}, n, impl), [0, 0.3, 1, 1]) - 3.3) < 0.0000001)
                assert (abs(re.hamming_distance(get_position(set(), n, impl),
                                             get_position({3}, n, impl), [0, 0.3, 1, 1]) - 0.3) < 0.0000001)

                assert (re.hamming_distance(get_position({1, 2, 5}, n, impl), get_position({5}, n, impl),
                                                     [1, 2, 3, 4]) == 11)
                assert (re.hamming_distance(get_position({3, 4, 5, 6, 7}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 18)
                assert (re.hamming_distance(get_position({3, 4, 5, 6, 7}, n, impl), get_position({3}, n, impl),
                                                     [1, 2, 3, 4]) == 15)
                assert (re.hamming_distance(get_position({3, 4, 5}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 14)
                assert (re.hamming_distance(get_position({2, 3, 4, 5}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({1, 2}, n, impl), get_position({1}, n, impl), [1, 2, 3, 4]) == 9)
                assert (re.hamming_distance(get_position({1, 2}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 10)
                assert (re.hamming_distance(get_position({1, 2}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 10)
                assert (re.hamming_distance(get_position({1, 2, 3}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 11)
                assert (re.hamming_distance(get_position({1, 2, 3}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 12)

                assert (re.hamming_distance(get_position({1, 2, 3}, n, impl), get_position({2, 3}, n, impl),
                                                     [1, 2, 3, 4]) == 9)
                assert (re.hamming_distance(get_position({1, 2, 3, 4}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({1, 2, 3, 4}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({1, 2, 3, 4}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 14)
                assert (re.hamming_distance(get_position({1, 2, 3, 4}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 14)
                assert (re.hamming_distance(get_position({1, 2, 3, 4}, n, impl), get_position({2, 3}, n, impl),
                                                     [1, 2, 3, 4]) == 11)
                assert (re.hamming_distance(get_position({4, 5}, n, impl), get_position({1}, n, impl), [1, 2, 3, 4]) == 12)
                assert (re.hamming_distance(get_position({4, 5}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 10)
                assert (re.hamming_distance(get_position({4, 5}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({4, 5, 6}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 14)

                assert (re.hamming_distance(get_position({4, 5, 6}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 12)
                assert (re.hamming_distance(get_position({4, 5, 6}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 15)
                assert (re.hamming_distance(get_position({4, 5, 6, 7}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 16)
                assert (re.hamming_distance(get_position({4, 5, 6, 7}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 14)
                assert (re.hamming_distance(get_position({4, 5, 6, 7}, n, impl), get_position({1, 3}, n, impl),
                                                     [1, 2, 3, 4]) == 17)
                assert (re.hamming_distance(get_position({1, 5}, n, impl), get_position({1}, n, impl), [1, 2, 3, 4]) == 9)
                assert (re.hamming_distance(get_position({1, 5}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 7)
                assert (re.hamming_distance(get_position({1, 5}, n, impl), get_position({2, 3}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({1, 5, 6}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 11)
                assert (re.hamming_distance(get_position({1, 5, 6}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 14)

                assert (re.hamming_distance(get_position({1, 5, 6}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 9)
                assert (re.hamming_distance(get_position({1, 5, 6}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 12)
                assert (re.hamming_distance(get_position({6, 7}, n, impl), get_position({1}, n, impl), [1, 2, 3, 4]) == 12)
                assert (re.hamming_distance(get_position({6, 7}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 13)
                assert (re.hamming_distance(get_position({6, 7}, n, impl), get_position({1, 7}, n, impl),
                                                     [1, 2, 3, 4]) == 10)
                assert (re.hamming_distance(get_position({1, 3, 6}, n, impl), get_position({1}, n, impl),
                                                     [1, 2, 3, 4]) == 11)
                assert (re.hamming_distance(get_position({1, 3, 6}, n, impl), get_position({2}, n, impl),
                                                     [1, 2, 3, 4]) == 14)
                assert (re.hamming_distance(get_position({1, 3, 6}, n, impl), get_position({1, 5}, n, impl),
                                                     [1, 2, 3, 4]) == 12)
                assert (re.hamming_distance(get_position({1, 3, 6}, n, impl), get_position({2, 3}, n, impl),
                                                     [1, 2, 3, 4]) == 12)


    # todo: re-state give None until re_process was called

    def test_basic_re_process_standard_example(self):
        for re_impl in model_implementations:
            self.log.info(f"Testing reflective equilibrium class of type: {re_impl['re_class_name']}")

            # DIALECTICAL STRUCTURES
            # hamming distance of the standard model is independent of dia and sentence pool
            n = 7
            # The example from the paper
            args = [[1, 3],
                    [1, 4],
                    [1, 5],
                    [1, -6],
                    [2, -4],
                    [2, 5],
                    [2, 6],
                    [2, 7]]
            re = get_re(args, n , re_impl)

            for impl in model_implementations:
                # skipping LocalNumpyReflectiveEquilibrium
                if re_impl['re_class_name'] != 'LocalNumpyReflectiveEquilibrium' and \
                        re_impl['re_class_name'] != 'StandardLocalReflectiveEquilibrium':
                    # case A
                    re.set_initial_state(get_position({3, 4, 5}, n, impl))
                    re.re_process()
                    logging.info(f"State: {re.state().as_dict()}")
                    assert (re.state().last_commitments() == SetBasedPosition({1, 3, 4, 5, -6, -2}, n))
                    assert (re.state().last_theory() == SetBasedPosition({1}, n))

                    # case B

                    re.set_initial_state(get_position({2, 3, 4, 5}, n, impl))
                    re.re_process()
                    assert (re.state().last_commitments() == SetBasedPosition({1, 3, 4, 5, -2, -6}, n))
                    assert (re.state().last_theory() == SetBasedPosition({1}, n))

                    # case C
                    re.set_initial_state(get_position({3, 4, 5, 6, 7}, n, impl))
                    re.re_process()
                    assert (re.state().last_commitments() == SetBasedPosition({1, 3, 4, 5, -2, -6}, n) or
                            re.state().last_commitments() == SetBasedPosition({2, 5, 6, 7, -1, -4}, n))
                    assert (re.state().last_theory() == SetBasedPosition({1}, n) or
                            re.state().last_theory() == SetBasedPosition({2}, n))

                    # case D
                    re.set_initial_state(get_position({3, 4, 5, -6, 7}, n, impl))
                    re.re_process()
                    assert (re.state().last_commitments() == SetBasedPosition({1, 3, 4, 5, -2, -6}, n))
                    assert (re.state().last_theory() == SetBasedPosition({1}, n))


    # testing all given implementations against a produced datasets (the datasets were produced with
    # the Bitarray implementation
    def test_re_process_consistency(self):
        # Assumptions for the test:
        # (i)  The test data has the json-form:
        #      [{"full-branch":[model_run_11, model_run_12, ...]}, {"full-branch":[model_run_21, model_run_22, ...], ... ]
        # (ii) Model runs in one full branch represent all posible re-processes of one specific model, with a specific set of
        #      model parameter and one specific set of initial commitments.
        # working dir of notebook
        # ToDo: That does only work if we start pytest from the package dir.
        #  We should, rather, declare the data files as package resources!
        current_dir = getcwd()
        data_dir = path.join(current_dir, 'test_data')
        data_dir = path.abspath(data_dir)
        #print(f'DATA DIR: {data_dir}')
        # The following data sets contain fullbranch model-runs with the standard model and standard parameters and
        # differ between number different dialectical structures and number of different initial conditions.
        # Remark: Failing these consistency checks for different implementations of the standard model does not
        # necessarily mean that there is some problem. Different implementations might result in slightly different
        # achievement values, which, in turn, might end in a different evolution of states.
        tar_files = [# 500 full-branch runs with 50 randomly generated initial positions and
                     # using standard parameters and
                     # 100 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     #path.join(data_dir, 'test_data_re_01.json.tar.gz'),
                     # **********************************************
                     # 100 full-branch runs with 10 randomly generated initial positions and
                     # using standard parameters and
                     # 10 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     path.join(data_dir, 'test_data_re_02.json.tar.gz'),
                     # **********************************************
                     # 4 full-branch runs with 2 randomly generated initial positions and
                     # using standard parameters and
                     # 2 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     path.join(data_dir, 'test_data_re_03.json.tar.gz'),
                     # **********************************************
                     # 24 full-branch runs with 2 randomly generated initial positions and
                     # 2 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     # and with a variation of alpha values with extreme values:
                     # ([[0.0, 0.0, 1.0],
                     #  [0.0, 0.5, 0.5],
                     #  [0.0, 1.0, 0.0],
                     #  [0.5, 0.0, 0.5],
                     #  [0.5, 0.5, 0.0],
                     #  [1.0, 0.0, 0.0]])
                     # Attention: extreme long runs (and extreme big data file)
                     #path.join(data_dir, 'test_data_re_04.json.tar.gz'),
                     # **********************************************
                     # 12 full-branch runs with 2 randomly generated initial positions and
                     # 2 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     # and with a variation of alpha values without extreme values:
                     # [[0.25, 0.25, 0.5], [0.25, 0.5, 0.25], [0.5, 0.25, 0.25]]
                     path.join(data_dir, 'test_data_re_05.json.tar.gz'),
                     # **********************************************
                     # ToDo: (to discuss with @Andreas)
                     # We have with this data set a consistency problem. [BC]: Do not know so far
                     # whether this is a problem or due to rounding errors (see remark above). The problem is
                     # that the extreme values cause major underdetermination of states (=lots of branches).
                     # 6 full-branch runs with 1 randomly generated initial positions and
                     # 1 randomly generated dialectical structures with the following parameters:
                     #    - sentence pool of 6,
                     #    - number of arguments between 3 and 8
                     #    - number of premises between 1 and 2
                     # and with a variation of alpha values with extreme values:
                     # ([[0.0, 0.0, 1.0],
                     #  [0.0, 0.5, 0.5],
                     #  [0.0, 1.0, 0.0],
                     #  [0.5, 0.0, 0.5],
                     #  [0.5, 0.5, 0.0],
                     #  [1.0, 0.0, 0.0]])
                     # Attention: extreme long runs (and big data file)
                     #path.join(data_dir, 'test_data_re_06.json.tar.gz'),
                     # **********************************************
                    ]

        for tar_file in tar_files:
            # load all test data as python dict
            with tarfile.open(tar_file) as tar:
                for tarinfo in tar:
                    json_name = tarinfo.name
                tar.extractall(data_dir)
            with open(path.join(data_dir, json_name), "r") as json_file:
                ensemble_list_dict = json.load(json_file)
            remove(path.join(data_dir, json_name))
            #with open(data_file, "r") as json_file:
            #    ensemble_list_dict = json.load(json_file)
            self.log.info(f"Testing re-process consistency with {len(ensemble_list_dict)} full-branch "
                          f"model runs from {json_name}.")
            self.log.info(f"Testing the following implementations: "
                          f"{[impl['re_class_name'] for impl in model_implementations]}")
            re_container = FullBranchREContainer()
            for full_branch in ensemble_list_dict:
                branches = full_branch['full_branch']

                if (len(branches) > 0 ):
                    # loading all model results of the branches as REState for later comparison
                    result_states = [rethon_loads(json.dumps(branch['state'])) for branch in branches]
                    # reproducing all branches for all implementation
                    # by picking the first element to determine dia structure and model parameters
                    for impl in model_implementations:
                        #logging.info(f"Testing implementation {impl['re_class_name']}")
                        # instantiating the dialectical structure
                        skip_test = False
                        dialectical_structure_class_ = getattr(importlib.import_module(impl['tau_module_name']),
                                                               impl['dialectical_structure_class_name'])
                        ds = dialectical_structure_class_.from_arguments(branches[0]['dialectical_structure']['arguments'],
                                                                         branches[0]['dialectical_structure'][
                                                                             'n_unnegated_sentence_pool'])
                        # instantiating the RE class
                        reflective_equilibrium_class_ = getattr(importlib.import_module(impl['re_module_name']),
                                                                impl['re_class_name'])
                        re = reflective_equilibrium_class_(ds)
                        re.set_model_parameters(branches[0]['model_parameters'])
                        # ToDo (@Basti): will be later set by `set_model_parameters`
                        # LocalNumpyRE should have the same results if we set neighbourdepth to sentencepool

                        # initial position
                        position_class_ = getattr(importlib.import_module(impl['tau_module_name']),
                                                  impl['position_class_name'])
                        position = position_class_.from_set(branches[0]['state']['evolution'][0]['position'],
                                                            branches[0]['state']['evolution'][0][
                                                                'n_unnegated_sentence_pool'])
                        # running the model

                        if impl['re_class_name'] == 'LocalNumpyReflectiveEquilibrium' or \
                           impl['re_class_name'] == 'StandardLocalReflectiveEquilibrium':
                            #re.set_neighbourhood_depth(ds.sentence_pool().sentence_pool())
                            re.set_model_parameters(neighbourhood_depth = ds.sentence_pool().size())
                            # We are setting not only the initial position but also the first theory.
                            # Only then, we expect the same results.
                            sentence_pool = branches[0]['state']['evolution'][0]['n_unnegated_sentence_pool']

                            first_theory = position_class_.from_set(branches[0]['state']['evolution'][1]['position'],
                                                                    sentence_pool)
                            #alternatives2 = {position_class_.from_set(pos['position'], sentence_pool) for
                            #                 pos in branches[0]['state']['alternatives'][1]}
                            # check if the first theory is already underdetermined (if yes we do not use for the
                            # unit-test, since we cannot branch.
                            if len(branches[0]['state']['alternatives'][1]) != 0:
                                skip_test = True
                                logging.info("Skipping unit test for the following result state:")
                                for state in result_states:
                                    logging.info(state.as_dict())
                            else:
                                re.set_state(REState(finished=False,
                                                     evolution=[position, first_theory],
                                                     alternatives=[set(), set()],
                                                     time_line = [0, 1]))
                                logging.info(f"Setted state to {re.state()}")
                                #reproduced_states = list(re.re_processes(track_branching=True))
                                reproduced_states = re_container.result_states(re)
                        else:
                            re.set_initial_state(position)
                            # self.log.info('***** Model ***********************************')
                            # self.log.info(re_dumps(re))
                            #reproduced_states = list(re.re_processes(track_branching=True))
                            reproduced_states = re_container.result_states(re)

                        if not skip_test:
                            equal_state = compare_re_states(result_states, reproduced_states)
                            if not equal_state:
                                self.log.error("Testing re processes. Reproduced RE states do not match the test data.")
                                self.log.error('***** Model ***********************************')
                                self.log.error(rethon_dumps(re))
                                self.log.error('***** Result states from the test data: *******')
                                for state in result_states:
                                    self.log.error(state.as_dict())
                                self.log.error('***** Reproduced state ************************')
                                for state in reproduced_states:
                                    self.log.error(state.as_dict())
                            assert equal_state

    def test_global_optima(self):

        class BruteForceGlobalOptimaRE(GlobalSetBasedReflectiveEquilibrium):

            def global_optima(self, initial_commitments):
                global_optima = set()
                for theory in self.dialectical_structure().consistent_positions():
                    if self.dialectical_structure().closure(theory).size() != 0:
                        for commitments in self.dialectical_structure().minimally_consistent_positions():
                            achievement = self.achievement(commitments, theory, initial_commitments)
                            if len(global_optima) == 0:
                                global_optima.add((theory, commitments))
                                achievement_best = achievement
                            elif achievement_best == achievement:
                                global_optima.add((theory, commitments))
                            elif achievement_best < achievement:
                                global_optima = {(theory, commitments)}
                                achievement_best = achievement
                return global_optima

        sample_size = 10
        for a in range(sample_size):
            # here we just use small dia-structures
            n = randint(3, 5)
            n_arguments = randint(2, 4)
            # random small ds
            args = create_random_arguments(n_sentences=n, n_arguments=n_arguments, n_max_premises=1)
            if args:
                # skipping LocalNumpyReflectiveEquilibrium (has no method global_optima)
                res_to_compare = [get_re(args, n, impl) for impl in model_implementations if
                                  impl['re_class_name'] != 'LocalNumpyReflectiveEquilibrium' and \
                                  impl['re_class_name'] != 'StandardLocalReflectiveEquilibrium']

                # serves as goldstandard
                ds = DAGSetBasedDialecticalStructure(n, args)
                brute_force_re = BruteForceGlobalOptimaRE(ds)
                # random position
                for pos in random_positions(n, k=20, allow_empty_position=False):
                    global_optima_2 = brute_force_re.global_optima(SetBasedPosition(pos, n))
                    for re in res_to_compare:
                        global_optima_1 = re.global_optima(SetBasedPosition(pos, n))
                        assert( global_optima_1 == global_optima_2)
