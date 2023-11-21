"""
.. module:: base
    :synopsis: module defining basic abstract classes

"""

# see https://stackoverflow.com/questions/33533148
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from copy import copy
from math import trunc
from typing import List, Set, Tuple, Dict, Iterator, Any
from itertools import product
import random
import numpy as np

from tau import Position, DialecticalStructure



class ReflectiveEquilibrium(ABC):
    """Abstract class representing a process of finding epistemic states that are in a reflective quilibrium.

    The process of finding a reflective equilbrium starts with a set of initial commitments :math:`\\mathcal{C}_0`.
    In the next step a first theory :math:`\\mathcal{T}_0` is chosen. The process is repeated and represents a step-wise adjusted of
    theories and commitments until a final state is reached the represents the reflective equilibrium.

    .. math:: \\mathcal{C_0} \\rightarrow \\mathcal{T_0} \\rightarrow \\mathcal{C_1} \\rightarrow \\mathcal{T_1} \\rightarrow \\dots \\rightarrow \\mathcal{T_{final}} \\rightarrow \\mathcal{C_{final}}

    Each step is triggered by :py:func:`next_step` which will, depending on whether the next step is choosing a new
    theory or a set of new commitments, succeed in the following way:

    * Either :py:func:`commitment_candidates` or :py:func:`theory_candidates` will be called to determine a set of candidate commitments or theories.
    * :py:func:`pick_commitment_candidate` or :py:func:`pick_theory_candidate` will be called to decide on the next commitments/theory according to some additional criterion (which will be non-trivial if the former sets have more than one element).

    The function :py:func:`finished` is used to determine the final state of the process.

    Accordingly, subclasses must implement these functions to specify re processes.
    """

    def __init__(self,
                 dialectical_structure: DialecticalStructure = None,
                 initial_commitments: Position = None,
                 model_name: str = None):
        self.__dialectical_structure = dialectical_structure
        self.__dirty = True
        self.__model_parameter = {}
        self.__state = None
        self.id = None
        if initial_commitments:
            self.set_initial_state(initial_commitments)

        if model_name:
            self.__model_name = model_name
        else:
            self.__model_name = self.__class__.__name__
        self.model_parameters_set_to_default()

    def set_dialectical_structure(self, dialectical_structure: DialecticalStructure):
        """Set the dialectical structure on which the model is based."""
        self.__dialectical_structure = dialectical_structure
        self.set_dirty(True)

    def dialectical_structure(self) -> DialecticalStructure:
        """Return the dialectical structure on which the model is based."""
        return self.__dialectical_structure

    def set_id(self, id:int):
        self.id = id

    def get_id(self):
        return self.id

    @abstractmethod
    def theory_candidates(self, time: int = None, **kwargs) -> Set[Position]:
        """Theory candidates for choosing the next theory."""
        pass

    @abstractmethod
    def commitment_candidates(self, time: int = None, **kwargs) -> Set[Position]:
        """Commitment candidates for choosing next commitments."""
        pass

    @abstractmethod
    def pick_theory_candidate(self, theory_candidates: Set[Position], time: int = None, **kwargs) -> Position:
        """Determination of the next theory given the theory candidates."""
        pass

    @abstractmethod
    def pick_commitment_candidate(self, commitments_candidates: Set[Position], time: int = None, **kwargs) -> Position:
        """Determination of the next commitments given the commitments candidates."""
        pass

    @abstractmethod
    def finished(self, **kwargs) -> bool:
        """Criterion of when the RE process reached a final state."""
        pass

    # In this abstract class the model is set to dirty, if the initial state changes or the internal modelparameter
    # change
    def set_dirty(self, dirty: bool):
        """Should be used to indicate whether attributes are reset that demand updating other attributes of the model."""
        self.__dirty = dirty

    def is_dirty(self):
        """Checks whether the model demand an update of internal attributes."""
        return self.__dirty

    def update(self, **kwargs):
        """Subclasses can extend/override this method to update internal attributes of the model."""
        pass

    def set_initial_state(self, initial_commitments: Position):
        """Set the initial state of the model."""
        self.set_state(REState(False,
                               evolution=[initial_commitments],
                               alternatives=[set()],
                               time_line=[0]))

    def set_state(self, state: REState):
        """Setting the current state of the model.

        The instance is set to dirty only when the state is set the first time or the initial
        state differs.
        """
        if self.__state is None:
            self.__state = state
            self.set_dirty(True)
            return
        if state.initial_commitments() != self.state().initial_commitments():
            self.set_dirty(True)
        self.__state = state

    def state(self) -> REState:
        """Getting the current state of the model as :py:class:`REState`.
        """
        return self.__state

    def re_process(self, initial_commitments: Position = None,
                   max_steps: int = 50):
        """Process of finding a reflective equilibrium based on given initial commitments.

        Starting with the initial commitments :math:`\\mathcal{C}_0` as the \
        initial epistemic state, the epistemic state is successively revised \
        until the process is finished (as defined by :py:func:`finished`).

        :param initial_commitments: A position representing the initial \
        commitments at the outset of an RE process.

        :param max_steps: The number of steps (i.e. theory or commitments \
        adjustments) before the process is aborted, raising a \
        :code:`MaxLoopsWarning`. :code:`max_steps` defaults to 50.

        """
        if initial_commitments:
            self.set_initial_state(initial_commitments)
        if self.dialectical_structure() is None:
            raise AttributeError("Before running an RE-process a dialectical structure must be set.")
        if self.state().initial_commitments() is None:
            raise AttributeError("Before running an RE-process initial commitments must be set.")
        # if the process already finished reset to initial state
        if self.state().finished:
            self.set_initial_state(self.state().initial_commitments())

        # if necessary update internal attributes
        self.update()
        step_counter = 0

        while not self.state().finished:
            step_counter += 1
            if step_counter > max_steps:
                raise MaxLoopsWarning("Reached max loop count for re_process without finding a fixed point."
                                     f"Current state is: {self.state().as_dict()}")
            self.next_step()

    def next_step(self, time: int = None, **kwargs):
        """Triggers search for next commitments/theory."""
        if self.state().next_step_is_theory():
            canditates = self.theory_candidates(time=time, **kwargs)
            next_position = self.pick_theory_candidate(canditates, time=time, **kwargs)
        else:  # next step is to choose a commitment
            canditates = self.commitment_candidates(time=time, **kwargs)
            next_position = self.pick_commitment_candidate(canditates, time=time, **kwargs)
        if not canditates:
            raise RuntimeError(f"{self.__class__.__name__} did not yield theory or commitments "
                               f"candidates for the RE process.")
        canditates.remove(next_position)
        self.state().add_step(next_position, canditates, time)
        # ToDo: That should be redundant
        self.state().finished = self.finished(**kwargs)

    def model_name(self) -> str:
        """Model name.

        :return: A name of the model implemented by that class.
        """
        return self.__model_name

    def model_parameter_names(self):
        """Returns names (keys) of the model parameters."""
        return self.__model_parameter.keys()

    def model_parameter(self, name: str):
        """Returns model parameters by names."""
        if name not in self.__model_parameter.keys():
            raise KeyError(f"The model parameter {name} is currently not set.")
        return self.__model_parameter[name]

    def model_parameters(self) -> Dict:
        """Getting all model parameters as dict.
        """
        return self.__model_parameter

    def reset_model_parameters(self, parameters: Dict):
        """Resetting model parameters.

        Replaces the current model parameters with :code:`paramters`.
        """
        self.__model_parameter = parameters
        self.set_dirty(True)

    def set_model_parameters(self, parameters: Dict = None, **kwargs):
        """Setting model parameters either by a dictionary or key-value pairs.

        The given parameters will be added to the given parameters or updated respectively.
        """
        if parameters:
            for key in parameters.keys():
                if (key not in self.__model_parameter.keys()) or parameters[key] != self.__model_parameter[key]:
                    self.set_dirty(True)
                    self.__model_parameter[key] = parameters[key]

        for key in kwargs:
            if (key not in self.__model_parameter.keys()) or kwargs[key] != self.__model_parameter[key]:
                self.set_dirty(True)
                self.__model_parameter[key] = kwargs[key]

    @staticmethod
    @abstractmethod
    def default_model_parameters() -> Dict:
        """Implementing classes should use this method to define default model parameters."""
        pass

    def model_parameters_set_to_default(self):
        """Resets the model parameters to their default values."""
        self.__model_parameter = {}


class StandardReflectiveEquilibrium(ReflectiveEquilibrium):
    """ Abstract class that describes RE in terms of optimizing an achievement function.

    The class partially implements :py:class:`ReflectiveEquilibrium` and provides additional functions to
    calculate an achievement function :math:`Z` (:py:func:`achievement`) for each step that
    can be used by subclasses to propose theory and commitments candidates.

    This class already determines the behaviour in the case of an underdetermination of commitments and theory
    candidates: In this case a set of random commitments/ a random theory is chosen.

    """
    def __init__(self, dialectical_structure: DialecticalStructure = None,
                 initial_commitments: Position = None,
                 model_name: str = "StandardModel"):
        super().__init__(dialectical_structure, initial_commitments, model_name)

    def achievement(self, commitments: Position, theory: Position, initial_commitments: Position) -> float:
        """The achievement function :math:`Z`.

        The achievement function is a convex combination of account, systematicity and faitfulness:

        .. math:: Z(\\mathcal{C},\\mathcal{T} | \\mathcal{C}_0):= \\alpha_A A(\\mathcal{C}, \\mathcal{T})\
        + \\alpha_S S(\\mathcal{T}) + \\alpha_F F(\\mathcal{C}| \\mathcal{C}_0)

        The weighing factors :math:`\\alpha_A, \\alpha_S, \\alpha_F` are non-negative real numbers that add up to 1.

        :param commitments: The (current) commitments :math:`\\mathcal{C}`.
        :param theory: The (current) theory :math:`\\mathcal{T}`.
        :param initial_commitments: The initial commitments :math:`\\mathcal{C}_0`.
        :return:
        """

        return (self.model_parameter("weights")['account'] * self.account(commitments, theory)
                + self.model_parameter("weights")['systematicity'] * self.systematicity(theory)
                + self.model_parameter("weights")['faithfulness'] * self.faithfulness(commitments, initial_commitments))

    def account(self, commitments: Position, theory: Position) -> float:
        """Account of the theory w.r.t. the position.

         The account :math:`A` is a measure of how well the theory :math:`\\mathcal{T}` accounts for the given
         commitments :math:`\\mathcal{C}` and is defined by:

         .. math:: A(\\mathcal{C}, \\mathcal{T}):=\\left( 1- \\left(\\frac{D_{0,0.3,1,1}(\\mathcal{C}\
         ,\\overline{\\mathcal{T}})}{N}\\right)^2 \\right)

         with :math:`D` being the weighted (asymmetric) Hamming Distance (see :py:func:`hamming_distance`).


         :param commitments: The commitments :math:`\\mathcal{C}`.
         :param theory: The theory :math:`\\mathcal{T}`.
         :return:
         """
        return 1 - (self.hamming_distance(commitments, self.dialectical_structure().closure(theory),
                                          self.model_parameter("account_penalties")) /
                    self.dialectical_structure().sentence_pool().size()) ** 2

    def faithfulness(self, commitments: Position, initial_commitments: Position) -> float:
        """Faithfulness of the commitments w.r.t. the initial commitments.

         The faithfulness :math:`F` is a measure of how faithfull the commitments :math:`\\mathcal{C}`
         are to the initial commitments :math:`\\mathcal{C}_0` and is defined by:

         .. math:: A(\\mathcal{C} | \\mathcal{C}_0):=\\left( 1- \\left(\\frac{D_{0,0,1,1}(\\mathcal{C}_0\
         ,\\mathcal{C})}{N}\\right)^2 \\right)

         :param commitments: The commitments :math:`\\mathcal{C}`.
         :param initial_commitments: The initial commitments :math:`\\mathcal{C}_0`.
         :return:
        """
        # Normalisation by the size of the unnegated half of the sentence pool
        # (Beisbart, Betz, Brun 2021, p. 463ff)
        return 1 - (self.hamming_distance(initial_commitments, commitments, self.model_parameter("faithfulness_penalties"))
             / self.dialectical_structure().sentence_pool().size()) ** 2

    def systematicity(self, theory: Position) -> float:
        """Systematicity of the theory.

         The systematicity :math:`S` is a measure of the systematising power of a theory :math:`\\mathcal{T}`
         and is defined by:

         .. math:: S(\\mathcal{T}):=\\left( 1- \\left(\\frac{|\\mathcal{T}|-1}{|\\overline{\\mathcal{T}}|}\\right)^2 \\right)

         :param theory: The theory :math:`\\mathcal{T}`.
         :return:
         """
        try:
            return 1 - ((theory.size() - 1) /
                        self.dialectical_structure().closure(theory).size()) ** 2

        except ZeroDivisionError:  # theory's closure is empty
            return 0

    # Remark: for the current implementations the method could be static, but we better leave it that
    # way. Other implementations might want to define a hamming distance that depends on the internal
    # state of the re-process
    def hamming_distance(self, position1: Position, position2: Position, penalties: List[float]) -> float:
        """ The weighted Hamming distance.

        A weighted (asymmetric) Hamming Distance :math:`D` between two
        positions :math:`\\mathcal{A}` and :math:`\\mathcal{B}` which is defined by

        .. math:: D_{d_0,d_1,d_2,d_3}(\\mathcal{A}, \\mathcal{B}):= \\sum_{\\{s,\\neg s\\} \subset S} \
        d_{{d_0,d_1,d_2,d_3}}(\\mathcal{A}, \\mathcal{B}, \\{s,\\neg s\\})

        and based on the :py:func:`penalty`-function.

        :param position1:
        :param position2:
        :param penalties: A float-list of penalty values for the :py:func:`penalty`-function.
        :return:
        """
        d = 0


        n = self.dialectical_structure().sentence_pool().size()
        for i in range(1, n + 1):
            d += self.penalty(position1, position2, i, penalties)
        return d

    def penalty(self, position1: Position, position2: Position, sentence: int, penalties: List[float]) -> float:
        """ A penalty function.

        The penalty function calculates a penalty value for two sentences of two positions given a list of penalty-value
        and is defined by

        .. math:: d_{{d_0,d_1,d_2,d_3}}(\\mathcal{A}, \\mathcal{B}, \\{s,\\neg s\\}):= \
        \\begin{cases} d_3 \\text{ if } \\{s,\\neg s\\}\\subset(\\mathcal{A}\\cup \\mathcal{B}),\
        \\\ d_2 \\text{ if } \\{s,\\neg s\\}\cap \\mathcal{A} \\neq \\emptyset \\text{ and } \
        \\{s,\\neg s\\}\cap \\mathcal{B} = \\emptyset, \
        \\\ d_3 \\text{ if } \\{s,\\neg s\\}\cap \\mathcal{A} = \\emptyset \\text{ and } \
        \\{s,\\neg s\\}\cap \\mathcal{B} \\neq \\emptyset, \
        \\\ d_0 \\text{ otherwise.}\\end{cases}


        :param position1:
        :param position2:
        :param sentence: The index of the sentences with both positions.
        :param penalties: A float-list of penalty values.
        :return:
        """
        pass

    @staticmethod
    def default_model_parameters() -> Dict:
        """Default model parameters of the standard model.

        Default model parameter for the calculation of achievement and the determination of the vicinity of positions:

        .. code:: python

            {'weights': {'account': 0.35, 'systematicity': 0.55, 'faithfulness': 0.1},
             'account_penalties': list(np.array([0, 0.3, 1, 1], dtype=np.float32)),
             'faithfulness_penalties': list(np.array([0, 0, 1, 1], dtype=np.float32))
             }

        """
        return {# standard weights, can be changed later
                'weights': {'account': 0.35, 'systematicity': 0.55, 'faithfulness': 0.1},
                #: A :code:`float`-list  that represent the penalty points for the
                #: :py:func:`account`-method. Initial values are :code:`[0, 0.3, 1, 1]`.
                'account_penalties': list(np.array([0, 0.3, 1, 1], dtype=np.float32)),
                #: A :code:`float`-list  that represent the penalty points for the
                #: :py:func:`faithfulness`-method. Initial values are :code:`[0, 0, 1, 1]`.
                'faithfulness_penalties': list(np.array([0, 0, 1, 1], dtype=np.float32))
                }

    def model_parameters_set_to_default(self):
        super().model_parameters_set_to_default()
        self.set_model_parameters(StandardReflectiveEquilibrium.default_model_parameters())


    def pick_theory_candidate(self, theory_candidates: Set[Position], **kwargs) -> Position:
        """ Implements :py:func:ReflectiveEquilibrium.pick_theory_candidate.

        Chooses randomly a theory from the theory candidates.
        """
        # single maximum
        if len(theory_candidates) == 1:
            return next(iter(theory_candidates))
        # randomly select a theory for the current branch
        return random.choice(list(theory_candidates))

    def pick_commitment_candidate(self, commitments_candidates: Set[Position], **kwargs) -> Position:
        """ Implements :py:func:ReflectiveEquilibrium.pick_commitment_candidate.

        Chooses randomly a next commitments from the commitment candidates.
        """

        if len(commitments_candidates) == 1:
            return next(iter(commitments_candidates))
        # randomly select commitment for the current branch
        return random.choice(list(commitments_candidates))

    def finished(self, **kwargs) -> bool:
        """ Implements :py:func:`ReflectiveEquilibrium.finished`.

        An re process of the standard model finishes with a state :math:`(\\mathcal{C}_i,\\mathcal{T}_i)` iff
        :math:`(\\mathcal{C}_i,\\mathcal{T}_i) = (\\mathcal{C}_{i-1},\\mathcal{T}_{i-1})`
        """
        if self.state().finished:
            return True
        else:
            self.state().finished = (len(self.state()) > 3 and not self.state().next_step_is_theory()
                                     and self.state().last_commitments() == self.state().past_commitments(-1)
                                     and self.state().last_theory() == self.state().past_theory(-1))
            return self.state().finished


class LocalReflectiveEquilibrium(StandardReflectiveEquilibrium):
    """ A locally searching RE process.

    This (abstract) class implements the following behaviour for choosing commitments/theory candidates by
    :py:func:`commitment_candidates` and :py:func:`theory_candidates`:

    * Theory candidates for the next theory are dialectically consistent positions in the vicinity of the last theory
      (as defined by the model parameter :code:`neighbourhood_depth`, see :py:func:`basics.Position.neighbours`) that
      maximize the achievement function.
    * Commitments candidates for the next commitments are minimally consistent positions in the vicinity of the last
      commitments (as defined by the model parameter :code:`neighbourhood_depth`,
      see :py:func:`basics.Position.neighbours`) that maximize the achievement function.
    * The first theory is chosen by :py:func:`first_theory`.


    *Remark:* Note, that commitment candidates can be dialectically inconsistent.

    .. note:: This class should be used together with dialectical structures that are based on binary decision trees
        (e.g. :py:class:`model.BDDDialecticalStructure`).
    """

    def __init__(self, dialectical_structure: DialecticalStructure = None, initial_commitments: Position = None,
                 model_name='LocalReflectiveEquilibrium'):
        super().__init__(dialectical_structure, initial_commitments, model_name)

    @abstractmethod
    def first_theory(self) -> Position:
        """Criterion to chose the first theory."""
        pass

    @staticmethod
    def default_model_parameters() -> Dict:
        """Default model parameters of the standard model.

        Default model parameter for the calculation of achievement:

        .. code:: python

            {'weights': {'account': 0.35, 'systematicity': 0.55, 'faithfulness': 0.1},
             'account_penalties': list(np.array([0, 0.3, 1, 1], dtype=np.float32)),
             'faithfulness_penalties': list(np.array([0, 0, 1, 1], dtype=np.float32)),
             'neighbourhood_depth': 1
             }

        """
        # see https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        default_parameters = super(LocalReflectiveEquilibrium, LocalReflectiveEquilibrium).default_model_parameters()
        default_parameters['neighbourhood_depth'] = 1
        return default_parameters

    def model_parameters_set_to_default(self):
        super().model_parameters_set_to_default()
        self.set_model_parameters(neighbourhood_depth = 1)

    def theory_candidates(self, **kwargs) -> Set[Position]:
        """Implements :py:func:`basics.ReflectiveEquilibrium.theory_candidates`"""
        candidate_theories = set()
        max_achievement = 0

        # first theory is empty theory
        if len(self.state())<=1:
            return {self.first_theory()}

        # iterate through all theory candidates in the current systematicity group
        for theory_candidate in self.state().last_theory().neighbours(self.model_parameter('neighbourhood_depth')):

            # exclude inconsistent candidates
            if self.dialectical_structure().is_consistent(theory_candidate):

                # faithfulness is fixed during theory adjustment, hence it does
                # not occur in the calculation of the current achievement
                current_achievement = self.model_parameter("weights")["systematicity"] \
                                      * self.systematicity(theory_candidate) \
                                      + self.model_parameter("weights")["account"] \
                                      * self.account(self.state().last_commitments(), theory_candidate)
                # update achievement and candidates
                if current_achievement > max_achievement:
                    candidate_theories = {theory_candidate}
                    max_achievement = current_achievement

                elif current_achievement == max_achievement:
                    candidate_theories.add(theory_candidate)
        # the old theory is again a candidate
        if self.state().last_theory() in candidate_theories:
            return {self.state().last_theory()}
        return candidate_theories

    def commitment_candidates(self, **kwargs) -> Set[Position]:
        """Implements :py:func:`basics.ReflectiveEquilibrium.commitment_candidates`"""
        candidate_commitments = set()
        max_achievement = 0

        for candidate_commitment in \
                self.state().last_commitments().neighbours(self.model_parameter('neighbourhood_depth')):

            # systematicity is fixed during theory adjustment, hence it does
            # not occur in the calculation of the current achievement
            current_achievement = self.model_parameter("weights")["faithfulness"] \
                                  * self.faithfulness(candidate_commitment, self.state().initial_commitments()) \
                                  + self.model_parameter("weights")["account"] \
                                  * self.account(candidate_commitment, self.state().last_theory())

            # update achievement and candidates
            if current_achievement > max_achievement:
                candidate_commitments = {candidate_commitment}
                max_achievement = current_achievement

            elif current_achievement == max_achievement:
                candidate_commitments.add(candidate_commitment)

        # the old commitments are again a candidate with maximal achievement
        if self.state().last_commitments() in candidate_commitments:
            return {self.state().last_commitments()}

        return candidate_commitments

class GlobalReflectiveEquilibrium(StandardReflectiveEquilibrium):
    """ A globally searching RE process.

    This class implements the following behaviour for choosing commitments/theory candidates by
    :py:func:`commitment_candidates` and :py:func:`theory_candidates`:

    * Theory candidates for the next theory are all dialectically consistent positions that maximize the achievement function.
    * Commitments candidates for the next commitments are all minimally consistent positions that maximize the achievement function.

    *Remark:* Note, that commitment candidates can be dialectically inconsistent.

    .. note:: This class should be used together with dialectical structures that are based on directed acyclic
        graph (e.g. :py:class:`model.DAGDialecticalStructure`). Globally searching RE processes are computationally complex.
        Hence, you probably won't be able to compute RE processes with a bigger sentence pool in a reasonable time.
    """

    def __init__(self, dialectical_structure: DialecticalStructure = None,
                 initial_commitments: Position = None,
                 model_name='GlobalStandardReflectiveEquilibrium'):
        super().__init__(dialectical_structure, initial_commitments, model_name)

        # for grouping positions according to their systematicity/faithfulness values
        self.__systematicity_groups = {}
        self.__faithfulness_groups = {}

    def update(self, **kwargs):
        """Implements :py:func:`basics.ReflectiveEquilibrium.update`"""
        if self.is_dirty():
            # reset and create groups of positions with identical systematicity or faithfulness
            # for more efficient candidate testing during RE process or global optima search
            logging.info(f"Updating instance of {self.__class__.__name__}.")
            self.__systematicity_groups = {}
            for theory in self.dialectical_structure().consistent_positions():
                # exclude empty theory as candidate unless its closure is non-empty
                if theory.size() or self.dialectical_structure().closure(theory).size():
                    self.__systematicity_groups.setdefault(self.systematicity(theory), set()).add(theory)

            self.__faithfulness_groups = {}
            for coms in self.dialectical_structure().minimally_consistent_positions():
                if coms.size():  # excluding the empty position as commitment candidate
                    self.__faithfulness_groups.setdefault(self.faithfulness(coms, self.state().initial_commitments()),
                                                          set()).add(coms)

            self.set_dirty(False)

    def theory_candidates(self, **kwargs) -> Set[Position]:
        """Implements :py:func:`basics.ReflectiveEquilibrium.theory_candidates`"""
        # sort systematicty values in descending order
        systematicity_values = sorted(list(self.__systematicity_groups.keys()), reverse=True)
        current_systematicity = systematicity_values.pop(0)

        candidate_theories = set()
        max_achievement = 0

        # while it is theoretically possible that a theory candidate
        # yields a better achievement value, iterate through the candidate
        # theories of a systematicity group
        while (self.model_parameter("weights")["account"] * 1 + self.model_parameter("weights")[
            "systematicity"] * current_systematicity >= max_achievement):

            # iterate through all theory candidates in the current systematicity group
            for theory_candidate in self.__systematicity_groups[current_systematicity]:

                # faithfulness is fixed during theory adjustment, hence it does
                # not occur in the calculation of the current achievement
                current_achievement = self.model_parameter("weights")["systematicity"] * current_systematicity \
                                      + self.model_parameter("weights")["account"] * \
                                      self.account(self.state().last_commitments(),
                                                                               theory_candidate)
                # update achievement and candidates
                if current_achievement > max_achievement:
                    candidate_theories = {theory_candidate}
                    max_achievement = current_achievement

                elif current_achievement == max_achievement:
                    candidate_theories.add(theory_candidate)

            if systematicity_values:
                # get next value
                current_systematicity = systematicity_values.pop(0)
            else:
                break
        # the old theory is again a candidate
        if self.state().last_theory() in candidate_theories:
            return {self.state().last_theory()}

        return candidate_theories


    def commitment_candidates(self, **kwargs) -> Set[Position]:
        """Implements :py:func:`basics.ReflectiveEquilibrium.commitment_candidates`"""
        # sort faithfulness values in descending order
        faithfulness_values = sorted(list(self.__faithfulness_groups.keys()), reverse=True)
        current_faithfulness = faithfulness_values.pop(0)

        candidate_commitments = set()
        max_achievement = 0

        # while it is theoretically possible that a commitment candidate
        # yields a better achievement value, iterate through the candidate
        # commitments of a faithfulness group

        while (self.model_parameter("weights")["account"] * 1 + self.model_parameter("weights")[
            "faithfulness"] * current_faithfulness >= max_achievement):

            for candidate_commitment in self.__faithfulness_groups[current_faithfulness]:

                # systematicity is fixed during theory adjustment, hence it does
                # not occur in the calculation of the current achievement
                current_achievement = self.model_parameter("weights")["faithfulness"] * current_faithfulness \
                                      + self.model_parameter("weights")["account"] * \
                                      self.account(candidate_commitment, self.state().last_theory())

                # update achievement and candidates
                if current_achievement > max_achievement:
                    candidate_commitments = {candidate_commitment}
                    max_achievement = current_achievement

                elif current_achievement == max_achievement:
                    candidate_commitments.add(candidate_commitment)

            if faithfulness_values:
                # get next value
                current_faithfulness = faithfulness_values.pop(0)
            else:
                break

        # the old theory is again a candidate
        if self.state().last_commitments() in candidate_commitments:
            return {self.state().last_commitments()}

        return candidate_commitments

    def global_optima(self, initial_commitments: Position) -> Set[Tuple[Position, Position]]:
        """Searches for globally optimal theory-commitment pairs (according to
        the achievement function).

        :param initial_commitments: A Position

        :return: A set of globally optimal theory-commitment-pairs as Positions.
        """

        self.set_initial_state(initial_commitments)
        self.update()

        # The values of systematicity and faithfulness can be calculated for each position individually,
        # because their calculation does not depend on the theory-commitment relation expressed as pair.
        # Thus, positions can be grouped according to their values:
        # all theory candidates belong to a systematicity group, all commitments candidates to a faithfulness group.

        # In nested loops iterate through groups of theories (outer) and commitments (inner)
        # in decreasing order of their values.
        # If the values have the potential to exceed the current maximal achievement (while condition),
        # go on constructing all theory-commitment-pairs from the groups, otherwise break.
        # The achievement of every constructed pair is calculated and compared to the current maximal achievement.

        # On average, this method is expected to be more efficient than brute forcing,
        # because it relies on heuristics to determine an early breakpoint.

        # sort systematicity values in decreasing order to prepare iteration
        systematicity_values = sorted(list(self.__systematicity_groups.keys()), reverse=True)

        counter = 0  # number of constructed pairs for comparison with brute force algo
        optimal_pairs = set()
        max_achievement = 0

        current_systematicity = systematicity_values.pop(0)  # first systematicity value

        while (current_systematicity * self.model_parameter("weights")['systematicity']
               + 1 * self.model_parameter("weights")['faithfulness']
               + 1 * self.model_parameter("weights")['account'] >= max_achievement):

            # sort/reset faithfulness values in decreasing order for iteration
            faithfulness_values = sorted(list(self.__faithfulness_groups.keys()), reverse=True)

            # grab first faithfulness value
            current_faithfulness = faithfulness_values.pop(0)

            while (current_systematicity * self.model_parameter("weights")['systematicity'] + current_faithfulness
                   * self.model_parameter("weights")['faithfulness']
                   + 1 * self.model_parameter("weights")['account'] >= max_achievement):

                # construct every theory-commitment pair from the current groups
                for (theory, coms) in product(self.__systematicity_groups[current_systematicity],
                                              self.__faithfulness_groups[current_faithfulness]):
                    counter += 1
                    achievement = self.achievement(coms, theory, initial_commitments)

                    # decide whether the theory-commitment pair is better or equally good as the current optimal pairs.
                    if achievement > max_achievement:

                        max_achievement = achievement
                        optimal_pairs = {(theory, coms)}

                    elif achievement == max_achievement:

                        optimal_pairs.add((theory, coms))

                if faithfulness_values:
                    # get next value
                    current_faithfulness = faithfulness_values.pop(0)
                else:
                    break  # exit faithfulness loop, -> next iteration of systematicity loop

            if systematicity_values:
                # get next value
                current_systematicity = systematicity_values.pop(0)
            else:
                break

        # optionally, max_achievement or counter could also be returned
        return optimal_pairs


# ToDo: Add class docstring
class REContainer:

    def __init__(self, re_models: List[ReflectiveEquilibrium] = None):
        self.re_models = re_models
        self._objects = dict()

    @abstractmethod
    def re_processes(self, re_models: List[ReflectiveEquilibrium] = None) -> Iterator[ReflectiveEquilibrium]:
        pass

    # ToDo: add docstring
    def add_object(self, key: Any, object: Any):
        self._objects[key] = object

    # ToDo: Add docstring
    def get_object(self, key: Any) -> [Any, None]:
        if key in self._objects.keys():
            return self._objects[key]
        else:
            return None

    def get_objects(self):
        return self._objects

# ToDo: Add for all methods more information and examples (see the defined test cases).
class REState:
    """Class that represent the internal state of an RE process.

    *Remark:* The attribute :code:`time_line` can be used to externally assign points in time to the steps of
    the RE process. This feature will be used to coordinate different re processes with each other.

    Attributes:

        finished (bool): A boolean indicating whether the process terminated in a fixed point.
        evolution (List[Position]): The evolution of steps beginning with the initial state
            (i.e., a set of commitments).
        alternatives (List[Set[Position]]): For each step a set of possible alternatives the process could have
            used as step according to :py:func:`ReflectiveEquilibrium.commitment_candidates`
            (or :py:func:`ReflectiveEquilibrium.theory_candidates` respectively).
        time_line (List[int]): For each step an integer that represents a point in time according to an (external)
            timeline.
        error_code: An integer that represents an error code and which can be set if the process throws an error.
    """

    error_codes = {# Should be used if there is no more specific error code available.
                   0: 'The process could not finish due to an unexpected error.',
                   # Should be used if some max_loop value was exceeded.
                   1: 'The process did not converge under a specified maximum number of steps (see process logs for' +\
                      ' further details).'
                  }

    def __init__(self,
                 finished: bool,
                 evolution: List[Position],
                 alternatives: List[Set[Position]],
                 time_line: List[int],
                 error_code: int = None):
        # check wither the time_line is ordered
        if not all(time_line[i] < time_line[i + 1] for i in range(len(time_line) - 1)):
            raise ValueError("Timeline must be an ordered sequence of integers.")
        if len(evolution) != len(alternatives) or len(evolution) != len(time_line):
            raise ValueError("The given lists (evolution, alternatives, time_line) must be of equal length.")

        self.finished = finished
        self.evolution = evolution
        self.alternatives = alternatives
        self.time_line = time_line
        self.error_code = error_code

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.finished == other.finished and self.evolution == other.evolution \
                   and self.alternatives == other.alternatives and self.time_line == other.time_line

        else:
            return False

    def __len__(self):
        return len(self.evolution)

    def as_dict(self) -> Dict:
        """The state as python dictionary.

        Returns:
            A dictionary with the state's attributes as key-value pairs.
        """
        return {'finished': self.finished,
                'evolution': self.evolution,
                'alternatives': self.alternatives,
                'time_line': self.time_line}

    @staticmethod
    def from_dict(state_dict: Dict) -> REState:
        """Instantiation via dict.

        Args:
            state_dict (Dict): A dictionary with key-value pairs representing the attribute of :class:`REState`
        """
        if any(key not in state_dict for key in ['finished',
                                                 'evolution',
                                                 'alternatives',
                                                 'time_line']):
            raise KeyError("state_dict needs to comprise the following keys: "
                           "finished, evolution , alternatives, time_line")

        return REState(state_dict['finished'],
                       state_dict['evolution'],
                       state_dict['alternatives'],
                       state_dict['time_line'])

    def commitments_evolution(self) -> List[Position]:
        """Evolution of commitments.

        Returns:
            The steps of the process that represents commitments.
        """
        if self.evolution:
            return [self.evolution[i * 2] for i in np.arange(trunc(len(self.evolution) / 2 + len(self.evolution) % 2))]
        return None

    def theory_evolution(self) -> List[Position]:
        """Evolution of theories.

        Returns:
            The steps of the process that represents theories.
        """
        if self.evolution:
            return [self.evolution[i * 2 + 1] for i in np.arange(trunc(len(self.evolution) / 2))]
        return None

    def initial_commitments(self) -> Position:
        """The initial commitments of the process."""
        if self.evolution:
            return self.evolution[0]
        return None

    def next_step_is_theory(self) -> bool:
        """Checks whether the next step is a theory."""
        if self.evolution:
            return len(self.evolution) % 2 == 1
        return False

    def last_commitments(self) -> Position:
        """The last chosen commitments."""
        return self.past_commitments(past_step=0)

    def past_commitments(self, past_step: int = None, time: int = None) -> Position:
        """Past commitments.

        Convenience method to access past commitments either by their timeline or by commitment
        steps counting backwards. You have to provide exactly one of the arguments.

        Args:
            past_step (int): The past counted stepwise beginning from the last commitment (:code:`0` indicating the
                last commitments, :code:`-1` the next-to-last commitments and so on).
            time (int): The point in time according to :py:attr:`REState.evolution`.
        """
        if past_step != None and time != None:
            raise ValueError("You cannot assign to both arguments a value.")
        if past_step is None and time is None:
            raise ValueError("You have to provide a value to one of the arguments.")
        if time is not None:
            # We do not assume that processses end with commitments
            # timeline of coms
            time_line_coms = [self.time_line[i] for
                              i in np.arange(0, (len(self.time_line) + len(self.time_line) % 2), 2)]
            times = np.where([step_time <= time for step_time in time_line_coms])[0]
            past_coms = None if times.size == 0 else self.commitments_evolution()[times[-1]]
            return past_coms
        if past_step is not None:
            index = -1 + 2 * past_step if self.next_step_is_theory() else -2 + 2 * past_step
            if index + len(self.evolution) >= 0:
                return self.evolution[index]

        return None

    def last_theory(self) -> Position:
        """The last chosen theory."""
        return self.past_theory(past_step=0)

    def past_theory(self, past_step: int = None, time: int = None) -> Position:
        """Past theory.

        Convenience method to access past theories either by their timeline or by theory
        steps counting backwards. You have to provide exactly one of the arguments.

        Args:
            past_step (int): The past counted stepwise beginning from the last theory (:code:`0` indicating the
                last theory, :code:`-1` the next-to-last theory and so on).
            time (int): The point in time according to :py:attr:`REState.evolution`.
        """
        if past_step != None and time != None:
            raise ValueError("You cannot assign to both arguments a value.")
        if past_step is None and time is None:
            raise ValueError("You have to provide a value to one of the arguments.")
        if time is not None:
            # We do not assume that processses end with commitments
            # timeline of coms
            time_line_theories = [self.time_line[i] for
                                  i in np.arange(1, (len(self.time_line) - len(self.time_line) % 2), 2)]
            times = np.where([step_time <= time for step_time in time_line_theories])[0]
            past_theory = None if times.size == 0 else self.theory_evolution()[times[-1]]
            return past_theory
        if past_step is not None:
            index = -2 + 2 * past_step if self.next_step_is_theory() else -1 + 2 * past_step
            if index + len(self.evolution) >= 0:
                return self.evolution[index]
        return None

    def add_step(self, position: Position, alternatives: Set[Position], time: int = None):
        """Adding a new step to the state.

        Args:
            position (Position): The added step's position.
            alternatives (Set[Position]): The alternatives to the added step's position.
            time (int): A point in time that attributes the next step to an external timeline.
        """
        self.evolution.append(position)
        self.alternatives.append(alternatives)
        if time:
            if time <= self.time_line[-1]:
                raise ValueError("Time must be greater than past times.")
            self.time_line.append(time)
        else:
            self.time_line.append(self.time_line[-1] + 1)

    def last_step(self) -> Position:
        """Last step of the evolution."""
        return self.evolution[-1]

    def past_step(self, time: int) -> Position:
        """A past step according to the timeline.

        *Remark:* The time line of the state maps each step to a point in time on an (external) timeline.
        An re process can, among other things, pause according to this timeline in the case that the states'
        timeline looks e.g. like :code:`[0,1,2,5]`. Accordingly, the third step persist through the time points
        2,3,4 on the external timeline in this specific example.

        Args:
            time (int): The point in time attributed to the requested step.
        Returns:
            Returns :code:`None` if :code:`time` falls before the time of the initial state. Otherwise, it
                will return the requested :class:`Position`.
        """
        times = np.where([step_time <= time for step_time in self.time_line])[0]
        past_step = None if times.size == 0 else self.evolution[times[-1]]
        return past_step

    def last_alternatives(self):
        """Alternatives of the last step."""
        if self.alternatives:
            return self.alternatives[-1]
        else:
            return None

    def indices_of_non_empty_alternatives(self) -> List[int]:
        """ Steps in the process with non-empty alternatives.
        """
        return list(np.where([alt != set() for alt in self.alternatives])[0])

    def past(self, past: int) -> REState:
        """A past state as copy.

        Args:
            past (int): Int indicating the past as counted by steps. :code:`-1` represents the last state,
                :code:`-2` represents the next-to-last state and so on.
        Returns:
            A state representing the past state as a copy.
        """
        if past > 0:
            raise ValueError("Past must be <=0.")
        if past < -len(self) + 1:
            raise ValueError("Past can maximally reach to the initial start state.")

        return REState(finished=False if past < 0 else self.finished,
                       evolution=copy(self.evolution[0:len(self.evolution) + past]),
                       alternatives=copy(self.alternatives[0:len(self.alternatives) + past]),
                       time_line=copy(self.time_line[0:len(self.time_line) + past]))

class MaxLoopsWarning(RuntimeWarning):

    def __init__(self, message: [str, None] = None ):
        msg = "Reached max loop count for processes without finishing all processes."
        if message:
            super().__init__(message)
        else:
            super().__init__(msg)

class MaxBranchesWarning(RuntimeWarning):

    def __init__(self, message: [str, None] = None ):
        msg = "Reached max branches count for processes without finishing all processes."
        if message:
            super().__init__(message)
        else:
            super().__init__(msg)