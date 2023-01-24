from __future__ import annotations

from tau import (
    Position,
    DialecticalStructure,
    BitarrayPosition
)
from tau.util import inferential_density, get_principles
from .base import ReflectiveEquilibrium
from rethon import REState
from .core import FullBranchREContainer, REContainer

from abc import ABC, abstractmethod
from copy import copy
from pandas import Series
import pandas as pd
import numpy as np
import statistics
from typing import Set, Iterator, List, Dict
from collections.abc import Callable
from os import path, remove
import tarfile
import importlib
import logging

class AbstractEnsembleGenerator(ABC):
    """Abstract base class for all ensemble generators.

    This abstract class should be used to implement ensemble generators. It implements basic functionalities to
    add data items that will be used to generate dictionaries for each model run and that will contain information
    about each model run according to the specified data items. Data items (key-value pairs in the produced dict)
    can be defined by specifying a key and a function that returns the desired data (see :py:func:`add_item`).

    """

    def __init__(self):
        self._obj = {}
        self._items = {}
        self._item_funs = {}

        self.current_dialectical_structure = None
        self.current_state = None
        self.current_reflective_equilibrium = None
        self.current_initial_commitments = None
        self.current_ensemble_states = None

    @abstractmethod
    def ensemble_iter(self) -> Iterator[ReflectiveEquilibrium]:
        """Iterator through all model runs.

        This method should be overriden by subclasses and is responsible to call :py:func:`init_tau_fields`,
        :py:func:`init_re_start_fields`, :py:func:`init_re_final_fields` and :py:func:`init_ensemble_fields` at
        appropriate times. Addionally, it should reset the internal attributes :py:attr:`current_dialectical_structure`,
        :py:attr:`current_state`, :py:attr:`current_reflective_equilibrium`, :py:attr:`current_initial_commitments` and
        :py:attr:`current_ensemble_states` that can be used to define data item (see :py:func:`add_item`).
        """
        pass

    def ensemble_items_iter(self) -> Iterator[Dict]:
        """ Iterator through data items of model runs.

        This methods uses :py:func:`ensemble_iter` to return for each model run a dictionary with
        predefined data items as key-value pairs. The data items can be added with :py:func:`add_item`.

        """
        for re in self.ensemble_iter():
            self._fill_cols()
            yield copy(self._items)

    def ensemble_items_to_csv(self,
                              output_file_name: str,
                              output_dir_name: str,
                              archive = False,
                              save_preliminary_results: bool = False,
                              preliminary_results_interval: int = 500, append = False):
        """Saving model runs as csv file.

        Using :py:func:`ensemble_items_iter` to save the data items of each model run as csv file.

        Args:
            output_file_name (str): Name of the csv file.
            output_dir_name (str): Directory of the csv file.
            archive (bool): If :code:`True` the csv file will be archived as `tar.gz`. The csv file will not be removed.
            save_preliminary_results (bool): If :code:`True` the method will save preliminary results every other
                model run as defined by :code:`preliminary_results_interval`.
            append (bool): If :code:`True` the rows will be added to the file (if it already exists).
        """
        output_file = path.join(output_dir_name, output_file_name)
        if not append and path.exists(output_file):
            remove(output_file)

        loop_counter = 0
        rows = []

        for items_dict in self.ensemble_items_iter():
            rows.append(items_dict)
            loop_counter += 1
            if save_preliminary_results and loop_counter % preliminary_results_interval == 0:
                data = pd.DataFrame(rows)
                data.to_csv(output_file,
                            mode='a',
                            header=not path.exists(output_file),
                            index=False)
                rows = []

        data = pd.DataFrame(rows)
        data.to_csv(output_file,
                    mode='a',
                    header=not path.exists(output_file),
                    index=False)
        if archive:
            tar_file = path.join(output_dir_name, output_file_name + '.tar.gz')
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(output_file, recursive=False, arcname=output_file_name)


    def init_tau_fields(self, tau):
        """Initiating data objects that rely on the dialectical structure.

        This method can be called by implementations of :py:func:`ensemble_iter` and can be used to store objects
        by :py:func:`add_obj` that in turn can be accessed by :py:func:`get_obj` in the definition of data items.

        The idea is to reuse data instead of recalculating it in every model run inasmuch (depending on the
        implementation of this class) several model runs can be based on the same dialectical structure :math:`\\tau`.
        Accordingly, implementations of :py:func:`ensemble_iter` should call this method once for every model run
        that is based on a particular dialectical structure. A subclass, say :code:`MyEnsembleGenerator`,
        can then override this method:

        .. code:: python

            def init_tau_fields(self, tau):
                self.add_obj('inferential_density',
                             tau.inferential_density())

        The implementation can then access this object in the definition of data items, for instance:

        .. code:: python

            generator = MyEnsembleGenerator()
            generator.add_item('calculated_inferential_density',
                               lambda x: x.get_obj('inferential_density'))

        """
        pass

    def init_re_start_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Initiating data objects that rely on the dialectical structure and an instantied RE.

        This method can be called by implementations of :py:func:`ensemble_iter` and can be used to store objects
        by :py:func:`add_obj` that in turn can be accessed by :py:func:`get_obj` in the definition of data items.
        Implementing classes should ensure that at least the initial commitments and model parameters of the
        RE instance are already set. (For a rationale of why using this method, see :py:func:`init_tau_fields`.)
        """
        pass

    def init_re_final_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Initiating data objects that rely on the dialectical structure and a finished model run.

        This method can be called by implementations of :py:func:`ensemble_iter` and can be used to store objects
        by :py:func:`add_obj` that in turn can be accessed by :py:func:`get_obj` in the definition of data items.
        Implementing classes should ensure that the model represented by :code:`reflective_equilibrium` finished in
        a fixed point. (For a rationale of why using this method, see :py:func:`init_tau_fields`.)
        """


    # idea: if branching fields are empty (and save_branches is not true), do not branch
    def init_ensemble_fields(self, ensemble_states: List[REState],
                             dialectical_structure: DialecticalStructure):
        """Initiating data objects that rely on the dialectical structure and a set of finished model runs.

         This method can be called by implementations of :py:func:`ensemble_iter` and can be used to store objects
         by :py:func:`add_obj` that in turn can be accessed by :py:func:`get_obj` in the definition of data items.
         Implementing classes should ensure that all models that are aggregated in a sub-ensemble finished in
         fixed points. (For a rationale of why using this method, see :py:func:`init_tau_fields`.)
         """

    def state(self) -> REState:
        """Current state when iterating through model runs."""
        return self.current_state

    def dialectical_structure(self) -> DialecticalStructure:
        """Current dialectical structure when iterating through model runs."""
        return self.current_dialectical_structure

    def reflective_equilibrium(self) -> ReflectiveEquilibrium:
        """Current state when iterating through the model runs."""

        return self.current_reflective_equilibrium

    def ensemble_states(self) -> List[REState]:
        """Current ensemble states when iterating through the model runs."""
        return self.current_ensemble_states

    def initial_commitments(self) -> Position:
        """Current initial commitments when iterating through the model runs."""
        return self.current_initial_commitments

    def add_item(self, key: str, fun: Callable[[AbstractEnsembleGenerator], None]):
        """Adds a data item.

        Adding a key-method pair that will be used to calculate a data item. Data items will be returned for each model
        run by :py:func:`ensemble_items_iter`. A data item represents data that describes a model run in some way. It
        can be defined by providing a function that can use the internal attributes (e.g. via its getters
        :py:func:`state()`, :py:func:`dialectical_structure()`, :py:func:`reflective_equilibrium()`,
        :py:func:`initial_commitments()` and :py:func:`ensemble_states()`) and deposited objects (via
        :py:func:`add_obj`). The function should specify an :py:class:`AbstractEnsembleGenerator`
        as its one argument.

        A simple example is the following:

        .. code:: python

            # defining the function
            def data_last_coms(ensemble_generator):
                return ensemble_generator.state().last_commitments()

            # adding the data item
            my_ensemble_generator.add_item('last_coms', data_last_coms)

            # looping through the ensemble using the item_iter will now
            # have a key-value pair that stores the final commitments
            # for each model run
            for model_run_data_items in my_ensemble_generator.ensemble_items_iter():
                print("The model run produced the following data items:")
                print(f"{model_run_data_items}.")

        Args:
            key (str): A key for the data item.
            fun (Callable[[AbstractEnsembleGenerator], None]): A function that defines a datum of a model run.

        """
        self._item_funs[key] = fun

    def get_item(self, key):
        """Returns the return value of the function that is defined by the data item."""
        # if already calculated return saved value
        if key in self._items.keys():
            return self._items[key]
        # otherwise caculate on the fly
        else:
            return self._item_funs[key](self)

    def remove_item(self, key: str):
        """Removing a data item."""
        del self._item_funs[key]

    def add_obj(self, key: str, obj):
        """Adding a data object.

        This method can be used to store data objects that persists through different model runs and can then be accessed
        via :py:func:`get_obj()`. See :py:func:`init_tau_fields()` for the rationale of using data objects.
        """
        self._obj[key] = obj

    def get_obj(self, key):
        """Returns a data object."""
        return self._obj[key]

    def _fill_cols(self):
        self._items.clear()
        for key in self._item_funs.keys():
            self._items[key] = self._item_funs[key](self)




class EnsembleGenerator(AbstractEnsembleGenerator):
    """ Ensemble generator base class for independent model runs.

    A class that provides iterators for model runs based on the given parameters of the constructor. The iterator
    will be build as a cartesian product of these arguments (see :py:func:`EnsembleGenerator.ensemble_iter`). The
    model runs of the ensemble are all based on the same sentence pool. If you whish to generate ensemble
    with differing sentence pool, you can concatenate instances of this class.

    Args:
        arguments_list: Dialectical structures as list of argument lists.
        n_sentence_pool: Number of (unnegated) sentences in the sentence pool.
        initial_commitments_list: List of initial commitments. Each commitments is represent as set of integers.
        model_parameters_list: A list of dictionaries that represents the model parameters and can be set by
            :py:func:`ReflectiveEquilibrium.set_model_parameter`.
        create_branches: If :code:`True` all branches are created.
        implementations: A list of dicts, each representing a specific implementation. Each dict should contain
            strings for the keys 'tau_module_name', 'rethon_module_name', 'position_class_name',
            'dialectical_structure_class_name'
            and 'reflective_equilibrium_class_name'. (If these classes are located in different modules, you can,
            alternatively, specify modules for each class by using the keys 'position_module_name',
            'dialectical_structure_module_name' and 'reflective_equilibrium_module_name')
    """
    def __init__(self,
                 arguments_list: List[List[List[int]]],
                 n_sentence_pool: int,
                 initial_commitments_list: List[Set[int]],
                 model_parameters_list: List[Dict] = None,
                 create_branches = False,
                 implementations: List[Dict] = None):

        super().__init__()
        self.arguments_list = arguments_list
        self.n_sentence_pool = n_sentence_pool
        self.initial_commitments_list = initial_commitments_list
        self.model_parameters_list = model_parameters_list
        self.create_branches = create_branches
        if implementations is None:
            self.implementations = _fill_module_names([{'tau_module_name': 'tau',
                                                        'position_class_name': 'StandardPosition',
                                                        'dialectical_structure_class_name': 'DAGDialecticalStructure',
                                                        'rethon_module_name': 'rethon',
                                                        'reflective_equilibrium_class_name': 'StandardGlobalReflectiveEquilibrium'
                                                        }])
        else:
            self.implementations = _fill_module_names(implementations)

    # returns iterator of dicts (which can serve as a row)
    def ensemble_iter(self) -> Iterator[ReflectiveEquilibrium]:
        """ Iterator through re processes.

            An ensemble iterator that produces all ensembles in the cartesian product of the given
            dialectical structures (:py:attr:`arguments_list`), the initial commitments
            (:py:attr:`initial_commitments_list`), the list of
            model parameters (:py:attr:`model_parameters_list`) and the given implementations
            (:py:attr:`implementations`). If :py:attr:`create_branches` is set to
            :code:`True` every branch resulting from an underdetermination of commitments or theory candidates will
            be returned as well.
        """
        ensemble_size = len(self.implementations)*len(self.arguments_list)\
                        *(len(self.model_parameters_list) if self.model_parameters_list else 1)\
                        *len(self.initial_commitments_list)
        logging.info(f"Starting ensemble generation with {ensemble_size} models runs (without branches)")
        for impl in self.implementations:

            for arguments in self.arguments_list:
                # instantiating dialectical structure
                ds_class_ = getattr(importlib.import_module(impl['dialectical_structure_module_name']),
                                    impl['dialectical_structure_class_name'])
                ds = ds_class_.from_arguments(arguments, self.n_sentence_pool)
                self.current_dialectical_structure = ds
                self.init_tau_fields(ds)
                # instantiating re
                reflective_equilibrium_class_ = getattr(importlib.import_module(
                        impl['reflective_equilibrium_module_name']),
                        impl['reflective_equilibrium_class_name'])
                re = reflective_equilibrium_class_(ds)

                self.current_reflective_equilibrium = re

                if not self.model_parameters_list:
                    self.model_parameters_list = [reflective_equilibrium_class_.default_model_parameters()]

                for model_parameters in self.model_parameters_list:
                    #logging.info(f"Re model param {re.model_parameter('weights')}")
                    re.reset_model_parameters(model_parameters)
                    #logging.info(f"Re set model param to {re.model_parameter('weights')}")
                    for pos_as_set in self.initial_commitments_list:
                        pos_class_ = getattr(importlib.import_module(impl['position_module_name']),
                                             impl['position_class_name'])
                        pos = pos_class_.from_set(pos_as_set, self.n_sentence_pool)
                        self.current_initial_commitments = pos
                        re.set_initial_state(pos)
                        self.init_re_start_fields(re, ds)
                        if self.create_branches:
                            # Idea: we first collect all finished states of the branches for 'init_branching_fields'
                            # and then iter through them.
                            re_container = FullBranchREContainer()
                            branching_res = list(re_container.re_processes([re]))
                            branching_states = [re.state() for re in branching_res]

                            self.current_ensemble_states = branching_states
                            self.init_ensemble_fields(branching_states, ds)

                            #logging.info(f"n branches: {len(branching_states)}")
                            for rec in branching_res:
                                self.current_reflective_equilibrium = rec
                                self.current_state = rec.state()
                                self.init_re_final_fields(rec, ds)
                                #logging.info(f"Yield re with params: {rec.model_parameter('weights')}")
                                #logging.info(f"branch copy is orig?: {re is rec}")
                                yield rec
                        else:
                            re.re_process()
                            self.init_re_final_fields(re, ds)
                            self.current_state = re.state()
                            # yield a copy since we reuse the instance with new model-parameters
                            yield copy(re)




#todo: comment those fields that are only visible if branching is switched on
class SimpleEnsembleGenerator(EnsembleGenerator):
    """
    This class extends :py:class:`EnsembleGenerator` by adding the following data items that are produced by
    :py:func:`AbstractEnsembleGenerator.ensemble_items_iter`:

    **FEATURES OF THE DIALECTICAL STRUCTURE**

    * :code:`model_name`: A name of the model as string.
    * :code:`ds`: The dialectical structure as list of int-lists. The first numbers of each list represent the premises,
      the last one the conclusion.
    * :code:`n_sentence_pool`: Number of unnegated sentences (half the full size).
    * :code:`ds_arg_size`: Number of arguments.
    * :code:`ds_infer_dens`: The inferential density of the structure.
    * :code:`ds_n_consistent_complete_positions`: Number of dialectically complete and consistent positions.
    * :code:`ds_mean_prem`: Mean number of premises per argument.
    * :code:`ds_variance_prem`: Variance of the number of premises per argument.
    * :code:`tau_truths`: Propositions that are true in every complete consistent position.
    * :code:`principles`: A list of tuples of the form :code:`(principle, multiplicity)` with :code:`multiplicity`
      indicating the multiplicity of the principle.
      A sentence counts as a principle iff it occurs in at least one argument as premise and it or its negation
      does not occur as a conclusion in an argument. The multiplicity counts in how many argument the sentence occurs
      as premise.

    **PARAMETERS OF THE RE-PROCESS**

    * :code:`account_penalties`:
    * :code:`faithfulness_penalties`:
    * :code:`weight_account`:
    * :code:`weight_systematicity`:
    * :code:`weight_faithfulness`:

    **PROCESS-FEATURES**

    * :code:`init_coms`: The initial commitments.
    * :code:`init_coms_size`: Number of initial commitments.
    * :code:`init_coms_n_tau_truths`: Number of tau-true propositions in the initial commitments.
    * :code:`init_coms_n_tau_falsehoods`: Number of tau-false propositions in the initial commitments.
    * :code:`init_coms_min_ax_bases`: Minimal-length axiomatic bases of :code:`init_coms` with sentences
      from :code:`init_coms` only. (Sentences :math:`\\mathcal{C}_a` are an axiomatic basis
      of :math:`\\mathcal{C}_1` (with sentences from :math:`\\mathcal{C}_2` only) iff :math:`\\mathcal{C}_a` dialectically
      implies :math:`\\mathcal{C}_1` and there is no proper subset of :math:`\\mathcal{C}_a` such that it implies
      :math:`\\mathcal{C}_1` (and :math:`\\mathcal{C}_a \subset \mathcal{C}_2`).)
    * :code:`n_init_coms_min_ax_base`: The length of the minimal-length axiomatic bases (not the number of such bases).
    * :code:`init_coms_n_consistent_complete_positions`: Number of consistent complete positions that extend the initial
      commitments.
    * :code:`init_coms_dia_consistent`: Whether the initial commitments are dialectically consistent.
    * :code:`init_coms_closed`: Whether the initial commitments are dialectically closed.
    * :code:`init_coms_closure`: Dialectical closure of initial commitments.
    * :code:`fixed_point_coms`: Final commitments.
    * :code:`fixed_point_coms_size`: The number of propositions in :code:`fixed_point_coms`.
    * :code:`fixed_point_coms_com_n_tau_truths`: Number of tau-true propositions in the final commitments.
    * :code:`fixed_point_coms_com_n_tau_falsehoods`: Number of tau-false propositions in the final commitments.
    * :code:`fixed_point_coms_closed`: Whether the final commitments is dialectically closed.
    * :code:`fixed_point_coms_closure`: Dialectical closure of final commitments.
    * :code:`fixed_point_coms_consistent`: Whether the final commitments is dialectically consistent.
    * :code:`fixed_point_theory`: The final theory.
    * :code:`fixed_point_theory_closure`: Closure of the final theory.
    * :code:`achievements_evolution`:
    * :code:`fixed_point_dia_consistent`: Whether the union of final theory & commitments is dialectically consistent.
    * :code:`init_final_coms_simple_hamming`: Simple hamming distance between initial and final commitments.
    * :code:`init_final_coms_hamming`: Hamming distance between initial and final commitments as defined in
      BBB with d_3 = 1, d_2 = 1, d_1 = 1, d_0 = 0.
    * :code:`init_final_coms_contradictions`: Amount of contradictions: number of sentences whose contradictions
      are in both positions (i.e. hamming distance as defined in BBB with d_3 = 1, d_2 = 0, d_1 = 0, d_0 = 0)
    * :code:`init_final_coms_expansions`: Amount of expansions without expansions that lead to
      contradictions or identities (i.e. hamming distance as defined in BBB with d_3 = 0, d_2 = 0, d_1 = 1, d_0 = 0)
    * :code:`init_final_coms_contractions`: Amount of contractions (i.e. hamming distance as
      defined in BBB with d_3 = 0, d_2 = 1, d_1 = 0, d_0 = 0)
    * :code:`init_final_coms_identities`: Amount of identities without identities that are associated with contradictions
      (i.e. hamming distance as defined in BBB with d_3 = 0, d_2 = 0, d_1 = 0, d_0 = 1)
      attention: counts also sentences on which both positions are indifferent.
    * :code:`random_choices`:
    * :code:`n_random_choices`: Number of random choice of the process.
    * :code:`comms_evolution`: The dynamic evolution of theories during the process. (Depicting the algorithmic
      process.)
    * :code:`theory_evolution`: The dynamic evolution of commitments during the process. (Depicting the algorithmic
      process.)
    * :code:`process_length`: The length of the process. Defined as the number of steps that will either produce a change
      in the commitments or a change in the theory, beginning the choosing the first theory. (I.e. this number will generally
      be smaller than the amount of elements combined in :code:`commitments_evolution` and :code:`theory_evolution`.)

    **PROCESS-INDEPENDENT FEATURES**

    In the case that the ensemble generator is configured (via its instantiation) to run all branches the following
    data items are added as well:

    * :code:`n_branches`: Number of branches of the process (i.e. paths to all fixed points w.r.t. the given initial
      commitments.

    *Fixed points*:

    * :code:`fixed_points`: All fixed points as theory-commitments-tuples.
    * :code:`n_fixed_points`: Number of fixed points
    * :code:`fp_coms_consistent`: A list of bools (:code:`List[bool]`) indicating whether commitments of the `fixed_points`
      are dialectically consistent. The order of the list represents the order in `fixed_points`.
    * :code:`fp_union_consistent`:  A list of bools (:code:`List[bool]`) indicating whether the unions of a
      commitment-theory-tuple of the `fixed_points` are dialectically consistent.The order of the list represents the
      order in `fixed_points`.
    * :code:`fp_account`: The account of each fixed point as :code:`List[float]`. The order of the list represents the
      order in `fixed_points`.
    * :code:`fp_faithfulness`: The faithfulness of each fixed as :code:`List[float]`. The order of the list represents
      the order in `fixed_points`.


    """

    def __init__(self, arguments_list: List[List[List[int]]], n_sentence_pool: int,
                 initial_commitments_list: List[Set[int]], model_parameters_list: List[Dict] = None,
                 create_branches=False, implementations: List[Dict] = None):
        super().__init__(arguments_list, n_sentence_pool, initial_commitments_list, model_parameters_list,
                         create_branches, implementations)
        _add_simple_data_items(self)
        if create_branches:
            _add_full_branch_data_items(self)

    def init_tau_fields(self, tau: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_tau_fields`.

        Adds the following data objects that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'ds_infer_dens', 'n_premises', 'principles', 'tau_truths' and 'tau_falsehoods'.
        """
        self.add_obj('ds_infer_dens', inferential_density(tau))
        self.add_obj('n_premises', [len(arg) for arg in tau.get_arguments()])
        self.add_obj('principles', get_principles(tau.get_arguments()))
        tau_truths = tau.closure(BitarrayPosition(set(),
                                                  tau.sentence_pool().size()))
        self.add_obj('tau_truths', tau_truths)
        self.add_obj('tau_falsehoods', BitarrayPosition({-prop for prop in tau_truths.as_set()},
                                                        tau.sentence_pool().size()))


    def init_re_start_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_re_start_fields`.

        Adds the following data object that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'init_com_min_ax_bases'.
        """
        init_coms = reflective_equilibrium.state().initial_commitments()

        if dialectical_structure.is_consistent(init_coms):
            init_com_min_ax_bases = _get_min_sets([axioms.as_set() for axioms in
                                                    dialectical_structure.axioms(
                                                        init_coms,
                                                        init_coms.subpositions())])
        else:
            init_com_min_ax_bases = np.nan
        self.add_obj('init_com_min_ax_bases', init_com_min_ax_bases)



    def init_ensemble_fields(self, ensemble_states: List[REState], dialectical_structure: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_ensemble_fields`.

        Adds the following data object that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'n_branches' and fixed_points', if the ensemble generator is set up to run every branch.
        """
        self.add_obj('n_branches', len(ensemble_states))

        # get all fixed_points of the process
        branched_theories = [branch_state.last_theory() for branch_state in ensemble_states]
        branched_commitments = [branch_state.last_commitments() for branch_state in ensemble_states]
        fixed_points = list(Series(zip(branched_theories, branched_commitments)).unique())
        self.add_obj('fixed_points', fixed_points)



class GlobalREEnsembleGenerator(SimpleEnsembleGenerator):
    """
    This class extends :class:`SimpleEnsembleGenerator` by adding the following data items that are produced by
    :py:func:`AbstractEnsembleGenerator.ensemble_items_iter`:

    Additional items for the fixed point:

    * :code:`fixed_point_is_global_optimum`: Whether the fixed point is a global optimum.
    * :code:`fixed_point_is_re_state`: Whether the fixed point is a RE state.
    * :code:`fixed_point_is_full_re_state`: Whether the fixed point is a full RE state.
    * :code:`fixed_point_coms_n_consistent_complete_positions`: The number of complete and dialectically consistent
      positions that extend the final commitments.
    * :code:`fixed_point_coms_min_ax_bases`: Minimal-length axiomatic bases of :code:`fixed_point_coms` with
      sentences from :code:`fixed_point_coms` only. (Sentences :math:`\\mathcal{C}_a` are an axiomatic basis
      of :math:`\\mathcal{C}_1` (with sentences from :math:`\\mathcal{C}_2` only) iff :math:`\\mathcal{C}_a` dialectically
      implies :math:`\\mathcal{C}_1` and there is no proper subset of :math:`\\mathcal{C}_a` such that it implies
      :math:`\\mathcal{C}_1` (and :math:`\\mathcal{C}_a \subset \mathcal{C}_2`).)
    * :code:`n_fixed_point_coms_min_ax_base`: The size of minimal-length axiomatic bases of :code:`fixed_point_coms` with
      sentences from :code:`fixed_point_coms` only.
    * :code:`fixed_point_coms_min_ax_bases_theory`: A minimal-length subset :math:`\\mathcal{C}'` of the
      final commitments such that the final commitments are entailed by the theory and :math:`\\mathcal{C}'`.
      (Or equivalently: A smallest subset :math:`\\mathcal{C}'` within the commitments such that there is an axiomatic
      basis :math:`A` for the commitments :math:`\\mathcal{C}` with:
      :math:`A= \\mathcal{C}' \cup \\mathcal{T}'` and :math:`\\mathcal{T}'` a subset of the theory)
    * :code:`n_fixed_point_coms_min_ax_base_theory`: Number of propositions in each set
      in :code:`fixed_point_coms_min_ax_bases_theory`.
    * :code:`fixed_point_theory_axioms`: Axiomatic bases of the final theory.

    Additional items for all fixed points (these items are only generated if branches are created):

    * :code:`fp_full_re_state`:  A list of bools (:code:`List[bool]`) indicating whether the `fixed_points` are full
      RE-states (i.e. whether the dialectical closure of the theory is identical to the commitments). The order of the list
      represents the order in `fixed_points`.
    * :code:`fp_global_optimum`: A list of bools (:code:`List[bool]`) indicating whether the `fixed_points` are
      `global_optima`.

    Additional items for global optima:

    * :code:`global_optima`: All global optima as theory-commitments-tuples.
    * :code:`n_global_optima`: Number of global optima.
    * :code:`go_coms_consistent`: A list of bools (:code:`List[bool]`) indicating whether commitments of the `global_optima`
      are dialectically consistent. The order of the list represents the order in `global_optima`.
    * :code:`go_union_consistent`: A list of bools (:code:`List[bool]`) indicating whether the unions of a
       commitment-theory-tuple of the `global_optima` are dialectically consistent. I.e. whether the global optima
       are RE states.The order of the list represents the order in `global_optima`.
    * :code:`go_full_re_state`: A list of bools (:code:`List[bool]`) indicating whether the `global_optima` are full
      RE-states (i.e. whether the dialectical closure of the theory is identical to the commitments). The order of the list
      represents the order in `global_optima`.
    * :code:`go_fixed_point`: A list of bools (:code:`List[bool]`) indicating which global optima are fixed points
      (i.e. which global optima are reachable via a re-process). The order of the list
      represents the order in `global_optima`.
    * :code:`go_account`: The account of each global optimum as :code:`List[float]`. The order of the list represents the
      order in `global_optima`.
    * :code:`go_faithfulness`: The faithfulness of each global optimum as :code:`List[float]`. The order of the list
      represents the order in `global_optima`.

    Additional items for RE states and full RE states:

    * :code:`re_states`:
    * :code:`n_re_states`:
    * :code:`full_re_states`:
    * :code:`n_full_re_states`:
    """

    def __init__(self, arguments_list: List[List[List[int]]],
                 n_sentence_pool: int,
                 initial_commitments_list: List[Set[int]],
                 model_parameters_list: List[Dict] = None,
                 create_branches = False,
                 implementations: List[Dict] = None):

        super().__init__(arguments_list, n_sentence_pool, initial_commitments_list,
                         model_parameters_list,
                         create_branches,
                         implementations)
        _add_global_data_items(self)
        if create_branches:
            _add_full_branch_global_data_items(self)

    def init_re_start_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Extends :py:class:`SimpleEnsembleGenerator.init_re_final_fields`.

        Adds the following data objects that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'global_optima', 're_states' and 'full_re_states'.
        """

        super().init_re_start_fields(reflective_equilibrium, dialectical_structure)

        init_coms = reflective_equilibrium.state().initial_commitments()
        # Global optima
        global_optima = list(reflective_equilibrium.global_optima(init_coms))
        self.add_obj('global_optima', global_optima)

        # RE-states
        re_states = {(theory, commitments) for (theory, commitments) in global_optima
                     if dialectical_structure.is_consistent(BitarrayPosition.union({commitments, theory}))}
        self.add_obj('re_states', re_states)

        # full RE-states
        full_re_states = {(theory, commitments) for (theory, commitments) in global_optima
                          if dialectical_structure.closure(theory) == commitments}
        self.add_obj('full_re_states', full_re_states)

    def init_re_final_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_re_final_fields`.

        Adds the following data objects that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'fixed_point_coms_min_ax_bases' and 'fixed_point_coms_min_ax_bases_theory'.
        """
        fp_comms_min_ax_bases = _fp_comms_min_ax_bases(dialectical_structure,
                                                                    reflective_equilibrium.state().last_commitments())
        self.add_obj('fixed_point_coms_min_ax_bases', fp_comms_min_ax_bases)
        fp_comms_min_ax_bases_th = _fp_comms_min_ax_bases_given_theory(dialectical_structure,
                                                     reflective_equilibrium.state().last_commitments(),
                                                     reflective_equilibrium.state().last_theory())
        self.add_obj('fixed_point_coms_min_ax_bases_theory', fp_comms_min_ax_bases_th)

class LocalREEnsembleGenerator(SimpleEnsembleGenerator):
    """
    This class extends :class:`SimpleEnsembleGenerator` by adding the following data items that are produced by
    :py:func:`AbstractEnsembleGenerator.ensemble_items_iter`:

    * :code:`neigbourhood_depth`: The neighbourhood depth that is used to search for next commitments
        and theory candidates.
    """

    def __init__(self, arguments_list: List[List[List[int]]], n_sentence_pool: int,
                 initial_commitments_list: List[Set[int]], model_parameters_list: List[Dict] = None,
                 create_branches=False,
                 implementations: List[Dict] = None):
        super().__init__(arguments_list, n_sentence_pool, initial_commitments_list, model_parameters_list,
                         create_branches, implementations)
        _add_local_data_items(self)

class SimpleMultiAgentREContainer(REContainer):
    """An :py:class:`REContainer` for multi-agent ensembles.

    This container manages and executes model runs that are defined as an multi-agent ensemble. The container will
    execute for each particular point in time the next step of all model and will then proceed accordingly
    with the time point. Each model will be provided with the current model states of the other models by references
    to the other models via the argument :code:`other_model_runs`. This argument is accessible by overriding or
    extending the following methods in each model:

    * :py:func:`ReflectiveEquilibrium.theory_candidates`,
    * :py:func:`ReflectiveEquilibrium.commitment_candidates`,
    * :py:func:`ReflectiveEquilibrium.pick_theory_candidate`,
    * :py:func:`ReflectiveEquilibrium.pick_commitment_candidate` and
    * :py:func:`ReflectiveEquilibrium.finished`.

    """

    def __init__(self,
                 re_models: List[ReflectiveEquilibrium],
                 initial_commitments_list: List[Position],
                 max_re_length = 300):
        super().__init__(re_models)
        if len(re_models) != len(initial_commitments_list):
            raise ValueError("The containter must instantiated with a matching amount " +
                             "of models and initial commitments.")
        self.re_models = {index:re_models[index] for index in range(len(re_models))}
        self._max_re_length = max_re_length
        self._initial_commitments_list = initial_commitments_list

    def re_processes(self, re_models: List[ReflectiveEquilibrium] = None) -> List[ReflectiveEquilibrium]:
        # set initial states and update internal attributes if necessary
        if(re_models):
            self.re_models = {index:re_models[index] for index in range(len(re_models))}
        for key in range(len(self._initial_commitments_list)):
            self.re_models[key].set_initial_state(self._initial_commitments_list[key])
            # Might be used to update internal things. So far, we do not need it though.
            # self.re_models[key].update(self.re_models)

        active_process_keys = set(self.re_models.keys()) #set(range(len(self.re_models)))

        step_counter = 0
        while active_process_keys:
            step_counter += 1
            if step_counter > self._max_re_length:
                raise RuntimeWarning("Reached max loop count for processes without finishing all processes.")
            for key in active_process_keys.copy():
                re = self.re_models[key]
                if re.finished():
                    active_process_keys.remove(key)
                else:
                    #other_model_runs = self.re_models[0:index] + self.re_models[index+1:len(self.re_models)]
                    re.next_step(model_runs = self.re_models, container = self, self_key = key)

        return self.re_models

class MultiAgentEnsemblesGenerator(AbstractEnsembleGenerator):
    """ Ensemble generator base class for multi-agent (=interdependent) model runs.

    A class that provides iterators for interdependent model runs based on the given parameters of the constructor.
    One ensemble corresponds to a dialectical structure together with a set of agents (represented by their
    initial commitments). The generator will run all ensembles successively.

    This structure can be used to generate different ensembles.
    For instance, you can vary the number of agents for one particular dialectical structure by repeating the dialectical
    structure in the above list and vary the list of initial positions. All list must have the same length and the
    lenght of the lists corresponds to the number of ensembles (defined by them).
    Note that this design adheres to the
    following confinements: All agents in one ensemble share the implementing classes and their model parameters.

    The following data items are defined by default: 'ensemble_id' and 'ensemble_size' for each multi-agent ensemble.

    Args:
        arguments_list: A list of n dialectical structures as list of argument lists. Each dialectical structure
            corresponds an multi-agent ensemble.
        n_sentence_pool: Number of (unnegated) sentences in the sentence pool.
        initial_commitments_list: For each dialectical structure a list of initial commitments.
            (The initial commitments can be thought of as different agents.)
        tau_names: A list of names for the dialectical structures.
        model_parameters_list: For each dialectical structure a specification of model parameters as dictionary
            that can be set via :py:func:`ReflectiveEquilibrium.set_model_parameters`.
        implementations: A list of dicts, each representing a specific implementation. Each dict should contain
            strings for the keys 'module_name', 'position_class_name', 'dialectical_structure_class_name'
            and 'reflective_equilibrium_class_name'. (If these classes are located in different modules, you can,
            alternatively, specify modules for each class by using the keys 'position_module_name',
            'dialectical_structure_module_name' and 'reflective_equilibrium_module_name')
    """
    def __init__(self,
                 arguments_list: List[List[List[int]]],
                 n_sentence_pools: List[int],
                 initial_commitments_list: List[List[Set[int]]],
                 tau_names: List[str] = None,
                 initial_commitments_names: List[str] = None,
                 implementations: List[Dict] = None,
                 model_parameters_list: List[Dict] = None):
        super().__init__()
        self.arguments_list = arguments_list
        self.n_sentence_pools = n_sentence_pools
        self.initial_commitments_list = initial_commitments_list
        self.model_parameters_list = model_parameters_list
        self.tau_names = tau_names
        self.initial_commitments_names = initial_commitments_names

        if implementations is None:
            self.implementations = _fill_module_names([{'tau_module_name': 'tau',
                                                        'position_class_name': 'StandardPosition',
                                                        'dialectical_structure_class_name': 'DAGDialecticalStructure',
                                                        'rethon_module_name': 'rethon',
                                                        'reflective_equilibrium_class_name': 'StandardGlobalReflectiveEquilibrium'
                                                        }])
        else:
            self.implementations = _fill_module_names(implementations)
        self.ensemble_counter = 0

        self.add_item('ensemble_id', lambda x: x.get_obj('ensemble_id'))
        self.add_item('ensemble_size', lambda x: x.get_obj('ensemble_size'))
        self.add_item('agents_name', lambda x: x.get_obj('agents_name'))

    def ensemble_iter(self) -> Iterator[ReflectiveEquilibrium]:
        """ Iterator through the re processes.

        An ensemble iterator through all model runs of all multi-agents ensembles as defined by the class
        attributes (implements :py:func:`AbstractEnsembleGenerator.ensemble_iter`). Model runs that belong to
        the same multi-agent ensemble can be identified via their ensemble id, which can be accessed via
        :code:`get_obj('ensemble_id')`.
        """

        # iterating through ensembles
        for i in range(len(self.arguments_list)):
            # instantiating dialectical structure
            ds_class_ = getattr(importlib.import_module(self.implementations[i]['dialectical_structure_module_name']),
                                self.implementations[i]['dialectical_structure_class_name'])
            tau_name = None if self.tau_names is None else self.tau_names[i]
            ds = ds_class_.from_arguments(self.arguments_list[i], self.n_sentence_pools[i], tau_name)
            self.current_dialectical_structure = ds
            self.init_tau_fields(ds)

            # instantiating res
            reflective_equilibrium_class_ = getattr(importlib.import_module(
                self.implementations[i]['reflective_equilibrium_module_name']),
                self.implementations[i]['reflective_equilibrium_class_name'])

            # one re insance for each agents in the ensemble
            res = [reflective_equilibrium_class_(ds) for pos in self.initial_commitments_list[i]]

            # getting specified model parameters for this ensemble
            if self.model_parameters_list:
                # for j in range(len(self.model_parameters_list)):
                #     res[j].reset_model_parameters(self.model_parameters_list[j])
                for j in range(len(res)):
                    res[j].reset_model_parameters(self.model_parameters_list[i])

            # instantiating initial coms (=agents) for the ensemble
            agents = []
            for pos_as_set in self.initial_commitments_list[i]:
                pos_class_ = getattr(importlib.import_module(self.implementations[i]['position_module_name']),
                                     self.implementations[i]['position_class_name'])
                pos = pos_class_.from_set(pos_as_set, self.n_sentence_pools[i])
                agents.append(pos)

            # ToDo: It should be possible for the user to dynamically provide an REContainer
            multi_agent_container = SimpleMultiAgentREContainer(res, agents)
            multi_agent_container.re_processes()

            self.current_ensemble_states = [re.state() for re in res]
            if self.initial_commitments_names:
                self.init_ensemble_fields(self.current_ensemble_states, ds,
                                          self.initial_commitments_names[i])
            else:
                self.init_ensemble_fields(self.current_ensemble_states, ds)
            for re in res:
                self.current_reflective_equilibrium = re
                self.current_initial_commitments = re.state().initial_commitments()
                self.current_state = re.state()
                self.init_re_start_fields(re, ds)
                self.init_re_final_fields(re, ds)
                yield re

    def ensemble_items_to_csv(self, output_file_name: str, output_dir_name: str, archive=False,
                              save_preliminary_results: bool = False, preliminary_results_interval: int = 500,
                              append = False):
        """Extends :py:class:`AbstractEnsembleGenerator.ensemble_items_to_csv.`

        If :code:`append` is set to :code:`True` the method will make sure that different ensembles have different
        ensemble ids (which are saved via an extra column for each model run).
        """
        output_file = path.join(output_dir_name, output_file_name)
        # if we append and the file already exists check for existing ensemble ids
        if path.exists(output_file) and append:
            re_data = pd.read_csv(output_file)
            self.ensemble_counter = max(set(re_data['ensemble_id']))+1

        super().ensemble_items_to_csv(output_file_name, output_dir_name, archive, save_preliminary_results,
                                      preliminary_results_interval, append)

    def init_ensemble_fields(self,
                             re_states: List[REState],
                             dialectical_structure: DialecticalStructure,
                             init_commitments_name: [str, None] = None):
        """Overrides :py:func:`AbstractEnsembleGenerator.init_ensemble_fields`

        Adds for every multi-agent ensemble the following data objects: 'ensemble_id' and 'ensemble_size'.
        """
        super().init_ensemble_fields(re_states, dialectical_structure)
        self.add_obj('ensemble_id', self.ensemble_counter)
        self.ensemble_counter += 1
        self.add_obj('ensemble_size', len(re_states))
        self.add_obj('agents_name', init_commitments_name)

class SimpleMultiAgentEnsemblesGenerator(MultiAgentEnsemblesGenerator):
    """A :py:class:`MultiAgentEnsembleGenerator` with predefined data items.

    This class extends :py:class:`MultiAgentEnsembleGenerator` by adding data items that are produced by
    :py:func:`AbstractEnsembleGenerator.ensemble_items_iter`. The ensemble generator is initiated with the same
    data fields as the :py:class:`SimpleEnsembleGenerator` (except data fields that are only created in the
    case that the :py:class:`SimpleEnsembleGenerator` runs all branches).
    """

    def __init__(self, arguments_list: List[List[List[int]]], n_sentence_pools: List[int],
                 initial_commitments_list: List[List[Set[int]]],
                 tau_names: List[str] = None,
                 initial_commitments_names: List[str] = None,
                 implementations: List[Dict] = None,
                 model_parameters_list: List[Dict] = None):
        super().__init__(arguments_list, n_sentence_pools, initial_commitments_list,
                         tau_names,
                         initial_commitments_names,
                         implementations,
                         model_parameters_list)
        _add_simple_data_items(self)

    def init_tau_fields(self, tau: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_tau_fields`.

        Adds the following data objects that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'ds_infer_dens', 'n_premises', 'principles', 'tau_truths' and 'tau_falsehoods'.
        """
        self.add_obj('ds_infer_dens', inferential_density(tau))
        self.add_obj('n_premises', [len(arg) for arg in tau.get_arguments()])
        self.add_obj('principles', get_principles(tau.get_arguments()))
        tau_truths = tau.closure(BitarrayPosition(set(),
                                                  tau.sentence_pool().size()))
        self.add_obj('tau_truths', tau_truths)
        self.add_obj('tau_falsehoods', BitarrayPosition({-prop for prop in tau_truths.as_set()},
                                                        tau.sentence_pool().size()))


    def init_re_start_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
                             dialectical_structure: DialecticalStructure):
        """Overrides :py:class:`AbstractEnsembleGenerator.init_re_start_fields`.

        Adds the following data object that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
        'init_com_min_ax_bases'.
        """
        init_coms = reflective_equilibrium.state().initial_commitments()
        if dialectical_structure.is_consistent(init_coms):
            init_com_min_ax_bases =  dialectical_structure.axioms(init_coms,
                                                        init_coms.subpositions())
            # init_com_min_ax_bases might be None
            if init_com_min_ax_bases is not None:
                init_com_min_ax_bases = _get_min_sets([axioms.as_set() for axioms in
                                                       init_com_min_ax_bases])
            else:
                init_com_min_ax_bases = np.nan
        else:
            init_com_min_ax_bases = np.nan
        self.add_obj('init_com_min_ax_bases', init_com_min_ax_bases)

    # def init_re_final_fields(self, reflective_equilibrium: ReflectiveEquilibrium,
    #                          dialectical_structure: DialecticalStructure):
    #     """Overrides :py:class:`AbstractEnsembleGenerator.init_re_final_fields`.
    #
    #     Adds the following data objects that can be accessed via :py:func:`AbstractEnsembleGenerator.get_obj`:
    #     'fixed_point_coms_min_ax_bases' and 'fixed_point_coms_min_ax_bases_theory'.
    #     """
    #     fp_comms_min_ax_bases = _fp_comms_min_ax_bases(dialectical_structure,
    #                                                                 reflective_equilibrium.state().last_commitments())
    #     self.add_obj('fixed_point_coms_min_ax_bases', fp_comms_min_ax_bases)
    #     fp_comms_min_ax_bases_th = _fp_comms_min_ax_bases_given_theory(dialectical_structure,
    #                                                  reflective_equilibrium.state().last_commitments(),
    #                                                  reflective_equilibrium.state().last_theory())
    #     self.add_obj('fixed_point_coms_min_ax_bases_theory', fp_comms_min_ax_bases_th)



# Classes using this method to add data item must provide the following key-object pairs:
# 'ds_infer_dens', 'n_premises', 'tau_truths', 'principles',
# 'tau_falsehoods', 'init_com_min_ax_bases'

# todo: rename 'ds_x' to 'tau_x'
def _add_simple_data_items(ensemble_generator: AbstractEnsembleGenerator):
    ensemble_generator.add_item('model_name',
                                lambda x: x.reflective_equilibrium().model_name())
    ensemble_generator.add_item('ds',
                                lambda x: x.dialectical_structure().get_arguments())  # dialectical structure
    ensemble_generator.add_item('tau_name',
                                lambda x: x.dialectical_structure().get_name())

    # number of unnegated sentences (half the full size)
    ensemble_generator.add_item('n_sentence_pool',
                                lambda x: x.dialectical_structure().sentence_pool().size())
    ensemble_generator.add_item('ds_arg_size',
                                lambda x: len(x.dialectical_structure().get_arguments()))

    # FEATURES OF DIALECTICAL STRUCTURE
    ensemble_generator.add_item('ds_infer_dens',
                                lambda x: x.get_obj('ds_infer_dens'))
    ensemble_generator.add_item('ds_n_consistent_complete_positions',
                                lambda x: x.dialectical_structure().n_complete_extensions())
    ensemble_generator.add_item('ds_mean_prem',
                                lambda x: statistics.mean(x.get_obj('n_premises')))
    ensemble_generator.add_item('ds_variance_prem',
                                lambda x: statistics.variance(x.get_obj('n_premises')))
    ensemble_generator.add_item('tau_truths',
                                lambda x: x.get_obj('tau_truths'))
    # A sentence counts as a principle iff it occurs in at least one argument as premise and it or its negation
    # does not occur as a conclusion in an argument. The multiplicity counts in how many argument the sentence occurs
    # as premise.
    ensemble_generator.add_item('principles',
                                lambda x: x.get_obj('principles'))

    ####### PARAMETERS OF RE-PROCESS ##############################################################
    ensemble_generator.add_item('account_penalties',
                                lambda x: x.reflective_equilibrium().model_parameter('account_penalties'))
    ensemble_generator.add_item('faithfulness_penalties',
                                lambda x: x.reflective_equilibrium().model_parameter('faithfulness_penalties'))
    ensemble_generator.add_item('weight_account',
                                lambda x: x.reflective_equilibrium().model_parameter('weights')['account'])
    ensemble_generator.add_item('weight_systematicity',
                                lambda x: x.reflective_equilibrium().model_parameter('weights')['systematicity'])
    ensemble_generator.add_item('weight_faithfulness',
                                lambda x: x.reflective_equilibrium().model_parameter('weights')['faithfulness'])
    ###### PROCESS-FEATURES: ######################################################################
    ensemble_generator.add_item('init_coms',
                                lambda x: x.initial_commitments().as_set())
    ensemble_generator.add_item('init_coms_size',
                                lambda x: x.initial_commitments().size())
    ensemble_generator.add_item('init_coms_n_tau_truths',
                                lambda x: len(BitarrayPosition.intersection({x.initial_commitments(),
                                                                             x.get_obj('tau_truths')}).as_set()))
    ensemble_generator.add_item('init_coms_n_tau_falsehoods',
                                lambda x: len(
                                    BitarrayPosition.intersection({x.initial_commitments(),
                                                                   x.get_obj('tau_falsehoods')}).as_set()))

    # ToDo: behavior so far unclear - what to do with (minimally) inconsistent positions (ex falso quodlibet?)
    ensemble_generator.add_item('init_coms_n_consistent_complete_positions',
                                lambda x: x.dialectical_structure().n_complete_extensions(x.initial_commitments())
                                if x.dialectical_structure().is_consistent(x.initial_commitments()) else 0)
    ensemble_generator.add_item('init_coms_dia_consistent',
                                lambda x: x.dialectical_structure().is_consistent(x.initial_commitments()))
    ensemble_generator.add_item('init_coms_closed',
                                lambda x: x.dialectical_structure().is_closed(x.initial_commitments())
                                if x.dialectical_structure().is_consistent(x.initial_commitments()) else np.nan)
    ensemble_generator.add_item('fixed_point_coms',
                                lambda x: x.state().last_commitments().as_set())
    ensemble_generator.add_item('fixed_point_coms_size',
                                lambda x: x.state().last_commitments().size()),
    ensemble_generator.add_item('fixed_point_coms_n_tau_truths',
                                lambda x: len(BitarrayPosition.intersection({x.state().last_commitments(),
                                                                             x.get_obj('tau_truths')}).as_set()))
    ensemble_generator.add_item('fixed_point_coms_n_tau_falsehoods',
                                lambda x: len(BitarrayPosition.intersection({x.state().last_commitments(),
                                                                             x.get_obj('tau_falsehoods')}).as_set()))

    ensemble_generator.add_item('fixed_point_coms_closed',
                                lambda x: x.dialectical_structure().is_closed(
                                    x.state().last_commitments()) if x.dialectical_structure().is_consistent(
                                    x.state().last_commitments()) else np.nan)
    ensemble_generator.add_item('fixed_point_coms_consistent',
                                lambda x: x.dialectical_structure().is_consistent(x.state().last_commitments()))
    ensemble_generator.add_item('fixed_point_coms_n_consistent_complete_positions',
                                lambda x: x.dialectical_structure().n_complete_extensions(
                                    x.state().last_commitments()) if x.dialectical_structure().is_consistent(
                                    x.state().last_commitments()) else 0)

    ensemble_generator.add_item('fixed_point_theory',
                                lambda x: (x.state().last_theory().as_set()))
    ensemble_generator.add_item('fixed_point_theory_closure',
                                lambda x: x.dialectical_structure().closure(x.state().last_theory()).as_set())
    # minimal-length axiomatic bases of init_coms with sentences from init_coms only
    ensemble_generator.add_item('init_coms_min_ax_bases',
                                lambda x: x.get_obj('init_com_min_ax_bases'))
    ensemble_generator.add_item('n_init_coms_min_ax_base',
                                lambda x: np.nan if (x.get_obj('init_com_min_ax_bases') is np.nan) else
                                len(x.get_obj('init_com_min_ax_bases')[0]))
    def achievements(re, state):
        return [0] + [re.achievement(state.evolution[x],
                                     state.evolution[y],
                                     state.initial_commitments()) for x, y in
                      list(zip([i - (i % 2) for i in range(1, len(state))],
                               [i - (i % 2) + 1 for i in range(len(state) - 1)]))]

    ensemble_generator.add_item('achievements_evolution',
                                lambda x: achievements(x.reflective_equilibrium(), x.state()))

    # whether the union of final theory & commitments is dialectically consistent
    ensemble_generator.add_item('fixed_point_dia_consistent',
                                lambda x: x.dialectical_structure().is_consistent(
                                    BitarrayPosition.union({x.state().last_commitments(), x.state().last_theory()})))
    ensemble_generator.add_item('init_final_coms_simple_hamming',
                                lambda x: len(BitarrayPosition.union({x.state().last_commitments(), x
                                                                     .state().initial_commitments()}).as_set().
                                              difference(BitarrayPosition.intersection({x.state().last_commitments(),
                                                                                        x.initial_commitments()}).as_set())))
    # SetBasedPosition.union({pos1, pos2}).difference(SetBasedPosition.intersection({pos1, pos2})).size()
    # hamming distance as defined in BBB with d_3 = 1, d_2 = 1, d_1 = 1, d_0 = 0
    ensemble_generator.add_item('init_final_coms_hamming',
                                lambda x: x.reflective_equilibrium().hamming_distance(x.initial_commitments(),
                                                                                      x.state().last_commitments(),
                                                                                      [0, 1, 1, 1]))

    # amount of contradictions: number of sentences whose contradictions are in both positions
    # i.e. hamming distance as defined in BBB with d_3 = 1, d_2 = 0, d_1 = 0, d_0 = 0
    ensemble_generator.add_item('init_final_coms_contradictions',
                                lambda x: x.reflective_equilibrium().hamming_distance(x.initial_commitments(),
                                                                                      x.state().last_commitments(),
                                                                                      [0, 0, 0, 1]))
    # amount of expansions (without expansions that lead to contradictions or identities)
    # i.e. hamming distance as defined in BBB with d_3 = 0, d_2 = 0, d_1 = 1, d_0 = 0
    ensemble_generator.add_item('init_final_coms_expansions',
                                lambda x: x.reflective_equilibrium().hamming_distance(x.initial_commitments(),
                                                                                      x.state().last_commitments(),
                                                                                      [0, 1, 0, 0]))
    # amount of contractions
    # i.e. hamming distance as defined in BBB with d_3 = 0, d_2 = 1, d_1 = 0, d_0 = 0
    ensemble_generator.add_item('init_final_coms_contractions',
                                lambda x: x.reflective_equilibrium().hamming_distance(x.initial_commitments(),
                                                                                      x.state().last_commitments(),
                                                                                      [0, 0, 1, 0]))
    # amount of identities (without identities that are associated with contradictions )
    # !: counts also sentences on which both positions are indifferent
    # i.e. hamming distance as defined in BBB with d_3 = 0, d_2 = 0, d_1 = 0, d_0 = 1
    ensemble_generator.add_item('init_final_coms_identities',
                                lambda x: x.reflective_equilibrium().hamming_distance(x.initial_commitments(),
                                                                                      x.state().last_commitments(),
                                                                                      [1, 0, 0, 0]))
    ensemble_generator.add_item('random_choices',
                                lambda x: x.state().indices_of_non_empty_alternatives())
    ensemble_generator.add_item('n_random_choices',
                                lambda x: len(x.state().indices_of_non_empty_alternatives()))
    ensemble_generator.add_item('coms_evolution',
                                lambda x: [pos.as_set() for pos in x.state().commitments_evolution()])
    ensemble_generator.add_item('theory_evolution',
                                lambda x: [pos.as_set() for pos in x.state().theory_evolution()])

    ensemble_generator.add_item('process_length',
                                lambda x: len(x.state()))

# Classes using this method to add data item must provide the following key-object pairs:
# 'n_branches', 'fixed_points'

def _add_full_branch_data_items(ensemble_generator: AbstractEnsembleGenerator):
    ######### PROCESS-INDEPENDENT FEATURES ###############################################
    ensemble_generator.add_item('n_branches',
                                lambda x: x.get_obj('n_branches'))
    # FIXED POINTS
    ensemble_generator.add_item('fixed_points',
                                lambda x: [(theory.as_set(), commitments.as_set())
                                           for (theory, commitments) in x.get_obj('fixed_points')])
    ensemble_generator.add_item('n_fixed_points',
                                lambda x: len(x.get_obj('fixed_points')))
    ensemble_generator.add_item('fp_coms_consistent',
                                lambda x: [x.dialectical_structure().is_consistent(commitments)
                                           for (theory, commitments) in x.get_obj('fixed_points')])
    ensemble_generator.add_item('fp_union_consistent',
                                lambda x: [x.dialectical_structure().is_consistent(
                                    BitarrayPosition.union({commitments, theory}))
                                           for (theory, commitments) in x.get_obj('fixed_points')])
    ensemble_generator.add_item('fp_account',
                                lambda x: [x.reflective_equilibrium().account(commitments, theory)
                                           for (theory, commitments) in x.get_obj('fixed_points')])
    ensemble_generator.add_item('fp_faithfulness',
                                lambda x: [x.reflective_equilibrium().faithfulness(commitments,
                                                                                   x.initial_commitments())
                                           for (theory, commitments) in x.get_obj('fixed_points')])

# Classes using this method to add data item must provide the following key-object pairs:
# 'global_optima', 're_states', 'full_re_states',
# 'fixed_point_coms_min_ax_bases', 'fixed_point_coms_min_ax_bases_theory',

def _add_global_data_items(ensemble_generator: AbstractEnsembleGenerator):
    ensemble_generator.add_item('fixed_point_is_global_optimum',
                  lambda x:(x.state().last_theory(), x.state().last_commitments()) in x.get_obj('global_optima'))
    ensemble_generator.add_item('fixed_point_is_re_state',
                  lambda x:(x.state().last_theory(), x.state().last_commitments()) in x.get_obj('re_states'))
    ensemble_generator.add_item('fixed_point_is_full_re_state',
                  lambda x: (x.state().last_theory(),
                                     x.state().last_commitments()) in x.get_obj('full_re_states'))

    # PROCESS-INDEPENDENT FEATURES:

    # global optima as theory-commitments-tupel
    ensemble_generator.add_item('global_optima',
                  lambda x:  [(theory.as_set(), commitments.as_set())
                              for (theory, commitments) in x.get_obj('global_optima')])
    ensemble_generator.add_item('n_global_optima',
                  lambda x:  len(x.get_obj('global_optima')))
    ensemble_generator.add_item('go_coms_consistent',
                  lambda x: [x.dialectical_structure().is_consistent(commitments)
                             for (theory, commitments) in x.get_obj('global_optima')])
    ensemble_generator.add_item('go_union_consistent',
                  lambda x: [x.dialectical_structure().is_consistent(BitarrayPosition.union({commitments, theory}))
                             for (theory, commitments) in x.get_obj('global_optima')])
    ensemble_generator.add_item('go_full_re_state',
                  lambda x: [global_optimum in x.get_obj('full_re_states')
                             for global_optimum in x.get_obj('global_optima')])
    ensemble_generator.add_item('go_account',
                  lambda x: [x.reflective_equilibrium().account(commitments,theory)
                             for (theory, commitments) in x.get_obj('global_optima')])
    ensemble_generator.add_item('go_faithfulness',
                  lambda x: [x.reflective_equilibrium().faithfulness(commitments, x.initial_commitments())
                             for (theory, commitments) in x.get_obj('global_optima')])

    # RE states & full RE states
    ensemble_generator.add_item('re_states',
                  lambda x:  [(theory.as_set(), commitments.as_set())
                              for (theory, commitments) in x.get_obj('re_states')])
    ensemble_generator.add_item('n_re_states',
                  lambda x:  len(x.get_obj('re_states')))
    ensemble_generator.add_item('full_re_states',
                  lambda x:  [(theory.as_set(), commitments.as_set())
                              for (theory, commitments) in x.get_obj('full_re_states')])
    ensemble_generator.add_item('n_full_re_states',
                  lambda x:  len(x.get_obj('full_re_states')))
    # minimal-length axiomatic bases of final coms with sentences from final coms only
    ensemble_generator.add_item('fixed_point_coms_min_ax_bases',
                                lambda x: x.get_obj('fixed_point_coms_min_ax_bases'))
    ensemble_generator.add_item('n_fixed_point_coms_min_ax_base',
                                lambda x: np.nan if (x.get_obj('fixed_point_coms_min_ax_bases') is np.nan)
                                else len(x.get_obj('fixed_point_coms_min_ax_bases')[0]))

    ensemble_generator.add_item('fixed_point_coms_min_ax_bases_theory',
                                lambda x: x.get_obj('fixed_point_coms_min_ax_bases_theory'))
    ensemble_generator.add_item('n_fixed_point_coms_min_ax_base_theory',
                                lambda x: np.nan if (x.get_obj('fixed_point_coms_min_ax_bases_theory') is np.nan)
                                else len(x.get_obj('fixed_point_coms_min_ax_bases_theory')[0]))
    ensemble_generator.add_item('fixed_point_theory_axioms',
                                lambda x: [axioms.as_set() for axioms in
                                           x.dialectical_structure().axioms(x.state().last_theory())])

# Classes using this method to add data item must provide the following key-object pairs:
# 'global_optima', 're_states', 'full_re_states', 'fixed_points'
def _add_full_branch_global_data_items(ensemble_generator: AbstractEnsembleGenerator):

    ensemble_generator.add_item('go_fixed_point',
                  lambda x:  [global_optimum in x.get_obj('fixed_points')
                              for global_optimum in x.get_obj('global_optima')])

    # FIXED POINTS
    ensemble_generator.add_item('fp_full_re_state',
              lambda x: [fixed_point in x.get_obj('full_re_states')
                         for fixed_point in x.get_obj('fixed_points')])
    #'fixed_points_that_are_global_optima': fixed_points_that_are_global_optima,
    ensemble_generator.add_item('fp_global_optimum',
                  lambda x:  [fixed_point in x.get_obj('global_optima')
                              for fixed_point in x.get_obj('fixed_points')])

def _add_local_data_items(ensemble_generator: AbstractEnsembleGenerator):
    ####### PARAMETERS OF RE-PROCESS ##############################################################
    ensemble_generator.add_item('neigbourhood_depth',
                  lambda x: x.reflective_equilibrium().model_parameter('neighbourhood_depth'))


def _fp_comms_min_ax_bases(dia_structure, final_commitments):
    if dia_structure.is_consistent(final_commitments):
        return _get_min_sets([axioms.as_set() for axioms in dia_structure.axioms(final_commitments,
                                                                               final_commitments.subpositions())])
    return np.nan

def _fp_comms_min_ax_bases_given_theory(dia_structure, final_commitments, final_theory):
    # ToDo: rewrite "casting" if we have difference function of positions (axioms.as_set())
    if dia_structure.is_consistent(BitarrayPosition.union({final_commitments, final_theory})):
        return [axioms for axioms in _min_ax_bases_com_given_theory(dia_structure,
                                                                       final_commitments,
                                                                       final_theory)]
    return np.nan

# returns from a list of lists or sets those sets with minimal length (as list of lists/sets)
# ToDo: a little bit cumbersome; perhaps a simple for loop will do
def _get_min_sets(sets, min_sets=None):
    if not min_sets and len(sets) == 1:
        return sets
    elif not min_sets:
        return _get_min_sets(sets[1:], [sets[0]])
    elif len(sets) == 0:
        return min_sets
    elif len(sets[0]) == len(min_sets[0]):
        min_sets.append(sets[0])
        return _get_min_sets(sets[1:], min_sets)
    elif len(sets[0]) < len(min_sets[0]):
        return _get_min_sets(sets[1:], [sets[0]])
    else:
        return _get_min_sets(sets[1:], min_sets)


# searches for the smallest subset C* within the commitments C such that there is an axiomatic
# basis AB for the commitments C with: AB =  C* U T* (with T* being a subset of the theory)
# ToDo: Does not check for consistency of coms and theory (that is, will throw an uncatched error in this case)
def _min_ax_bases_com_given_theory(diastructure, commitments, theory):
    # ToDo: rewrite when we have difference functions for Positions
    axioms_sets_coms_given_theory = diastructure.axioms(commitments,
                                                        BitarrayPosition.union({commitments, theory}).subpositions())
    res = []
    for axioms in axioms_sets_coms_given_theory:
        if len(res) > 0:
            if len(axioms.as_set().difference(theory.as_set())) < len(res[0]): #res[0].size():
                res = [axioms.as_set().difference(theory.as_set())]
            elif len(axioms.as_set().difference(theory.as_set())) == len(res[0]): #res[0].size():
                res.append(axioms.as_set().difference(theory.as_set()))
        else:
            res = [axioms.as_set().difference(theory.as_set())]
    return res

def _conditional_friedman_consistence(diastructure, commitments, theory):
    return _min_ax_bases_com_given_theory(diastructure, commitments, theory)[0].size()

def _fill_module_names(implementations: List[Dict]) -> Dict:
    for impl in implementations:
        if 'position_module_name' not in impl.keys():
            impl['position_module_name'] = impl['tau_module_name']
        if 'dialectical_structure_module_name' not in impl.keys():
            impl['dialectical_structure_module_name'] = impl['tau_module_name']
        if 'reflective_equilibrium_module_name' not in impl.keys():
            impl['reflective_equilibrium_module_name'] = impl['rethon_module_name']

    return implementations