"""
A collection of convenient helper-functions.

"""
from __future__ import annotations

from tau import Position
from tau.util import TauJSONEncoder, tau_decoder
from .base import ReflectiveEquilibrium, StandardReflectiveEquilibrium
from rethon import REState

import os
from ast import literal_eval
import csv
import importlib
from typing import List, Tuple, Iterator, Dict
from json import dumps, dump, loads, load


def re_from_text_file(dir: str, file_name: str,
                      re_implementations: List[Tuple[str, str]] = [('rethon', 'BitarrayReflectiveEquilibrium')],
                      ds_module_name: str = 'tau',
                      ds_class_name: str = 'BitarrayDialecticalStructure',
                      position_module_name: str = 'tau',
                      position_class_name: str = 'BitarrayPosition') -> List[Tuple[ReflectiveEquilibrium, List[Position]]]:
    """Creates a list of RE classes  from a text-file.

    The text file consists of a list (i.e. [...]) of dictionaries (i.e. {...})
    and commented lines used for the description of the example (starting with #).
    The dictionaries have to contain comma separated entries for the number of unnegated
    sentences (key: 'n', e.g. 'n':7) and the arguments (key: 'arguments') as a list of lists of numbers
    (e.g.'arguments':[[1,-3], [2,3,5]]).

    Optionally, the dictionary may contain entries 'account_penalties', 'faithfulness_penalties' and
    'weights' if non-standard values should be considered. Finally, an entry of 'initial_commitments'
    may be provided, which is returned as a list of positions in the desired implementation.

    A simple schematic example is:

    .. code:: python

        [
        # the standard examples (cases A,B,C and D) contain only arguments with one premise.
        {
            'n': 7,
            'arguments': [
                [1,3],
                [1,4],
                [1,5],
                [1,-6],
                [2,-4],
                [2,5],
                [2,6],
                [2,7]
            ],
            'initial_commitments': [{3,4,5}, {2,3,4,5}, {3,4,5,6,7}, {3,4,5,-6,7}],
            'weights' : {'account': 0.35, 'systematicity': 0.55, 'faithfulness': 0.1},
            'account_penalties' : [0, 0.3, 1, 1],
            'faithfulness_penalties' : [0, 0, 1, 1]
        },
        { # some other dialectical structure ...
        }
        ]

    :param dir: Directory of the text-file.
    :param file_name: Name of the text-file to be used.
    :param re_implementations: String-representation of the implementation to be used (defined by module- and class name).
    :param ds_module_name: Modulename of the implementation of :code:`DialecticalStructure` to be used.
    :param ds_class_name: Classname of the implementation of :code:`DialecticalStructure` to be used.
    :param position_module_name: Module name of the implementation of :code:`Position` to be used.
    :param position_class_name: Class name of the implementation of :code:`Position` to be used.
    :return: List of the form :code:`[(ReflectiveEquilibrium,[Position,Position,...]),(ReflectiveEquilibrium,[Position,Position,...]),...]`
        with the inner lists of :code:`Position` possibly being :code:`None`.

    """

    with open(os.path.join(dir, file_name), 'r') as data:
        re_dict_list = literal_eval(data.read())

    re_list = []
    ds_class_ = getattr(importlib.import_module(ds_module_name), ds_class_name)
    position_class_ = getattr(importlib.import_module(position_module_name), position_class_name)
    for re_dict in re_dict_list:
        # if implementation == "SetBased":
        #
        #     dia = SetBasedDialecticalStructure(re_dict['n'])
        #     dia.add_arguments(re_dict['arguments'])
        #     re = SetBasedReflectiveEquilibrium(dia)
        #
        # elif implementation == "Bitarray":
        #     dia = BitarrayDialecticalStructure(re_dict['n'])
        #     dia.add_arguments(re_dict['arguments'])
        #     re = BitarrayReflectiveEquilibrium(dia)
        #
        # else:
        #     raise ValueError("Unknown implementation")
        dia = ds_class_(re_dict['n'])
        dia.add_arguments(re_dict['arguments'])

        for re_module_name, re_class_name in re_implementations:
            re_class_ = getattr(importlib.import_module(re_module_name), re_class_name)
            re = re_class_(dia)

            # optionally, set weights and penalties
            if 'weights' in re_dict.keys():
                re.weights = re_dict['weights']

            if 'account_penalties' in re_dict.keys():
                re.account_penalties = re_dict['account_penalties']

            if 'faithfulness_penalties' in re_dict.keys():
                re.faithfulness_penalties = re_dict['faithfulness_penalties']

            # ToDo: Alternatively, we could integrate initial commitments into the RE
            #  class as a variable and define a method to set/change them
            if 'initial_commitments' in re_dict.keys():
                # if implementation == "SetBased":
                #     re_list.append([re, [SetBasedPosition(pos) for pos in re_dict['initial_commitments']]])
                # else:
                #     re_list.append([re, [BitarrayPosition(SetBasedPosition(pos).
                #                                           as_bitarray(n_unnegated_sentence_pool=re_dict['n']))
                #                          for pos in re_dict['initial_commitments']]])
                re_list.append([re, [position_class_(pos, re_dict['n']) for pos in re_dict['initial_commitments']]])
            else:
                # re_list.append([re])
                re_list.append((re, None))

    return re_list

def re_weight_variations(re, initial_commitments, DIR, fileName, resolution=20):
    """Simulate RE processes with
    todo
    """
    with open(os.path.join(DIR, fileName), "w") as fi:
        writer = csv.writer(fi, delimiter=',')

        # Headers
        to_file = ['initial_coms', 'account', 'systematicity', 'faithfulness',
                   'RE_coms', 'RE_the', 'achievements', 'random_choices', 'com_evo',
                   'the_evo']
        writer.writerow(to_file)

        for i in range(1, resolution):
            for j in range(1, (resolution - i) + 1):
                a = i / resolution
                s = j / resolution
                f = 1 - (a + s)

                re.set_weights(a, s, f)
                # simulate RE process with
                results = re.re_process(initial_commitments)

                to_file2 = [initial_commitments.as_set(),
                            re.weights['account'],
                            re.weights['systematicity'],
                            re.weights['faithfulness'],
                            results['commitments'].as_set(),
                            results['theory'].as_set(),
                            results['achievements'],
                            results['random_choices'],
                            results['commitments_evolution'],
                            results['theory_evolution']]
                writer.writerow(to_file2)

    return None


def varied_alphas(alpha_resolution, with_extremes=False) -> Iterator[List[float]]:
    """A list of alpha values that can be used as model parameters."""
    if with_extremes:
        for i in range(0, alpha_resolution):
            for j in range(0, (alpha_resolution - i)):
                alpha_account = i / (alpha_resolution - 1)
                alpha_systematicity = j / (alpha_resolution - 1)
                alpha_faithfulness = 1 - (alpha_account + alpha_systematicity)
                yield [alpha_account, alpha_systematicity, alpha_faithfulness]
    else:
        for i in range(1, alpha_resolution + 1):
            for j in range(1, (alpha_resolution + 1 - i)):
                alpha_account = i / (alpha_resolution + 1)
                alpha_systematicity = j / (alpha_resolution + 1)
                alpha_faithfulness = 1 - (alpha_account + alpha_systematicity)
                yield [alpha_account, alpha_systematicity, alpha_faithfulness]


def standard_model_params_varied_alphas(alpha_resolution, with_extremes=False) -> Iterator[Dict]:
    """Standard model parameters with varied alphas.

    Convenience method that uses :py:func:`varied_alphas` to generate a list of model parameters for the
    standard model (see :py:class:`StandardReflectiveEquilibrium`) as dict. Alphas are
    varied; all other parameters a set to default (see :py:class:`basics.ReflectiveEquilibrium` for details about
    model parameters).
    """
    alpha_values = list(varied_alphas(alpha_resolution=alpha_resolution,
                                      with_extremes=with_extremes))

    for alpha_account, alpha_systematicity, alpha_faithfulness in alpha_values:
        model_parameters = StandardReflectiveEquilibrium.default_model_parameters()
        model_parameters['weights'] = {'account': alpha_account,
                                       'faithfulness': alpha_faithfulness,
                                       'systematicity': alpha_systematicity}
        yield model_parameters


def local_re_model_params_varied_alphas(alpha_resolution, with_extremes=False) -> Iterator[Dict]:
    """Model parameters for Local RE with varied alphas.

    Convenience method that uses :py:func:`varied_alphas` to generate a list of model parameters for the
    locally searching standard model (see :py:class:`LocalReflectiveEquilibrium`) as dict. Alphas are
    varied; all other parameters a set to default (see :py:class:`basics.ReflectiveEquilibrium` for details about
    model parameters).

    """
    for params in standard_model_params_varied_alphas(alpha_resolution, with_extremes):
        params['neighbourhood_depth'] = 1
        yield params


class RethonJSONEncoder(TauJSONEncoder):

    def __init__(self, serialize_implementation=False, **kwargs):
        super(RethonJSONEncoder, self).__init__(**kwargs)
        self.serialize_implementation = serialize_implementation

    def default(self, o):
        """ An implemention of :py:func:`JSONEncoder.default`.

        This implementation handles the serialization of :class:`Position`-,
        :class:`DialecticalStructure`-, :class:`REState`- and :class:`ReflectiveEquilibrium` instances.

        """

        if isinstance(o, REState):
            return o.as_dict()

        if isinstance(o, ReflectiveEquilibrium):
            re_process = {'model_name': o.model_name(),
                          'dialectical_structure': o.dialectical_structure(),
                          'model_parameters': o.model_parameters(),
                          'state': o.state()
                          }
            if self.serialize_implementation:
                re_process['module_name'] = o.__module__
                re_process['class_name'] = type(o).__name__
            return re_process

        return TauJSONEncoder.default(self, o)

    # This class will serialize a model and a model run together with all branches.




def rethon_decoder(json_obj,
                   use_json_specified_type=False,
                   position_module='tau',
                   position_class='BitarrayPosition',
                   dialectical_structure_module='tau',
                   dialectical_structure_class='BitarrayDialecticalStructure',
                   reflective_equilibrium_module='rethon',
                   reflective_equilibrium_class='BitarrayReflectiveEquilibrium'):
    """ Object hook for :py:func:`json.loads` and :py:func:`json.load`.


    :param use_json_specified_type: If :code:`True` the methods used the implementation details
            (module name and class name) that are specified in the json string, if there are any. Otherwise,
            the methode uses implementation details as given by the other parameters.

    """
    if 'model_name' in json_obj:
        if use_json_specified_type and 'module_name' in json_obj and 'class_name' in json_obj:

            re_class_ = getattr(importlib.import_module(json_obj['module_name']),
                                json_obj['class_name'])
        else:
            re_class_ = getattr(importlib.import_module(reflective_equilibrium_module),
                                reflective_equilibrium_class)
        re = re_class_(json_obj['dialectical_structure'])
        re.set_model_parameters(json_obj['model_parameters'])
        re.set_state(json_obj['state'])
        return re
    if 'evolution' in json_obj:
        # alternatives: `[[pos11,pos12,...], ..., [posn1,posn2,...]]` -> `[{pos11,pos12,...}, ..., {posn1,posn2,...}]`
        alternativess = [set(alternatives) for alternatives in json_obj['alternatives']]
        return REState(json_obj['finished'], json_obj['evolution'], alternativess, json_obj['time_line'])

    return tau_decoder(json_obj,
                       use_json_specified_type,
                       position_module,
                       position_class,
                       dialectical_structure_module,
                       dialectical_structure_class)


def rethon_dumps(re_object, cls=RethonJSONEncoder, serialize_implementation=False, **kwargs):
    """Getting an object as JSON-String.

    This is a convenient method that calls :py:func:`json.dumps` with :class:`RethonJSONEncoder` as
    its default encoder, which will handle the JSON serialization of :class:`Position`-,
    :class:`DialecticalStructure`-, :class:`REState`- and :class:`ReflectiveEquilibrium` instances.

    **kwargs will be given to :py:func:`json.dumps`

    :param serialize_implementation: If :code:`True` implementation details (module name and class name) will
            be serialized.
    :return: The object as a JSON string.

    """
    return dumps(re_object, cls=cls, serialize_implementation=serialize_implementation, **kwargs)


def rethon_dump(re_object, fp, cls=RethonJSONEncoder, serialize_implementation=False, **kwargs):
    """Saving an object as JSON-String in a file.

    This is a convenient method that calls :py:func:`json.dump` with :class:`RethonJSONEncoder` as
    its default encoder, which will handle the JSON serialization of :class:`Position`-,
    :class:`DialecticalStructure`-, :class:`REState`- and :class:`ReflectiveEquilibrium` instances.

    **kwargs will be given to :py:func:`json.dumps`

    :param serialize_implementation: If :code:`True` implementation details (module name and class name) will
            be serialized.
    :return: The object as a JSON string.

    """
    dump(re_object, fp, cls=cls, serialize_implementation=serialize_implementation, **kwargs)


def rethon_loads(json_obj,
                 use_json_specified_type=False,
                 position_module='tau',
                 position_class='BitarrayPosition',
                 dialectical_structure_module='tau',
                 dialectical_structure_class='BitarrayDialecticalStructure',
                 reflective_equilibrium_module='rethon',
                 reflective_equilibrium_class='BitarrayReflectiveEquilibrium'):
    """Loading an object from a JSON string.

    This is a convenient method that calls :py:func:`json.loads` and uses :py:func:`rethon_decoder` as object hook
    to handle the instantiation of :class:`Position`-,
    :class:`DialecticalStructure`-, :class:`REState`- and :class:`ReflectiveEquilibrium` objects. Desired
    implementation details can be given by parameter values (see :py:func:`rethon_decoder`).

    """
    return loads(json_obj,
                 object_hook=lambda x: rethon_decoder(json_obj=x,
                                                      use_json_specified_type=use_json_specified_type,
                                                      position_module=position_module,
                                                      position_class=position_class,
                                                      dialectical_structure_module=dialectical_structure_module,
                                                      dialectical_structure_class=dialectical_structure_class,
                                                      reflective_equilibrium_module=reflective_equilibrium_module,
                                                      reflective_equilibrium_class=reflective_equilibrium_class))


def rethon_load(fp,
                use_json_specified_type=False,
                position_module='tau',
                position_class='BitarrayPosition',
                dialectical_structure_module='tau',
                dialectical_structure_class='BitarrayDialecticalStructure',
                reflective_equilibrium_module='rethon',
                reflective_equilibrium_class='BitarrayReflectiveEquilibrium'):
    """Loading an object from a JSON file.

    This is a convenient method that calls :py:func:`json.load` and uses :py:func:`rethon_decoder` as object hook
    to handle the instantiation of :class:`Position`-,
    :class:`DialecticalStructure`-, :class:`REState`- and :class:`ReflectiveEquilibrium` objects. Desired
    implementation details can be given by parameter values (see :py:func:`rethon_decoder`).

    """

    return load(fp,
                object_hook=lambda x: rethon_decoder(json_obj=x,
                                                     use_json_specified_type=use_json_specified_type,
                                                     position_module=position_module,
                                                     position_class=position_class,
                                                     dialectical_structure_module=dialectical_structure_module,
                                                     dialectical_structure_class=dialectical_structure_class,
                                                     reflective_equilibrium_module=reflective_equilibrium_module,
                                                     reflective_equilibrium_class=reflective_equilibrium_class))
