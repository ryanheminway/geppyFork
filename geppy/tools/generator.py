# coding=utf-8
"""
.. moduleauthor:: Shuhua Gao

This module :mod:`generator` provides functionality to generate a genome for a gene. That is, choose functions and
terminals randomly from a given primitive set to form a linear form gene expression.
"""
import random
from ..core.symbol import NN_Function_With_Jumpers
from ._util import _choose_a_terminal

# TODO (Ryan Heminway) added guided here
# TODO May want to make this compatible with any type of Function, not just NN_
def generate_genome(pset, head_length, guided=False):
    """
    Generate a genome with the given primitive set *pset* and the specified head domain length *head_length*.

    :param pset: a primitive set
    :param head_length: length of the head domain
    :return: a list of symbols representing a genome

    Supposing the maximum arity of functions in *pset* is *max_arity*, then the tail length is automatically
    determined to be ``tail_length = head_length * (max_arity - 1) + 1``.
    """
    h = head_length
    functions = pset.functions
    terminals = pset.terminals

    n_max = max(p.arity for p in functions)  # max arity
    t = h * (n_max - 1) + 1
    expr = [None] * (h + t)
    # head part: initialized with both functions and terminals
    for i in range(h):
        prob = random.random()
        if guided or prob < 0.5:
            # Get a new instance, for safety when handling jumpers
            chosen_function = random.choice(functions)
            expr[i] = NN_Function_With_Jumpers(name=chosen_function.name, arity=chosen_function.arity)
        else:
            expr[i] = _choose_a_terminal(terminals)
    # tail part: only terminals are allowed
    for i in range(h, h + t):
        expr[i] = _choose_a_terminal(terminals)
    return expr


def generate_dc(rnc_array_length, dc_length):
    """
    Generate a Dc domain for a RNC array of size *rnc_array_length*.

    :param rnc_array_length: length of the RNC array
    :param dc_length: length of the Dc domain
    :return: a list of integers of length *dc_length*, each element in range `[0, rnc_array_length - 1]`

    Refer to Chapter 5 of [FC2006]_ for more details.
    """
    return [random.randint(0, rnc_array_length - 1) for _ in range(dc_length)]
