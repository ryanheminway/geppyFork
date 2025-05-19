# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:35:53 2024

Unit tests got GeppyNN

@author: hemin
"""
from geppy.core.entity import *
from geppy.core.symbol import *
from geppy.tools.parser import *
from geppy.support.functions import *

from geppy.support.visualization import *

import pandas as pd
import numpy as np
import random
import os 
import torch
import unittest


class TestEvaluation(unittest.TestCase):
    def build_empty_gene(self, head_size, pset):
        tail_size = head_size * (pset.max_arity - 1) + 1
        
        return [None] * (head_size + tail_size)
    
    def setup_pset(self, input_names=['a', 'b']):
        """
        Setup a PrimitiveSet based on a given list of input names. 
        Accessible via `self.pset` after `setup()` is run. 
        """
        pset = PrimitiveSet('Main', input_names=input_names)
        pset.add_nn_function(dynamic_relu, 2, name="D_r")
        pset.add_nn_function(dynamic_sigmoid, 2, name="D_s")
        pset.add_nn_function(dynamic_tanh, 2, name="D_t")
        pset.add_nn_function(dynamic_add, 2, name="add")
        pset.add_nn_function(dynamic_sub, 2, name="sub")
        pset.add_nn_function(dynamic_mult, 2, name="mult")
        
        self.pset = pset
    
    def test_feedforward_basic(self):
        """
        Test that a basic neural network gets evaluated correctly. No jumpers
        exist in the tested individual.
        """
        self.setup_pset()
        
        head_size = 3
        
        # Head size of 3, full size will be 7
        gene_0 = self.build_empty_gene(head_size = head_size, pset = self.pset) 
        
        # add( sub(b, a), add(a, a) )
        gene_0[0] = NN_Function_With_Jumpers(name="add", arity=2)
        gene_0[1] = NN_Function_With_Jumpers(name="sub", arity=2)
        gene_0[2] = NN_Function_With_Jumpers(name="add", arity=2)
        gene_0[3] = SymbolTerminal('b')
        gene_0[4] = SymbolTerminal('a')
        gene_0[5] = SymbolTerminal('a')
        gene_0[6] = SymbolTerminal('a')
        
        # Weights and thresholds are all 1.0, for ease of computing by hand
        weights = [int(1)] * (head_size * self.pset.max_arity)
        
        gene_0.extend(weights) # Add weights (Dw)
        gene_0.extend(weights) # Add thresholds (Dt)
                
        # Make a GeneNN object
        gene_nn_0 = GeneNN.from_genome(gene_0, head_size, self.pset.max_arity, weights, weights)
        # Make a Chromosome object from the single Gene, no linker
        individual = Chromosome.from_genes([gene_nn_0], linker=None)
        
        # `compile_` uses old evaluation method (expression trees only)
        no_jumpers_func = compile_(individual, self.pset) 
        # `compile_recurrent_` acknowledges jumpers, so it can evaluate DAGs
        jumpers_func = compile_recurrent_(individual, self.pset)
        
        def ground_truth(a, b):
            return (b - a) + (a + a)
        
        # Try some random inputs to confirm that `compile_` works
        for i in range(10):
            a = random.uniform(0.0, 100.0)
            b = random.uniform(0.0, 100.0)
        
            self.assertEqual(no_jumpers_func(a, b), ground_truth(a, b))
            # `jumpers_func` takes inputs as sequences, since its an RNN
            self.assertEqual(jumpers_func([[a, b]]), ground_truth(a, b))
            
            
    def test_feedforward_basic_with_jumpers(self):
        """
        Test that a basic neural network gets evaluated correctly. Some valid 
        jumpers do exist. Recurrent jumpers will be ignored, since the inputs
        are a sequence of length 1 (not a sequence).
        
        Some INVALID jumpers exist as well, and these should be ignored.
        """
        self.setup_pset()
        
        head_size = 3
        
        # Head size of 3, full size will be 7
        gene_0 = self.build_empty_gene(head_size = head_size, pset = self.pset) 
        
        # add( sub(b, a), sub(b, a), add(a, a), add(a, a) ) ---- if jumpers are acknowledged
        node_0_function = NN_Function_With_Jumpers(name="add", arity=2)
        node_0_function.add_jumper(1, 1.0, recurrent=False)
        node_0_function.add_jumper(2, 1.0, recurrent=False)
        node_0_function.add_jumper(0, 1.0, recurrent=False) # INVALID, should be ignored
        gene_0[0] = node_0_function
        node_1_function = NN_Function_With_Jumpers(name="sub", arity=2)
        # (NOTE) All these recurrent jumpers are valid. But they should not
        # affected the result, since the inputs are not sequences (really its a length 1 sequence)
        node_1_function.add_jumper(1, 1.0, recurrent=True)
        node_1_function.add_jumper(0, 1.0, recurrent=True)
        node_1_function.add_jumper(2, 1.0, recurrent=True)
        gene_0[1] = node_1_function
        gene_0[2] = NN_Function_With_Jumpers(name="add", arity=2)
        gene_0[3] = SymbolTerminal('b')
        gene_0[4] = SymbolTerminal('a')
        gene_0[5] = SymbolTerminal('a')
        gene_0[6] = SymbolTerminal('a')
        
        # Weights and thresholds are all 1.0, for ease of computing by hand
        weights = [int(1)] * (head_size * self.pset.max_arity)
        
        gene_0.extend(weights) # Add weights (Dw)
        gene_0.extend(weights) # Add thresholds (Dt)
                
        # Make a GeneNN object
        gene_nn_0 = GeneNN.from_genome(gene_0, head_size, self.pset.max_arity, weights, weights)
        # Make a Chromosome object from the single Gene, no linker
        individual = Chromosome.from_genes([gene_nn_0], linker=None)
        
        # `compile_` uses old evaluation method (expression trees only)
        no_jumpers_func = compile_(individual, self.pset) 
        # `compile_recurrent_` acknowledges jumpers, so it can evaluate DAGs
        jumpers_func = compile_recurrent_(individual, self.pset)
        
        def ground_truth(a, b, is_dag = False):
            if is_dag:
                return (b - a) + (b - a) + (a + a) + (a + a)
            else:
                return (b - a) + (a + a)
        
        # Try some random inputs to confirm that `compile_` works
        for i in range(10):
            a = random.uniform(0.0, 100.0)
            b = random.uniform(0.0, 100.0)
        
            self.assertEqual(no_jumpers_func(a, b), ground_truth(a, b, is_dag=False))
            # `jumpers_func` takes inputs as sequences, since its an RNN
            self.assertEqual(jumpers_func([[a, b]]), ground_truth(a, b, is_dag=True))
            
    def test_recurrent_basic(self):
        """
        Test that a basic recurrent neural network gets evaluated correctly.
        
        Inputs to this network are a sequence, to test that recurrent connections
        work properly.
        """
        self.setup_pset()
        
        head_size = 3
        
        # Head size of 3, full size will be 7
        gene_0 = self.build_empty_gene(head_size = head_size, pset = self.pset) 
        
        # add( sub(b, a), add(a, a), hidden_state[0], hidden_state[1], hidden_state[2] )
        node_0_function = NN_Function_With_Jumpers(name="add", arity=2)
        # Valid recurrent jumpers
        node_0_function.add_jumper(1, 1.0, recurrent=True)
        node_0_function.add_jumper(0, 1.0, recurrent=True)
        node_0_function.add_jumper(2, 1.0, recurrent=True)
        gene_0[0] = node_0_function
        node_1_function = NN_Function_With_Jumpers(name="sub", arity=2)
        gene_0[1] = node_1_function
        gene_0[2] = NN_Function_With_Jumpers(name="add", arity=2)
        gene_0[3] = SymbolTerminal('b')
        gene_0[4] = SymbolTerminal('a')
        gene_0[5] = SymbolTerminal('a')
        gene_0[6] = SymbolTerminal('a')
        
        # Weights and thresholds are all 1.0, for ease of computing by hand
        weights = [int(1)] * (head_size * self.pset.max_arity)
        
        gene_0.extend(weights) # Add weights (Dw)
        gene_0.extend(weights) # Add thresholds (Dt)
                
        # Make a GeneNN object
        gene_nn_0 = GeneNN.from_genome(gene_0, head_size, self.pset.max_arity, weights, weights)
        # Make a Chromosome object from the single Gene, no linker
        individual = Chromosome.from_genes([gene_nn_0], linker=None)
        
        # `compile_recurrent_` acknowledges jumpers, so it can evaluate DAGs
        jumpers_func = compile_recurrent_(individual, self.pset)
        
        def generate_sequence(length):
            sequence = []
            for _ in range(length):
                # Generate a random floating-point number for each element in the inner list
                inner_list = [random.uniform(0.0, 100.0), random.uniform(0.0, 100.0)]
                sequence.append(inner_list)
            return sequence
        
        def ground_truth(sequence):
            last_node_2 = 0
            last_node_1 = 0
            last_node_0 = 0
            for input_set in sequence:
                # Calculate outputs for this timestep
                current_node_2 = input_set[0] + input_set[0]
                current_node_1 = input_set[1] - input_set[0]
                current_node_0 = current_node_1 + current_node_2 + last_node_2 + last_node_1 + last_node_0
                # Update hidden state
                last_node_2 = current_node_2
                last_node_1 = current_node_1
                last_node_0 = current_node_0
            return current_node_0
                
        for i in range(10):
            sequence = generate_sequence(5)
            self.assertAlmostEqual(jumpers_func(sequence), ground_truth(sequence), places=5)
            
                
        

            