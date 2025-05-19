# -*- coding: utf-8 -*-
"""
@author: Ryan Heminway

This :mod:`eant` module contains the classes that specify 
an individual encoding model using the Common Genetic Encoding (CGE),
and the functions to evaluate it as a neural network.
"""


from collections import deque
import numpy as np
import random


# -------------------- EANT Data Structure ------------------ #

class Node(object):
    """
    A basic node in a CGE individual (Genome). Can be either a Neuron,
    InputNode, or Jumper node. Tracks its previous output for use in recurrent
    networks.
    """
    def __init__(self, name, prev_output=0):
        self.name = name
        # Output from last timestep
        self.prev_output = 0
        # Output from current timestep
        self.curr_output = 0
    
    def step(self):
        """ 
        Progress a timestep by updating previous output.
        """
        self.prev_output = self.curr_output
        
    def reset(self):
        """
        Reset the state of this node. If it is to be used for a fresh forward
        pass.
        """
        self.prev_output = 0
        self.curr_output = 0
        
    def __str__(self):
        str_rep = "NODE: {} P_OUT: {} C_OUT: {}"
        return str_rep.format(self.name, self.prev_output, self.curr_output) 

    
class Neuron(Node):
    """
    A Neuron is one of the node types in an CGE individual (Genome). Represents
    a classical neural network unit with an activation function and a specified
    arity of inputs. 
    """
    def __init__(self, unique_id, arity, fn='relu', depth=0):
        super().__init__("Neuron{}".format(unique_id))
        self.fn = fn
        self.unique_id = unique_id
        self.arity = arity
        self.depth = depth
        
    def activate(self, val):
        if self.fn == 'relu':
            if val > 0:
                self.curr_output = val
            else:
                self.curr_output = 0
            return self.curr_output
        elif self.fn == 'sigmoid':
            self.curr_output = 1/(1 + np.exp(-val))
            # Check for overflow
            if (self.curr_output > 1):
                self.curr_output = 1
            elif (self.curr_output < 0):
                self.curr_output = 0
            return self.curr_output
        elif self.fn == 'identity':
            self.curr_output = val
            return self.curr_output
        else:
            return NotImplemented
    
    def __str__(self):
        str_rep = super().__str__() + " ID: {} ARITY: {} ACT_FN: {} DEPTH: {}"
        return str_rep.format(self.unique_id, self.arity, self.fn, self.depth)
            

class InputNode(Node):
    """
    An InputNode is one of the node types in an CGE individual (Genome). 
    Represents an input to the neural network. 
    """
    def __init__(self, name, value=None):
        super().__init__(name)
        self.value = value
        
    def __str__(self):
        str_rep = super().__str__() + " VAL: {}"
        return str_rep.format(self.value)    
    
    def set_value(self, value):
        self.value = value
    

class Jumper(Node):
    """
    A Jumper node is one of the node types in an CGE individual (Genome). 
    Represents forward or recurrent connections that break the standard 
    connectivity pattern for fully-connected MLPs. 
    """
    def __init__(self, unique_id, recurrent=False):
        super().__init__("Jumper{}{}".format("Rec" if recurrent else "For", unique_id))
        self.recurrent = recurrent 
        self.unique_id = unique_id

    def __str__(self):
        str_rep = super().__str__() + " ID: {} RECURRENT?: {}"
        return str_rep.format(self.unique_id, self.recurrent) 
    
    def is_similar(self, node):
        if (not (isinstance(node, Jumper))):
            return False
        similar = (self.recurrent == node.recurrent)
        return similar
    
    
class Genome(list):
    """
    Genome using the Common Genetic Encoding (CGE).
    Represents an individual in the population. The Genome
    is represented as a list of nodes, where any node can
    describe a Neuron, Input, or Jumper connection (forward or
    recurrent). The ordering of the nodes is pivotal and
    describes the shape of the neural network encoded by
    this genotype. 
    
    For this implementation, an entry in the list can be
    viewed as a Tuple representing (Node, weight). This allows
    Nodes to be unique. 
    """
    def __init__(self, content):
        list.__init__(self, content)
        
        if (isinstance(content, Genome)):
            self.neuron_list = content.neuron_list.copy()
            self.input_list = content.input_list.copy()
            self.node_list = content.node_list.copy()
            self.node_id_list = content.node_id_list.copy()
            self.max_id = content.max_id
        else:           
            self.neuron_list = []
            self.input_list = []
            self.node_list = []
            self.node_id_list = []
            self.max_id = 0
            # (TODO) Add thresholds
            # Create lists and do some sanity checks
            for (node, weight) in content:
                if (not (node in self.node_list)):
                    self.node_list.append(node)
                    if (isinstance(node, Neuron)):
                        if (node.unique_id in self.node_id_list):
                            print("ERROR DUPLICATE IDs")
                        else:
                            if (node.unique_id > self.max_id):
                                self.max_id = node.unique_id
                                
                            self.neuron_list.append(node)
                            self.node_id_list.append(node.unique_id)
                    elif (isinstance(node, InputNode)):
                        if (not (node in self.input_list)):
                            self.input_list.append(node)   
      
                     
    def clone(self):
        g = Genome(self.copy())
        g.input_list = self.input_list.copy()
        g.neuron_list = self.neuron_list.copy()
        g.node_list = self.node_list.copy()
        g.node_id_list = self.node_id_list.copy()
        g.max_id = self.max_id
        
        return g
    
    def get_weights(self):
        """
        Get a list with just the weight values used in this CGE Genome. 

        Returns
        -------
        list
            List of weights corresponding to each node in the CGE representation.
            Order matches that of the nodes in the encoding.

        """
        return [x[1] for x in self]
    
    def set_weights(self, weights):
        """
        Set new weights for this individual. Given list of weights must match
        length of this individual. One weight for each node. This method
        will mutate this object.

        Parameters
        ----------
        weights : List
            List of weights corresponding to each node in the CGE representation.

        Returns
        -------
        None.

        """
        assert(len(weights) == len(self))
        for i, w in enumerate(weights):
            # Each entry in self is a tuple (node, weight)
            node = self[i][0]
            self[i] = (node, w)
                     
                        
    def _set_inputs(self, **inputs):
        """
        Set values for the input nodes. Required for `evaluate` to return
        a valid result. 

        Parameters
        ----------
        **inputs : Dictionary containing values to apply to InputNodes. 
                   Keys must match names of InputNodes. 

        Returns
        -------
        None.
        """
        for name in inputs:
            for input_node in self.input_list:
                if input_node.name == name:
                    input_node.set_value(inputs[name])

    

      