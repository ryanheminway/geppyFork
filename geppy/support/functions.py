# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:43:10 2024

@author: hemin
"""
import torch

# -------------------- Function Set Definitions (start) --------------------- #
def binary_thresh(activation, threshold=1.0):
    return (activation > threshold).float()

def sigmoid(activation, threshold=1.0):
    return torch.sigmoid(activation)

def tanh(activation, threshold=1.0):
    return torch.tanh(activation)

def relu(activation, threshold=1.0):
    return torch.relu(activation)

def base_neuron_fn(in_tensor, w_tensor, threshold, activation="binary-thresh"):
    """
    Basic neuron functionality: dot product between inputs and weights, with
    activation function applied to the result. Expects inputs and weights 
    arguments to be given as PyTorch tensors.
    """
    activations = { 'binary-thresh' : binary_thresh,
                    'sigmoid' : sigmoid,
                    'tanh' : tanh,
                    'relu' : relu,
                 }
    act = torch.dot(in_tensor, w_tensor)
    act_fun = activations[activation]
    return act_fun(act, threshold)

def _dynamic_neuron_wrapper(*inputs, activation="binary-thresh", weights=None, threshold=1.0):
    if len(inputs) < 1:
        raise ValueError("At least one input tensor must be provided.")
        
    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError("Number of weights must match the number of input tensors.")
        
    if not isinstance(weights, (list, tuple)):
        raise TypeError("Weights must be provided as a list or tuple.")
        
    tensor_inputs = [torch.tensor(inp, dtype=torch.float) for inp in inputs]    
    inputs_tensor_tuple = (inp.reshape(1) for inp in tensor_inputs)
    in_tensor = torch.cat(tuple(inputs_tensor_tuple), 0)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    return base_neuron_fn(in_tensor, weights_tensor, threshold, activation)

def dynamic_relu(*inputs, weights=None, threshold=1.0):
    return _dynamic_neuron_wrapper(*inputs, activation="relu", weights=weights, threshold=threshold)

def dynamic_sigmoid(*inputs, weights=None, threshold=1.0):
    return _dynamic_neuron_wrapper(*inputs, activation="sigmoid", weights=weights, threshold=threshold)

def dynamic_tanh(*inputs, weights=None, threshold=1.0):
    return _dynamic_neuron_wrapper(*inputs, activation="tanh", weights=weights, threshold=threshold)

def dynamic_add(*inputs, weights=None, threshold=1.0):
    if len(inputs) < 1:
        raise ValueError("At least one input tensor must be provided.")
    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError("Number of weights must match the number of input values.")
    
    result = 0
    for inp, w in zip(inputs, weights):
        result = result + inp * w
    
    return result

def dynamic_sub(*inputs, weights=None, threshold=1.0):
    if len(inputs) < 1:
        raise ValueError("At least one input tensor must be provided.")
    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError("Number of weights must match the number of input values.")
    
    result = 0
    need_first_val = True
    for inp, w in zip(inputs, weights):
        if need_first_val:
            result = inp * w
            need_first_val = False
        else:
            result = result - inp * w
    
    return result

def dynamic_mult(*inputs, weights=None, threshold=1.0):
    if len(inputs) < 1:
        raise ValueError("At least one input tensor must be provided.")
    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError("Number of weights must match the number of input values.")
    
    result = 1
    for inp, w in zip(inputs, weights):
        result = result * inp * w
    
    return result

def dynamic_protected_div(*inputs, weights=None, threshold=1.0):
    if len(inputs) < 1:
        raise ValueError("At least one input tensor must be provided.")
    if weights is None:
        weights = [1.0] * len(inputs)
    elif len(weights) != len(inputs):
        raise ValueError("Number of weights must match the number of input values.")
    
    result = 0
    need_first_val = True
    for inp, w in zip(inputs, weights):
        next_val = inp * w
        if need_first_val:
            result = next_val
            need_first_val = False
        elif next_val < 1e-6:
            result = 1
        else:
            result = result / next_val
    
    return result

def nn_add(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 + a2

def nn_sub(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 - a2

def nn_mult(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    return a1 * a2

def nn_div_protected(in1, in2, weights=[1.0, 1.0], threshold=1.0):
    a1 = in1 * weights[0]
    a2 = in2 * weights[1]
    if abs(a2) < 1e-6:
        return 1
    return a1 / a2

def T_link_softmax(*inputs):
    inputs_tuple = ()
    for inp in inputs:
        inputs_tuple = inputs_tuple + (inp.reshape(1),)
    in_tensor = torch.cat(inputs_tuple, 0)
    softmax = torch.nn.Softmax(dim=0)
    return softmax(in_tensor)
# -------------------- Function Set Definitions (end) --------------------- #

