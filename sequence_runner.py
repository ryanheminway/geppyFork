# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:37:52 2024

Benchmarks for sequence learning using GEPNN.

Basic benchmarks tested so far:
    - sine wave prediction

@author: hemin
"""
from geppy.core.entity import *
from geppy.core.symbol import *
from geppy.tools.toolbox import *
from geppy.tools.parser import *
from geppy.tools.mutation import *
from geppy.tools.crossover import *
from geppy.algorithms.basic import *
from geppy.support.visualization import *
from geppy.support.functions import *

# (NOTE Ryan) did not edit deap 
from deap import creator, base, tools

# # Make a folder for the run
from pathlib import Path
import time
import pickle 

from dill_utils import *

import pandas as pd
import numpy as np
import random
import os 
import torch
import matplotlib.pyplot as plt

def generate_sine_wave_dataset(num_examples=100, sequence_length=20, step=0.25):
    """
    Generate a dataset for the Sine Wave Prediction task.
    
    Args:
        num_examples (int): Number of examples to generate.
        sequence_length (int): Length of the input sequence.
        step (float): Step size for time steps.
        
    Returns:
        pd.DataFrame: A DataFrame containing the input sequences and corresponding targets.
    """
    
    minimum = 0
    maximum = (sequence_length * step) + minimum
    time_steps = np.arange(minimum, maximum, step)
    data = []

    for _ in range(num_examples):
        start = np.random.uniform(0, 2 * np.pi)  # Random start point
        sequence = np.sin(start + time_steps) # Sine wave sequence 
        input_seq = sequence[:-2]  # Input sequence (all but the last element)
        output = sequence[-1]  # Output (last element)
        data.append({'input': input_seq, 'output': output})

    return pd.DataFrame(data)


# Generate the dataset
dataset = generate_sine_wave_dataset()

# Split the dataset into train and holdout sets
msk = np.random.rand(len(dataset)) < 0.8
train = dataset[msk]
holdout = dataset[~msk]

# Prepare the data for training
SINE_SEQUENCES = torch.from_numpy(np.array([[[ele] for ele in seq] for seq in train['input']])).float()
Y = torch.from_numpy(train['output'].values).float()

#print("Got sequences... ", SINE_SEQUENCES.numpy())
#print("First sequence: ", SINE_SEQUENCES.numpy()[0])


pset = PrimitiveSet('Main', input_names=['x'])
pset.add_nn_function(dynamic_relu, 2, name="D_r")
pset.add_nn_function(dynamic_relu, 3, name="T_r")
pset.add_nn_function(dynamic_relu, 4, name="Q_r")
pset.add_nn_function(dynamic_sigmoid, 2, name="D_s")
pset.add_nn_function(dynamic_sigmoid, 3, name="T_s")
pset.add_nn_function(dynamic_sigmoid, 4, name="Q_s")
pset.add_nn_function(dynamic_tanh, 2, name="D_t")
pset.add_nn_function(dynamic_tanh, 3, name="T_t")
pset.add_nn_function(dynamic_tanh, 4, name="Q_t")
pset.add_nn_function(dynamic_add, 2, name="add")
pset.add_nn_function(dynamic_sub, 2, name="sub")
pset.add_nn_function(dynamic_mult, 2, name="mult")

# Create Individual class and fitness measurement
creator.create("FitnessMin", base.Fitness, weights=(-1,)) # minimize fitness
creator.create("Individual", Chromosome, fitness=creator.FitnessMin)

def plot_sequences(sequence, true_output, predicted_output):
    """
    Plots the true sequence and the predicted sequence for a given sequence index.
    
    Args:
        true_sequences: The true sequences.
        predicted_sequences: The predicted sequences.
        sequence_idx (int): The index of the sequence to plot.
    """
    plt.figure(figsize=(8, 6))
    
    sequence_in_list = [ele[0] for ele in sequence]
    
    # Plot both sequences on the same subplot
    plt.plot(sequence_in_list + [true_output.numpy()], label='True Sequence', color='blue')
    plt.plot(sequence_in_list + [predicted_output.numpy()], label='Predicted Sequence', color='red')
    
    plt.title("Comparison of True and Predicted Sequences")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()  # Add a legend to distinguish the sequences
    
    plt.tight_layout()
    plt.show()
    


def evaluate(individual, plot=False): 
    #print("evaluating: ", str(individual))
    #start = time.perf_counter()
    
    rec_func = toolbox.compile_recurrent(individual)
    # Sequences is a list of elements, where each element is a sequence. 
    # Each sequence is a list of elements where each element is a list 
    # representing an input for one time step. 
    sequences = SINE_SEQUENCES
    Yp_list_from_sequences = list(map(rec_func, sequences))
    
    Yp_stacked = torch.Tensor(len(Yp_list_from_sequences)).float()
    torch.stack(Yp_list_from_sequences, dim=0, out=Yp_stacked)

    #print("Yp: ", Yp_stacked)
    #print("Y: ", Y)

    # Calculate loss
    loss = torch.nn.MSELoss()
    loss_value = loss(Yp_stacked, Y).item()

    #end = time.perf_counter()
    #secs = (end-start)
    #print("Evaluate took (secs): ", secs)
    
    if plot:
        print("Got Y: ", Y)
        print("Got sequences... ", sequences.numpy())
        print("Got Yp_stacked: ", Yp_stacked)
        plot_sequences(sequences[0].numpy(), Y[0], Yp_stacked[0])
    
    return loss_value,


# GEP NN parameters
guided = False
h = 10 # 8       # head length
n_genes = 1 # number of genes in a chromosome
r = 10      # length of RNC arrays

toolbox = Toolbox()
toolbox.register("weight_gen", random.uniform, -1, 1) # -2, 2)
toolbox.register("thresh_gen", random.randint, 1, 1)
toolbox.register("gene_gen", GeneNN, pset=pset, head_length=h, 
                  dw_rnc_gen=toolbox.weight_gen, dw_rnc_array_length=r, 
                  dt_rnc_gen=toolbox.thresh_gen, dt_rnc_array_length=r, func_head=guided)
toolbox.register("individual", creator.Individual, gene_gen=toolbox.gene_gen,
                  n_genes=n_genes, linker=None)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("compile", compile_, pset=pset)
toolbox.register("compile_recurrent", compile_recurrent_, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mut_dw", mutate_uniform_dw, ind_pb=0.044, pb=1)
toolbox.register("mut_dt", mutate_uniform_dt, ind_pb=0.044, pb=1)
toolbox.register("mut_tanspose_dw", transpose_dw, pb=0.1)
toolbox.register("mut_transpose_dt", transpose_dt, pb=0.1)
toolbox.register("mut_rncs_dw", mutate_rnc_array_dw, rnc_gen=toolbox.weight_gen, ind_pb='1p', pb=0.05)
toolbox.register("mut_rncs_dt", mutate_rnc_array_dt, rnc_gen=toolbox.thresh_gen, ind_pb='1p', pb=0.05)
toolbox.register("cx_1p", crossover_one_point, pb=0.6)
# (NOTE Ryan) Disable mut_add_jumper to test FFNN on recurrent problem
toolbox.register("mut_add_jumper", mutate_add_jumpers, weight_gen=toolbox.weight_gen, ind_pb=0.044, pb=1)
toolbox.register("mut_uniform", mutate_uniform, pset=pset, ind_pb=0.044, pb=1)
#toolbox.register("mut_is_transpose", is_transpose, pb=0.1)
#toolbox.register("mut_ris_transpose", ris_transpose, pb=0.1)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


n_pop = 100
n_gen = 25 # 1000

champs = 3

iters = 1 # 100

# Make a folder for the run
from pathlib import Path
import time
today = time.strftime("%Y%m%d")
run_dir = "runs"
model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_sinewave' 
if guided:
    model_path += "_guided"
model_path += "/"
Path(model_path).mkdir(parents=True, exist_ok=True)

results_file = model_path + '/results.txt'
def _write_to_file(file, content):
    f = open(file, 'a')
    f.write(content)  
    f.close()


_write_to_file(results_file, "Running GEPNN solver for a Sine Wave Prediction problem\n")

avg_fitness = 0
for i in range(iters):
    print("Running iteration: ", i)
    start = time.perf_counter()
    
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(champs) 
    # start evolution
    pop, log = gep_simple(pop, toolbox, n_generations=n_gen, n_elites=10,
                              stats=stats, hall_of_fame=hof, verbose=True)
    best_ind = hof[0]
    
    print("got best: ", best_ind)
    fitness_best = evaluate(best_ind, plot=True)[0]
    print("eval'd best: ", fitness_best)
    _write_to_file(results_file, "Iter {} got best fitness: {} \n".format(i, fitness_best))
    file_name = model_path + "sinewave_iter_{}_fit_{}".format(i, fitness_best) + ".png"
    rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
    export_expression_tree_nn(best_ind, rename_labels, file_name)
    avg_fitness += fitness_best
    
    pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
    pickle.dump(log, pkl_file)
    
    end = time.perf_counter()
    print(f"That iteration took {(end - start) / 60.0} minutes to complete.")
    
avg_fitness = avg_fitness / iters
_write_to_file(results_file, "Finished. Got AVG fitness: {} \n".format(i, avg_fitness))