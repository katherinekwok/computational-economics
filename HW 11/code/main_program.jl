# Author: Katherine Kwok
# Date: December 15, 2021

# This file contains the code for Problem Set 4 (for JF's portion) a.k.a.
# Problem Set 11 overall. The main program implements a dynamic model of
# inventory control.

using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames, CSV                       # load packages for handling data
using Latexify                                         # load package for outputting results
using Distributions, Optim, StatsBase                  # load packages for optimization stuff

include("helper_functions.jl")                         # load helper functions

root_path = pwd()                                      # set file paths
data_path = root_path * "/data/"
output_path = root_path * "/output/"

# ---------------------------------------------------------------------------- #
# (0) Set up primitives; read and process data
# ---------------------------------------------------------------------------- #

prim = Primitives()

sim_data = Array(DataFrame(load(data_path*prim.sim_data_file)))[:, 2:end]   
S = Array(DataFrame(load(data_path*prim.state_space_file)))[:, 2:end]
F_0 = Array(DataFrame(load(data_path*prim.trans_a0_file)))[:, 2:end]
F_1 = Array(DataFrame(load(data_path*prim.trans_a1_file)))[:, 2:end]

# ---------------------------------------------------------------------------- #
# (1) Solve the expected value function using implicit equation
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# (2) Solve for expected value function using CCP mapping
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# (3) Calculate MLE of α using nested fixed point algorithm
# ---------------------------------------------------------------------------- #
