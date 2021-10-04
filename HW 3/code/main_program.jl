# Author: Katherine Kwok
# Date: Sept 28, 2021

# This file contains the code for Problem Set 3, where we want to want to evaluate
# the value of the Social Security program. The program is broken down into the
# following steps:
#
#      (1) Solve the dynamic programming problem for individuals at different ages
#          (where if age >= 46, agent is retired; working if not)
#      (2) Solve for the steady-state distribution of agents over age, productivity
#          and asset holdings.
#      (3) Test counterfactuals

# ----------------------------------------------- #
#  (0) load packages, functions, initialize
# ----------------------------------------------- #

using Distributed, SharedArrays   # load package for running julia in parallel
using Parameters, Plots, Printf, LinearAlgebra # load standard packages
include("model_and_functions.jl") # import all functions and strucs

prim = initialize_prims()   # initialize benchmark primitives
res = initialize_results(prim)  # initialize results structures

# ----------------------------------------------- #
#  (1) solve dynamic programming problem
# ----------------------------------------------- #

v_backward_iterate(prim, res)
plot_ex_1(prim, res)
