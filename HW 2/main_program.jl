# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the code for Problem Set 2, where we want to compute the
# cross-sectional wealth distribution.
#    (1) initializing the algorithm
#    (2) value function iteration
#    (3) solving for the stationary distribution
#    (4) finding the asset market clearing bond price

using Distributed, SharedArrays # load package for running julia in parallel
using Parameters, Plots
include("value_function_iteration.jl") # import the functions that solves value function iteration

q = 0.9943

# ------------------------------- #
#  (1) value function iteration
# ------------------------------- #

prim, res = Initialize()          # initialize primitive and results structs 


# ------------------------------- #
#  (2) value function iteration
# ------------------------------- #

@time V_iterate(prim, res, q)     # run value function iteration
@unpack val_func, pol_func = res  # get value function and policy function
@unpack a_grid = prim             # get asset grid

# ------------------------------- #
#  (2) value function iteration
# ------------------------------- #
