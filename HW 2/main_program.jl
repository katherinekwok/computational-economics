# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the code for Problem Set 2, where we want to compute the
# cross-sectional wealth distribution.
#    (0) initializing the algorithm
#    (1) value function iteration
#    (2) solving for the stationary distribution
#    (3) finding the asset market clearing bond price

using Distributed, SharedArrays # load package for running julia in parallel
using Parameters, Plots
include("value_function_iteration.jl") # import the functions that solves value function iteration
include("stationary_distribution.jl")  # import the functions that solves for stationary distribution


q = 0.9943

# ----------------------------------------------- #
#  (0) initialize things for algorithm
# ----------------------------------------------- #

prim, res = Initialize()      # initialize primitives and results struct

# ----------------------------------------------- #
#  (1) value function iteration
# ----------------------------------------------- #

@time V_iterate(prim, res, q)     # run value function iteration
@unpack val_func, pol_func = res  # get value function and policy function
@unpack a_grid = prim                 # get asset grid

# ----------------------------------------------- #
#  (2) solving for the stationary distribution
# ---------------------------------------------- ##
a = Make_big_trans_matrix(prim, res)
