# Author: Katherine Kwok
# Date: October 22, 2021

# This file contains the code for Problem Set 5, where we solve the Krusell-Smith
# model.

# ------------------------------------------------------------------------ #
#  (0) load packages, functions, initialize
# ------------------------------------------------------------------------ #

using CSV, DataFrames                                  # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra         # load standard packages
using Interpolations, Optim                            # load package for interpolation and optimization
using Random, Distributions                            # load package for drawing shocks
include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1) initialize primitives, results (draw shocks at this stage)
# ------------------------------------------------------------------------ #

prim, algo, res, shocks, ϵ_seq, z_seq = initialize()
converged = 0

# ------------------------------------------------------------------------ #
#  (2) solve Krusell-Smith model
# ------------------------------------------------------------------------ #

while converged == 0
    res.pol_func, res.val_func = value_function_iteration(prim, res, shocks)
    # simulate capital path
    # estimate regression
    # check difference between initial vs. model estimate
end
