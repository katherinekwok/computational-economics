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
using GLM, DataFrames                                  # load package for regression
include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1) initialize primitives, results (draw shocks at this stage)
# ------------------------------------------------------------------------ #

prim, algo, res, shocks, ϵ_seq, z_seq = initialize()
converged = 0               # convergence flag
iter = 1                    # counter for iterations
K_path = zeros(algo.T, 3)   # initialize array for agg capital path

# ------------------------------------------------------------------------ #
#  (2) solve Krusell-Smith model
# ------------------------------------------------------------------------ #

@time while converged == 0
    # value func iteration using interpolation
    value_function_iteration(prim, res, shocks, algo)

    # simulate capital path using policy function
    K_path = simulate_capital_path(prim, res, algo, ϵ_seq, z_seq)

    # estimate regression for new coefficients
    a0_new, a1_new, b0_new, b1_new = estimate_regression(K_path, algo, res)

    # check for convergence
    converged = check_convergence(res, algo, a0_new, a1_new, b0_new, b1_new, iter)
    iter += 1
end
