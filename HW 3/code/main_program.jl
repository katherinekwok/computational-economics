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

# ------------------------------------------------------------------------ #
#  (0) load packages, functions, initialize
# ------------------------------------------------------------------------ #

using Distributed, SharedArrays                # load package for running julia in parallel
using Parameters, Plots, Printf, LinearAlgebra # load standard packages
include("model_and_functions.jl")              # import all functions and strucs

prim = initialize_prims()       # initialize benchmark primitives
res = initialize_results(prim)  # initialize results structures

# ------------------------------------------------------------------------ #
#  (1) solve dynamic programming problem given w_0, r_0
# ------------------------------------------------------------------------ #
v_backward_iterate(prim, res)
plot_ex_1(prim, res)

# ------------------------------------------------------------------------ #
#  (2) solve for steady distribution given w_0, r_0
# ------------------------------------------------------------------------ #
solve_ψ(prim, res)

# ------------------------------------------------------------------------ #
#  (3) test benchmark model and different experiments
# ------------------------------------------------------------------------ #

# want to converge to K = 3.35780, L = 0.34327, w = 1.45455, r = 2.36442, b = 0.22513
@time solve_model()            # (3a) test benchmark model
@time solve_model(θ_0 = 0.0)   # (3b) experiment 1: θ = 0 i.e. no social insurance

@time solve_model(z_h_0 = 0.5)            # (3c) experiment 2: z_h = z_l = 0.5 i.e. no idiosyncratic productivity
@time solve_model(z_h_0 = 0.5, θ_0 = 0.0) # (3d) experiment 3: z_h = z_l = 0.5 + θ = 0

@time solve_model(γ_0 = 1.0)              # (3e) experiment 4: γ = 1 i.e labor supply is exogenous
@time solve_model(γ_0 = 1.0, θ_0 = 0.0)   # (3f) experiment 4: γ = 1 + θ = 0
