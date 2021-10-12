# Author: Katherine Kwok
# Date: October 11, 2021

# This file contains the code for Problem Set 3, where we want to want to evaluate
# the value of the Social Security program. The program is broken down into the
# following steps:
#
#      exercise(1) Solve the dynamic programming problem for individuals at different ages
#                  (where if age >= 46, agent is retired; working if not)
#      exercise(2) Solve for the steady-state distribution of agents over age, productivity
#                  and asset holdings.
#      exercise(3) Test counterfactuals

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

# (3a) test benchmark model, guess K_0, L_0 close to steady state for quick convergence
@time p0, r0 = solve_model()

# (3b) experiment 1: θ = 0 i.e. no social insurance, same K_0, L_0 guess as above
@time p1, r1 = solve_model(θ_0 = 0.0)

# (3c) experiment 2: z_h = z_l = 0.5 i.e. no idiosyncratic productivity, guess K_0, L_0 close to steady state for quick convergence
@time p2, r2 = solve_model(K_0 = 1.0, L_0 = 0.1, z_h_0 = 0.5, λ = 0.3, tol = 1.0e-2)

# (3d) experiment 3: z_h = z_l = 0.5 + θ = 0, same guess as (3c) for K_0, L_0
@time p3, r3 = solve_model(K_0 = 1.0, L_0 = 0.1, z_h_0 = 0.5, θ_0 = 0.0,  λ = 0.3, tol = 1.0e-2)

# (3e) experiment 4: γ = 1 i.e labor supply is exogenous
@time p4, r4 = solve_model(γ_0 = 1.0)

# (3f) experiment 4: γ = 1 + θ = 0
@time solve_model(γ_0 = 1.0, θ_0 = 0.0)
