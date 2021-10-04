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


@unpack val_func, pol_func, lab_func = res
@unpack a_grid, nz, age_retire = prim

# plot value function for age 50
index_age_50 = (age_retire - 1) * nz + 50 - age_retire + 1
Plots.plot(a_grid, val_func[:, index_age_50], label = "", title = "Value Function for Retired Agent at 50")

# plot policy function for age 20
index_age_20_h = 20 * nz - 1
index_age_20_l = 20 * nz
Plots.plot(a_grid, pol_func[:, index_age_20_h] .- a_grid, label = "High productivity")
Plots.plot!(a_grid, pol_func[:, index_age_20_l] .- a_grid, label = "Low productivity",
title = "Policy Function for Worker at Age 20", legend = :topleft, xlims = [0, 80], ylims = [-1, 2])

# plot labor supply for age 20
Plots.plot(a_grid, lab_func[:, index_age_20_h], label = "High productivity")
Plots.plot!(a_grid, lab_func[:, index_age_20_l], label = "Low productivity",
title = "Labor Supply for Worker at Age 20", legend = :topleft)
