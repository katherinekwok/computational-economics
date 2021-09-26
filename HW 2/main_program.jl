# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the code for Problem Set 2, where we want to compute the
# cross-sectional wealth distribution.
#    (0) initializing the algorithm
#    (1) value function iteration
#    (2) solving for the stationary distribution
#    (3) finding the asset market clearing bond price
#    (4) make plots
#    (5) welfare calculations

# ----------------------------------------------- #
#  (0) load packages, functions, initialize
# ----------------------------------------------- #

using Distributed, SharedArrays # load package for running julia in parallel
using Parameters, Plots
include("value_function_iteration.jl") # import the functions and strucs that solves value function iteration
include("stationary_distribution.jl")  # import the functions that solves for stationary distribution and asset market clearing
include("produce_figures.jl")          # import the functions that supplements figure making

prim, res, loop = Initialize()    # initialize primitives, results, loop struct

# ----------------------------------------------- #
#  (1), (2), (3) solve model
# ----------------------------------------------- #

@time while loop.converged == 0
      V_iterate(prim, res, loop.q)           #  (1) value function iteration
      T_star_iterate(prim, res, loop.q)      #  (2) solve for the stationary distribution
      Check_asset_clearing(prim, res, loop)  #  (3) check asset market clearing
end

# ----------------------------------------------- #
#  (4) make plots
# ----------------------------------------------- #

Plot_pol_func(res, prim)                             # (a) plot policy functions
w, w_mass_e, w_mass_u = Plot_wealth_dist(res, prim)  # (b) plot wealth distribution
Plot_lorenz(w, w_mass_e, w_mass_u)                   # (c) plot lorenz curve

# ----------------------------------------------- #
#  (5) welfare calculations
# ----------------------------------------------- #
