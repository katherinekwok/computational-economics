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
using Parameters, Plots, Printf
include("model_and_functions.jl") # import all functions and strucs

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
Plot_λ(λ, prim, res)                                 # (a) calculate W_FB, λ
Calc_welfare(prim, res, λ)                           # (b) calculate W_INC, W_G, fraction in favor of complete mkt
