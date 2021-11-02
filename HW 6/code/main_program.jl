# Author: Katherine Kwok
# Date: November 2, 2021

# This file contains the code for Problem Set 6, where we solve the Hopenhayn-Rogerson
# model of firm dynamics.

# ------------------------------------------------------------------------ #
#  (0) load packages and functions
# ------------------------------------------------------------------------ #

using CSV, DataFrames                                  # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra         # load standard packages

include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (0) initialize algorithm
# ------------------------------------------------------------------------ #

# Primitives: This struct stores the primitives of the model
@withkw struct Primitives

    β::Float64 = 0.8    # firm discount rate for profits
    θ::Float64 = 0.64   # persistence value of shock
    A::Float64 = 1/200  # 

    c_f = 10            # fixed costs for staying in market
    c_e = 5             # entry costs for entering market

    s::Array{Float64, 1} = [3.98e-4, 3.58, 6.82, 12.18, 18.79] # shock on firm
    e::Array{Float64, 1} = [1.3e-9, 10, 60, 300, 1000]         # employment levels given shock

    # transition matrix for shock
    s_trans_mat::Array{Float64, 2} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                                      0.1997 0.7201 0.0420 0.0326 0.0056;
                                      0.2000 0.2000 0.5555 0.0344 0.0101;
                                      0.2000 0.2000 0.2502 0.3397 0.0101;
                                      0.2000 0.2000 0.2500 0.3400 0.0100]

    # invariant entrant distribution
    entrant_dist::Array{Float64, 1} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]


end


# Results: This struct stores the results of the algorithm
mutable struct Results

    pol_func::Array{Float64, 1} # exit policy function
    val_func::Array{Float64, 1} # firm's value function

    stat_dist::

end


# ------------------------------------------------------------------------ #
#  (1) solve for entry market clearing price
# ------------------------------------------------------------------------ #

# solve value function iteration

# solve entrant's value

# ------------------------------------------------------------------------ #
#  (2) solve for labor market clearing labor demand and supply
# ------------------------------------------------------------------------ #

# solve for stationary distribution

# solve for labor demand and supply

# ------------------------------------------------------------------------ #
#  (3) display and plot results
# ------------------------------------------------------------------------ #
