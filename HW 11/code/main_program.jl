# Author: Katherine Kwok
# Date: December 21, 2021

# This file contains the code for Problem Set 4 (for JF's portion) a.k.a.
# Problem Set 11 overall. The main program implements a dynamic model of
# inventory control.

using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames, CSV                       # load packages for handling data
using Latexify                                         # load package for outputting results
using Distributions, Optim, StatsBase                  # load packages for optimization stuff

include("helper_functions.jl")                         # load helper functions

root_path = pwd()                                      # set file paths
data_path = root_path * "/data/"
output_path = root_path * "/output/"

# ---------------------------------------------------------------------------- #
# (0) Set up primitives; read and process data
# ---------------------------------------------------------------------------- #

prim = Primitives()
dataset = process_data(data_path, prim)

# ---------------------------------------------------------------------------- #
# (1) Solve the expected value function using implicit equation
# ---------------------------------------------------------------------------- #

EV1 = val_func_iter(prim, dataset)

# ---------------------------------------------------------------------------- #
# (2) Solve for expected value function using CCP mapping
# ---------------------------------------------------------------------------- #

P_hat = get_P_hat(prim, dataset)             # use CCP on P hat (from simulated data)
EV2, P2 = ccp_mapping(P_hat, prim, dataset)

EV3, P3, F_vec = pol_func_iter(prim, dataset) # policy function iteration

# ---------------------------------------------------------------------------- #
# (3) Log-likelihood
# ---------------------------------------------------------------------------- #

bfgs = @time optimize(λ -> -log_likelihood(λ, prim, dataset), [prim.λ], BFGS(); inplace = false)
λ_bfgs = Optim.minimizer(bfgs)

# ---------------------------------------------------------------------------- #
# (4) Calculate MLE of α using nested fixed point algorithm
# ---------------------------------------------------------------------------- #

nested = @time optimize(λ -> -log_likelihood_nested(λ, prim, dataset), [prim.λ], BFGS(); inplace = false)
λ_nested = Optim.minimizer(nested)
