# Author: Katherine Kwok
# Date: November 22, 2021

# This file contains the code for Problem Set 2 (for JF's portion) a.k.a.
# Problem Set 9 overall. The main program does the following:

#   (0) load data, initialize
#   (1) use quadrature method to predict choice probabilities
#   (2) use GHK method to predict choice probabilities
#   (3) use accept/reject method to predict choice probabilities
#   (4) run maximum likelihood using quadrature method

using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames, CSV                       # load packages for handling data
using Latexify                                         # load package for outputting results
using Distributed, SharedArrays

@everywhere using Distributions, Optim, FiniteDiff, Random

include("helper_functions.jl")                         # load helper functions

root_path = pwd()                                      # set file paths
data_path = root_path * "/data/"

mortgage_data_path = data_path * "mortgage_performance_data.dta"
KPU_d1_path = data_path * "KPU_d1_l20.csv"
KPU_d2_path = data_path * "KPU_d2_l20.csv"

Random.seed!(12032020) # set seed

# ---------------------------------------------------------------------------- #
#   (0) load data and set up variables
# ---------------------------------------------------------------------------- #
param = Primitives()
X, Z, Y = read_mortgage_data(mortgage_data_path, param)
KPU_d1 = read_KPU_data(KPU_d1_path)
KPU_d2 = read_KPU_data(KPU_d2_path)


# ---------------------------------------------------------------------------- #
#   (1) use quadrature method to predict choice probabilities
# ---------------------------------------------------------------------------- #

quad_probs = quadrature_wrapper(X, Z, KPU_d1, KPU_d2, param)

# ---------------------------------------------------------------------------- #
#   (2) use GHK method to predict choice probabilities
# ---------------------------------------------------------------------------- #

ghk_probs = ghk_wrapper(X, Z, param)

# ---------------------------------------------------------------------------- #
#   (3) use accept/reject method to predict choice probabilities
# ---------------------------------------------------------------------------- #

a_r_probs = accept_reject_wrapper(X, Z, param)

# ---------------------------------------------------------------------------- #
#   (4) run maximum likelihood using quadrature method
# ---------------------------------------------------------------------------- #

θ_init = vcat([param.α0, param.α1, param.α2], param.β, param.γ, [param.ρ])
θ_bfgs_Y_0 = @time optimize(θ -> -log_likelihood(X, Z, Y, KPU_d1, KPU_d2, θ; T = 1), θ_init, BFGS(); inplace = false)
Optim.minimizer(θ_bfgs_Y_0)
