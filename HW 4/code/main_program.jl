# Author: Katherine Kwok
# Date: October 23, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model. We
# solve for the transition paths for two cases:
#
#       (0) load functions, packages
#       (1) compute steady states
#       (2) unexpected elimination of social security
#       (3) expected elimination of social security

# ------------------------------------------------------------------------ #
#  (0) load packages, functions
# ------------------------------------------------------------------------ #

using Distributed, SharedArrays                # load package for running julia in parallel
using CSV, DataFrames                          # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra # load standard packages

include("model_and_functions.jl")              # import all functions and strucs
include("TP_and_functions.jl")

# ------------------------------------------------------------------------ #
#  (1) compute steady states
# ------------------------------------------------------------------------ #

p0, r0, w0, cv0 = solve_model()                # solve θ = 0.11 model - with soc security
pT, rT, wT, cvT = solve_model(θ_0 = 0.0)       # solve θ = 0 model    - with no soc security

# ------------------------------------------------------------------------ #
#  (1) compute transition path for unexpected elimination of social security
# ------------------------------------------------------------------------ #

tp_u, pt_u = solve_algorithm("unanticipated", 50, p0, pT, r0, rT) # start with 50 periods

# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #

tp_a, pt_a = solve_algorithm("anticipated", 60, p0, pT, r0, rT; date_imple_input = 21) # start with 60 periods
