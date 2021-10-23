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
#  (0) load packages, functions; compute steady states
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

# initiate with 30 transition periods
tp_u, pt_u = solve_algorithm("unanticipated", 30)

# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #

# initiate with 50 transition periods, date when policy implemented at 21
tp_a, pt_a = solve_algorithm("anticipated", 50; date_imple_input = 21)
