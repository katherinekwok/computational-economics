# Author: Katherine Kwok
# Date: October 13, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model. We
# solve for the transition paths for two cases:
#
#       (1) unexpected elimination of social security
#       (2) expected elimination of social security

# ------------------------------------------------------------------------ #
#  (0) load packages, functions, initialize
# ------------------------------------------------------------------------ #

using Distributed, SharedArrays                # load package for running julia in parallel
using CSV, DataFrames                          # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra # load standard packages
include("models_and_functions_SS.jl")           # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1) compute transition path for unexpected elimination of social security
# ------------------------------------------------------------------------ #



# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #
