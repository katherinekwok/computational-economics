# Author: Katherine Kwok
# Date: November 17, 2021

# This file contains the code for Problem Set 7, where we estimate the AR(1)
# process using simulated method of moments (SMM). We run the SMM procedure
# for three different cases:
#
#   (1) Just identified (moments are mean and variance)
#   (2) Just identified (moments are variance and autocorrelation)
#   (3) Overidentified  (moments ares mean, variance, autocorrelation)

# ------------------------------------------------------------------------ #
#  (0) load packages and functions
# ------------------------------------------------------------------------ #

using CSV, DataFrames, Latexify                     # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra      # load standard packages

include("helper_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1) Just identified (moments are mean and variance)
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#  (2) Just identified (moments are variance and autocorrelation)
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#  (3) Overidentified  (moments ares mean, variance, autocorrelation)
# ------------------------------------------------------------------------ #
