# Author: Katherine Kwok
# Date: November 2, 2021

# This file contains the code for Problem Set 6, where we solve the Hopenhayn-Rogerson
# model of firm dynamics. The program implements the following:
#
#  (1) solve version 1 of model: standard
#  (2) solve version 2 of model: add action-specific shocks with α = 1
#                                add action-specific shocks with α = 2
#
#  (3) compare model moments:    price level,
#                                mass of incumbents/entrants/exits,
#                                labor aggregate/incumbents/entrants,
#                                fraction of labor in entrants
#
#  (4) plot decision rules of exit for each version
#
#  (5) solve models with c_f = 15 rather than default c_f = 10


# ------------------------------------------------------------------------ #
#  (0) load packages and functions
# ------------------------------------------------------------------------ #

using CSV, DataFrames                                  # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra         # load standard packages

include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1) solve version 1 of model: standard
# ------------------------------------------------------------------------ #

p_std, r_std = initialize()     # initialize
@time solve_price(p_std, r_std) # solve industry price


# solve for stationary distribution

# solve for labor demand and supply

# ------------------------------------------------------------------------ #
#  (2) solve version 2 of model: add action-specific shocks with α = 1
#                                add action-specific shocks with α = 2
# ------------------------------------------------------------------------ #



# ------------------------------------------------------------------------ #
#  (3) compare model moments
# ------------------------------------------------------------------------ #


# ------------------------------------------------------------------------ #
#  (4) plot decision rules of exit for each version
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#  (5) solve models with c_f = 15 rather than default c_f = 10
# ------------------------------------------------------------------------ #
