# Author: Katherine Kwok
# Date: November 2, 2021

# This file contains the code for Problem Set 6, where we solve the Hopenhayn-Rogerson
# model of firm dynamics. The program implements the following:
#
#  (1) solve version 1 of model
#  (2) solve version 2 of model
#  (3) compare model moments: price level,
#                             mass of incumbents/entrants/exits,
#                             labor aggregate/incumbents/entrants,
#                             fraction of labor in entrants
#  (4) plot decision rules of exit 


# ------------------------------------------------------------------------ #
#  (0) load packages and functions
# ------------------------------------------------------------------------ #

using CSV, DataFrames                                  # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra         # load standard packages

include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (0) initialize algorithm
# ------------------------------------------------------------------------ #




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
