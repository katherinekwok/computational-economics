# Author: Katherine Kwok
# Date: November 16, 2021

# This file contains the code for Problem Set 6, where we solve the Hopenhayn-Rogerson
# model of firm dynamics. The program implements the following:

#  For c_f values (fixed cost of entry) 10 and 15:
#
#  (1) solve version 1 of model: standard
#
#  (2) solve version 2 of model: add action-specific shocks with α = 1
#
#  (3) solve version 2 of model: add action-specific shocks with α = 2
#
#  (4) compare model moments and plot decision rules of exit for each version:
#      (price level, mass of incumbents/entrants/exits, labor aggregate/incumbents/
#       entrants, fraction of labor in entrants)
#


# ------------------------------------------------------------------------ #
#  (0) load packages and functions
# ------------------------------------------------------------------------ #

using CSV, DataFrames, Latexify                        # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra         # load standard packages

include("model_and_functions.jl")                      # import all functions and strucs

# ------------------------------------------------------------------------ #
#  (1-4) solve models with different specifics and output results
# ------------------------------------------------------------------------ #

c_f_init = [10, 15] # different fixed costs for entry to test, default = 10

for (c_index, c_f_val) in enumerate(c_f_init)

    # ------------------------------------------------------------------- #
    # (1) benchmark version
    # ------------------------------------------------------------------- #
    p_std, r_std = initialize(;c_f_init = c_f_val)
    @time solve_price(p_std, r_std)             # solve industry price
    @time solve_mass_entrants(p_std, r_std)     # solve mass of entrants

    # ------------------------------------------------------------------- #
    # (2) with action specific shock and α = 1
    # ------------------------------------------------------------------- #
    p_shock1, r_shock1 = initialize(;c_f_init = c_f_val)
    @time solve_price(p_shock1, r_shock1; shocks = true, α = 1) # solve industry price
    @time solve_mass_entrants(p_shock1, r_shock1)               # solve mass of entrants

    # ------------------------------------------------------------------- #
    # (3) with action specific shock and α = 2
    # ------------------------------------------------------------------- #
    p_shock2, r_shock2 = initialize(;c_f_init = c_f_val)
    @time solve_price(p_shock2, r_shock2; shocks = true, α = 2) # solve industry price
    @time solve_mass_entrants(p_shock2, r_shock2)               # solve mass of entrants

    # ------------------------------------------------------------------- #
    # (4) output moments and results
    # ------------------------------------------------------------------- #
    type = "c_f_"*string(c_f_val)
    compile_moments(p_std, r_std, p_shock1, r_shock1, p_shock2, r_shock2, type)
    plot_decisions(r_std, r_shock1, r_shock2, type)

end
