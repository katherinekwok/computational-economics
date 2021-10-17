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
include("models_and_functions_TP.jl")

# ------------------------------------------------------------------------ #
#  (1) compute transition path for unexpected elimination of social security
# ------------------------------------------------------------------------ #

# initialize transition path variables, both steady states prims (p1), results (r1)
# for θ = 0.11 (t=1) and prims (pT), results (rT) for θ = 0 (t=T)
tp, p0, r0, pT, rT = initialize_TP()

pt = initialize_prims() # initialize mutatable struc primitives for current period (t)
K_TP_1 = collect(range(p0.K_0, length = tp.TPs+1, stop = pT.K_0))   # intialize new K transition path
L_TP_1 = collect(range(p0.L_0, length = tp.TPs+1, stop = pT.L_0))      # intialize new L transition path
display_progress(tp, K_TP_1, L_TP_1, p0, pT)

converged = 0

while converged == 0
    # shoot backwards from T to 0 to solve dynamic household programming
    @time shoot_backward(pt, tp, r0, rT)

    # shoot forwards from 0 to T to solve cross-sec distribution for age and time
    @time shoot_forward(pt, tp, r0, K_TP_1, L_TP_1)

    # check progress and convergence
    display_progress(tp, K_TP_1, L_TP_1, p0, pT)
    converged = check_convergence_TP(pt, tp, K_TP_1, L_TP_1)
end

# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #
Plots.plot(a_grid, pol_func[:, index_age_20_h] .- a_grid, label = "High productivity")
Plots.plot!(a_grid, pol_func[:, index_age_20_l] .- a_grid, label = "Low productivity")
