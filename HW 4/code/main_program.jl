# Author: Katherine Kwok
# Date: October 13, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model. We
# solve for the transition paths for two cases:
#
#       (0) load functions, packages, compute steady states
#       (1) unexpected elimination of social security
#       (2) expected elimination of social security

# ------------------------------------------------------------------------ #
#  (0) load packages, functions; compute steady states
# ------------------------------------------------------------------------ #

using Distributed, SharedArrays                # load package for running julia in parallel
using CSV, DataFrames                          # load packages for exporting data to csv
using Parameters, Plots, Printf, LinearAlgebra # load standard packages

include("model_and_functions.jl")              # import all functions and strucs
include("TP_and_functions.jl")

p0, r0, w0, cv0 = solve_model()                # solve θ = 0.11 model - with soc security
pT, rT, wT, cvT = solve_model(θ_0 = 0.0)       # solve θ = 0 model    - with no soc security

# ------------------------------------------------------------------------ #
#  (1) compute transition path for unexpected elimination of social security
# ------------------------------------------------------------------------ #

experiment = "unanticipated" # type of experiment
TPs = 30                     # initial number of transition periods

converged_outer = 0          # convergence flag for outer while loop
iter_outer = 1               # iteration counter for outer while loop

while converged_outer == 0

    # initialize transition path variables
    tp = initialize_TP(p0, pT, TPs)
    # initialize mutatable struc primitives for current period (t)
    pt = initialize_prims()

    K_TP_1 = zeros(tp.TPs)   # initialize arrays for new transition path
    L_TP_1 = zeros(tp.TPs)
    converged_inner = 0      # convergence flag for inner while loop
    iter_inner = 1           # iteration counter for inner while loop

    while converged_inner == 0
        # shoot backwards from T-1 to 1 to solve dynamic household programming
        @time shoot_backward(pt, tp, r0, rT)

        # shoot forwards from 1 to T to solve cross-sec distribution for age and time
        @time shoot_forward(pt, tp, r0, K_TP_1, L_TP_1)

        # check progress and convergence
        converged_inner = check_convergence_TP(iter_inner, pt, tp, K_TP_1, L_TP_1, experiment)
        iter_inner += 1   # update iteration counter
    end

    converged_outer, update_TPs = check_convergence_SS(pT, tp, TPs, iter_outer) # check outer convergence, update
    TPs = update_TPs # update number of transition periods
    iter_outer += 1
end

# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #
