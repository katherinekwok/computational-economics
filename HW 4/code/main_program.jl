# Author: Katherine Kwok
# Date: October 25, 2021

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
using JLD                                      # load packages for exporting data
using Parameters, Plots, Printf, LinearAlgebra # load standard packages

include("model_and_functions.jl")              # import all functions and strucs
include("TP_and_functions.jl")

# ------------------------------------------------------------------------ #
#  (1) compute/load steady states
# ------------------------------------------------------------------------ #

# compute steady states if not already computed and stored
if isfile("data/steady_state_T.jld") == false && isfile("data/steady_state_0.jld")
    p0, r0, w0, cv0 = solve_model()           # solve θ = 0.11 model - with soc security
    pT, rT, wT, cvT = solve_model(θ_0 = 0.0)  # solve θ = 0 model    - with no soc security

    save("data/steady_states/steady_state_0.jld", "p0", p0, "r0", r0) # store to speed up future runs
    save("data/steady_states/steady_state_T.jld", "pT", pT, "rT", rT)

# load in primitives and results if already stored
else
    p0 = load("data/steady_states/steady_state_0.jld")["p0"]
    r0 = load("data/steady_states/steady_state_0.jld")["r0"]
    pT = load("data/steady_states/steady_state_T.jld")["pT"]
    rT = load("data/steady_states/steady_state_T.jld")["rT"]
end

# ------------------------------------------------------------------------ #
#  (1) compute transition path for unexpected elimination of social security
# ------------------------------------------------------------------------ #

tp_u, pt_u = solve_algorithm("unanticipated", 70, p0, pT, r0, rT) # start with 70 periods (will converge in ~9 iterations)
save("data/transition_paths/unexpected_transition_path.jld", "tp_u_saved", tp_u)   # save results
summarize_results("unanticipated", tp_u, pt_u, r0, p0)

# ------------------------------------------------------------------------ #
#  (2) compute transition path for expected elimination of social security
# ------------------------------------------------------------------------ #

tp_a, pt_a = solve_algorithm("anticipated", 70, p0, pT, r0, rT; date_imple_input = 21) # start with 70 periods
summarize_results("anticipated", tp_a, pt_a, r0, p0)
