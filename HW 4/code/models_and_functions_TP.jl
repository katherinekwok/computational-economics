# Author: Katherine Kwok
# Date: October 13, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model.
#
# The code below computes the TRANSITIONS PATHS, and is divided into the
# following sections:
#
#   (0) set up strucs and functions to initialize
#   (1) functions for shooting backward
#   (2) functions for shooting forward

include("model_and_functions_SS.jl")

# ---------------------------------------------------------------------------- #
#  (0) Set up strucs and functions to initialize
# ---------------------------------------------------------------------------- #

# Results_TP: This struct holds the variables for transition path
@everywhere mutable struct TransitionPaths
    TPs::Int64           # number of transition periods
    K_TP::Array{Float64, 1}     # transition path of K
    L_TP::Array{Float64, 1}     # transition path of L
    r_TP::Array{Float64, 1}     # transition of r
    w_TP::Array{Float64, 1}     # transition of w
    b_TP::Array{Float64, 1}     # transition of b
end

# initialize_TP: This function solves for the steady states, then initializes the
#                transition path variables (stored in the Results_TP struc).
function initialize_TP()
    p1, r1, w1, cv1 = solve_model()           # solve θ = 0.11 model - with soc security
    pT, rT, wT, cvT = solve_model(θ_0 = 0.0)  # solve θ = 0 model - no soc security

    @unpack α, δ, μ_r = p1 # unpack prims from θ = 0.11 model
    θ_t = p1.θ             # θ_t = 0.11
    θ_T = pT.θ             # θ_T = 0
    θ_TP = vcat(repeat([θ_t], 29), θ_T) # θ along the transition path

    TPs = 30 # number of transition periods
    K_TP = collect(range(p1.K_0, length = TPs, stop = pT.K_0)) # transition path of K
    L_TP = collect(range(p1.L_0, length = TPs, stop = pT.L_0)) # transition path of L
    r_TP = F_1.(α, K_TP, L_TP)     # transition path of r
    w_TP = F_2.(α,δ, K_TP, L_TP)   # transition path of w
    b_TP = calculate_b.(θ_TP, w_TP, L_TP, μ_r)

    tp = TransitionPaths(TPs, K_TP, L_TP, r_TP, w_TP, b_TP) # transition path of b
    tp # return initialized struc
end

# ---------------------------------------------------------------------------- #
#  (1) Shoot forward
# ---------------------------------------------------------------------------- #

function shoot_forward()

end
