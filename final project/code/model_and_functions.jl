# Author: Katherine Kwok
# Date: November 27, 2021

# This file contains the declaration of structs and helper functionst that are
# used to replicate the code in Arellano (2008).


# ---------------------------------------------------------------------------- #
# (0) Initialize primitives and results structs
# ---------------------------------------------------------------------------- #

@with_kw struct Primitives

    r::Float64 = 0.017 # risk free interest rate
    σ::Float64 = 2     # risk aversion
    ρ::Float64 = 0.945 # coefficient for log-normal AR(1) process for income
    η::Float64 = 0.025 # variance for error term in log-normal AR(1) process for income

    β::Float64 = 0.953 # discount factor
    θ::Float64 = 0.282 # re-entry probability after defaulting

    y_hat::Float64 = 0.969 # NOTE: Not sure about this one

    # note should define B grid and y grid

end


mutable struct Results

    q::Array{Float64, 2}  # bond price schedule

    pol_func_c::Array{Float64, 2} # consumption
    pol_func_B::Array{Float64, 2} # assets

    val_func::Array{Float64, 2}   # value function

    A::Array{Float64}             # income set where paying back is optimal
    D::Array{Float64}             # income set where defaulting is optimal

end

# ---------------------------------------------------------------------------- #
# (1) Value function iteration
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# (2) Bond price converence
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# (3) Business cycle statistics
# ---------------------------------------------------------------------------- #
