# Author: Katherine Kwok
# Date: December 15, 2021

# This file contains the helper funtions and structures for Problem Set 4
# (for JF's portion) a.k.a. Problem Set 11 overall. The main program implements
# a dynamic model of inventory control.

# ---------------------------------------------------------------------------- #
# (0) Structs and definitions
# ---------------------------------------------------------------------------- #

@with_kw struct Primitives

    α::Float64 = 2          # coefficient on consumption shock
    β::Float64 = 0.99       # discount factor
    λ::Float64 = -4         # stockout penalty (when consumption shock > 0 but 0 inventory)
    i_b::Float64 = 8        # average inventory
    p_s::Float64 = 1        # sales price
    p_r::Float64 = 4        # regular price
    c::Int64 = 1            # consumption shock (0, 1)

    v_bar_0 = fill(1/2, 4, 1)   # initial guess of v bar
    trans = [0.9 0.1; 0.9 0.1]  # transition matrix for prices (regular vs. sales price)

    sim_data_file = "PS4_simdata.csv" # define fill paths
    state_space_file = "PS4_state_space.csv"
    trans_a0_file = "PS4_transition_a0.csv"
    trans_a1_file = "PS4_transition_a1.csv"

end



# ---------------------------------------------------------------------------- #
# (1) Solve the expected value function using implicit equation
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# (2) Solve for expected value function using CCP mapping
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# (3) Calculate MLE of α using nested fixed point algorithm
# ---------------------------------------------------------------------------- #
