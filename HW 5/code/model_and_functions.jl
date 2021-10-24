# Author: Katherine Kwok (Using some reference code provided by Phil Coyle!)
# Date: October 22, 2021

# This file contains the code for Problem Set 5, where we solve the Krusell-Smith
# model.
#
# The code below is divided into the following sections:
#
#   (0) set up strucs and functions to initialize (including draw shocks)
#   (1) functions for value function iteration
#   (2) functions for simulating capital path
#   (3) functions for estimating regression and checking for convergence

# ------------------------------------------------------------------------ #
#   (0) set up strucs and functions to initialize (including draw shocks)
# ------------------------------------------------------------------------ #

# NOTE: Most of the structs below are just slightly modified versions of Phil's
#       reference code.

# Primitives: This struct holds all the model primitives, including paramter values
#             and the capital and aggregate capital grids, and states (employment
#             and technology shock).
@with_kw struct Primitives
    β::Float64 = 0.99               # discount factor
    α::Float64 = 0.36               # capital share
    δ::Float64 = 0.025              # capital depreciation rate

    k_lb::Float64 = 0.001           # capital lower bound
    k_ub::Float64 = 20.0            # capital upper bound
    n_k::Int64 = 21                 # capital grid size
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0            # aggregate capital lower bound
    K_ub::Float64 = 15.0            # aggregate capital upper bounder
    n_K::Int64 = 11                 # aggregate capital grid size
    K_grid::Array{Float64,1} = range(K_lb, stop = K_ub, length = n_K)

    ϵ_h::Float64 = 0.3271           # labor efficiency when employed (high state)
    ϵ_l::Float64 = 0.0              # labor efficiency when unemployed (low state)
    n_ϵ::Int64 = 2                  # number of employment states
    ϵ_grid::Array{Float64,1} = [ϵ_h, ϵ_l]

    z_g::Float64 = 1.01             # good economy i.e. high aggregate technology shock
    z_b::Float64 = 0.99             # bad economy i.e. low aggregate technology shock
    n_z::Int64 = 2                  # number of economy states i.e. aggregate technology shocks
    z_grid::Array{Float64,1} = [z_g, z_b]
end

# Algorithm: This struct stores all the paramters related to running the Krusell-Smith algorithm.
@with_kw struct Algorithm

    λ::Float64        = 0.5          # adjustment parameter for updating guesses for regression coefficients
    T::Int64          = 11000        # number of aggregate economy shocks to draw
    N::Int64          = 5000         # number of employment shocks to draw per aggregate economy shock
    burn::Int64       = 1000         # number of initial periods to ignore

    tol_vfi::Float64  = 1e-4         # tolerance value for value function iteration
    tol_simulate::Float64 = 1e-4     # tolerance value for simulating capital path
    tol_main::Float64   = 1.0 - 1e-2 # tolerance value for overall convergence
    max_iters::Int64  = 10000        # max number of iterations to run
end

# Shocks: This struct holds the parameters and definitions for transition
#         probabilities between economy/aggregate states and employment states.
@with_kw struct Shocks

    # parameters for transition probabilities
    g_dura::Float64 = 8.0  # average duration of good times
    b_dura::Float64 = 8.0  # average duration of bad times

    ug_dura::Float64 = 1.5 # unemployment duration in good times
    ub_dura::Float64 = 2.5 # unemployment duration in bad times

    u_g::Float64 = 0.04    # fraction of unemployed population in good times
    u_b::Float64 = 0.1     # fraction of employed population in bad times


    # transition probabilities for economy/aggregate state (good times vs. bad times)
    p_gg::Float64 = (g_dura-1.0)/g_dura   # prob of good times to good times
    p_bb::Float64 = (b_dura-1.0)/b_dura   # prob of bad times to bad times
    p_gb::Float64 = 1.0 - p_bb            # prob of good times to bad times
    p_bg::Float64 = 1.0 - p_gg            # prob of bad times to good times


    # transition probabilities for economy/aggregate states and STAYING UNEMPLOYED
    p_gg00::Float64 = (ug_dura-1.0)/ug_dura
    p_bb00::Float64 = (ub_dura-1.0)/ub_dura
    p_bg00::Float64 = 1.25 * p_bb00
    p_gb00::Float64 = 0.75 * p_gg00

    # transition probabilities for economy/aggregate states and STAYING EMPLOYED
    p_gg11::Float64 = 1.0 - (u_g - u_g * p_gg00)/(1.0 - u_g)
    p_bb11::Float64 = 1.0 - (u_b - u_b * p_bb00)/(1.0 - u_b)
    p_bg11::Float64 = 1.0 - (u_b - u_g * p_bg00)/(1.0 - u_g)
    p_gb11::Float64 = 1.0 - (u_g - u_b * p_gb00)/(1.0 - u_b)

    # transition probabilities for economy/aggregate states and BECOMING EMPLOYED
    p_gg01::Float64 = (u_g - u_g * p_gg00)/(1.0-u_g)
    p_bb01::Float64 = (u_b - u_b * p_bb00)/(1.0-u_b)
    p_bg01::Float64 = (u_b - u_g * p_bg00)/(1.0-u_g)
    p_gb01::Float64 = (u_g - u_b * p_gb00)/(1.0-u_b)

    # transition probabilities for economy/aggregate states and BECOMING UNEMPLOYED
    p_gg10::Float64 = 1.0 - p_gg00
    p_bb10::Float64 = 1.0 - p_bb00
    p_bg10::Float64 = 1.0 - 1.25 * p_bb00
    p_gb10::Float64 = 1.0 - 0.75 * p_gg00

    # markov transition matrix
    π_gg::Array{Float64,2}   = [p_gg11 p_gg01; p_gg10 p_gg00]
    π_bg::Array{Float64,2}   = [p_gb11 p_gb01; p_gb10 p_gb00]
    π_gb::Array{Float64,2}   = [p_bg11 p_bg01; p_bg10 p_bg00]
    π_bb::Array{Float64,2}   = [p_bb11 p_bb01; p_bb10 p_bb00]
    Π::Array{Float64,2} = [p_gg * π_gg p_gb * π_gb; p_bg * π_bg p_bb * π_bb]
end

# Results: This struct holds the main results for this program

mutable struct Results
    pol_func::Array{Float64,4}  # policy function for asset/savings
    val_func::Array{Float64,4}  # value function

    a0::Float64                 # regression coeffients for good state
    a1::Float64
    b0::Float64                 # regression coefficients for bad state
    b1::Float64
    R2::Array{Float64,1}        # R^2 values for model fit evaluation
end

# initialize_results: This function initializes the results struct
function initialize_results()
end

# draw_shocks: This function draws a sequence of z (economy/aggregate shocks) and
#              ϵ (employment/idiosyncratic shocks)
function draw_shocks(shocks::Shocks, N::Int64, T::Int64)
    @unpack p_gg, p_bb, m_gg, m_gb, m_bg, m_bb = shocks

    Random.seed!(12345678) # set seed
    dist = Uniform(0, 1)   # distribution to draw shocks from

    z_state = zeros(T)   # sequence of economy/aggregate shocks
    ϵ_state = zeros(N,T) # sequence of employment/idiosyncratic shocks

    z_state[1] = 1       # initialize: assume we start with z_g (good economy/aggregate state)
    ϵ_state[ : , 1] .= 1 # initialize: assume for first z state, everyone is employed


    for t = 2:T # for length of z shock sequence (from 2 onwards)
        z_shock = rand(dist) # draw a z shock

        if z_state[t-1] == 1 && z_shock < p_gg      # if previous z state = good
            z_state[t] = 1                          # and draw < prob of staying good, stay

        elseif z_state[t-1] == 1 && z_shock > p_gg  # if previous z state = good
            z_state[t] = 2                          # and draw > prob of staying good, change

        elseif z_state[t-1] == 2 && z_shock < p_bb  # if previous z state = bad
            z_state[t] = 2                          # and draw < prob of staying bad, stay

        elseif z_state[t-1] == 2 && z_shock > p_bb  # if previous z state = bad
            z_state[t] = 1                          # and draw > prob of staying bad, change
        end

        for n = 1:N # for length of ϵ shock sequence (for each z shock)
            ϵ_shock = rand(dist) # draw a ϵ shock

            if z_state[t-1] == 1 && z_state[t] == 1 # if economy stays good
                p_11 = m_gg[1,1] # prob of staying employed
                p_00 = m_gg[2,2] # prob of staying unemployed

            elseif z_state[t-1] == 1 && z_state[t] == 2 # if economy changes from good to bad
                p_11 = m_gb[1,1] # prob of staying employed
                p_00 = m_gb[2,2] # prob of staying unemployed

            elseif agg_state[t-1] == 2 && agg_state[t] == 1 # if economy changes from bad to good
                p_11 = m_bg[1,1] # prob of staying employed
                p_00 = m_bg[2,2] # prob of staying unemployed

            elseif agg_state[t-1] == 2 && agg_state[t] == 2 # if economy stays bad
                p_11 = m_bb[1,1] # prob of staying employed
                p_00 = m_bb[2,2] # prob of staying unemployed
            end

            if ϵ_state[n,t-1] == 1 && ϵ_shock < p_11     # if prev employed, shock < prob stay employed, stay
                ϵ_state[n,t] = 1
            elseif ϵ_state[n,t-1] == 1 && ϵ_shock > p_11 # if prev employed, shock > prob stay employed, change
                ϵ_state[n,t] = 2
            elseif ϵ_state[n,t-1] == 2 && ϵ_shock < p_00 # if prev unemployed, shock < prob stay employed, stay
                ϵ_state[n,t] = 2
            elseif ϵ_state[n,t-1] == 2 && ϵ_shock > p_00 # if prev unemployed, shock > prob stay employed, change
                ϵ_state[n,t] = 1
            end
        end
    end

    return ϵ_state, z_state
end


# ------------------------------------------------------------------------ #
#   (1) functions for value function iteration
# ------------------------------------------------------------------------ #
