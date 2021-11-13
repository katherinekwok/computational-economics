# Author: Katherine Kwok
# Date: November 2, 2021

# This file contains the code for Problem Set 6, where we solve the Hopenhayn-Rogerson
# model of firm dynamics.
#
# The code below is divided into the following sections:
#
#   (0) set up strucs and functions to initialize
#   (1) solve for entry market clearing price
#   (2) solve for labor market clearing labor demand and supply
#   (3) display and plot results


# ------------------------------------------------------------------------ #
#  (0) initialize algorithm
# ------------------------------------------------------------------------ #

# Primitives: This struct stores the primitives of the model
mutable struct Primitives

    β::Float64    # firm discount rate for profits
    θ::Float64    # persistence value of shock
    A::Float64    # employment to population ratio

    c_f::Float64  # fixed costs for staying in market
    c_e::Float64  # entry costs for entering market

    n_s::Float64         # number of productivity shocks
    s::Array{Float64, 1} # productivity shock on firm
    e::Array{Float64, 1} # firm employment levels given productivity shock

    s_trans_mat::Array{Float64, 2}  # transition matrix for productivity shock
    entrant_dist::Array{Float64, 1} # invariant entrant distribution

    p_lb::Float64  # lower bound on industry price
    p_ub::Float64  # upper bound on industry price

    m_lb::Float64  # lower bound on mass of entrants
    m_ub::Float64  # upper bound on mass of entrants

    n_choice::Float64              # number of firm
    exit_choice::Array{Float64, 1} # vector of choices (stay = 0, exit = 1)
end


# Results: This struct stores the results of the algorithm
mutable struct Results

    pol_func::Array{Float64, 2}  # exit policy function - 2D (for each productivity state)
    val_func::Array{Float64, 2}  # firm's value function - 2D (for each productivity state)

    stat_dist::Array{Float64, 1} # stationary distribution of firms

    p::Float64 # industry price
    m::Float64 # mass of entrants
end

# initialize: This function initializes the primitives and results
function initialize(;p_init = 0.5, m_init = 2.75, c_f_init = 10)

    β = 0.8          # firm discount rate for profits
    θ = 0.64         # persistence value of shock
    A = 1/200        # employment to population ratio

    c_f = c_f_init   # fixed costs for staying in market
    c_e = 5          # entry costs for entering market

    n_s = 5                                 # number of productivity shocks
    s = [3.98e-4, 3.58, 6.82, 12.18, 18.79] # shock on firm
    e = [1.3e-9, 10, 60, 300, 1000]         # firm employment levels given shock

    # transition matrix for firm shock
    s_trans_mat = [0.6598 0.2600 0.0416 0.0331 0.0055;
                   0.1997 0.7201 0.0420 0.0326 0.0056;
                   0.2000 0.2000 0.5555 0.0344 0.0101;
                   0.2000 0.2000 0.2502 0.3397 0.0101;
                   0.2000 0.2000 0.2500 0.3400 0.0100]

    # invariant entrant distribution
    entrant_dist = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]

    p_lb, p_ub = [0, 1]  # lower and upper bound on industry price
    m_lb, m_ub = [0, 10] # lower and upper bound on mass of entrants

    n_choice = 2         # number of firm
    exit_choice = [0, 1] # vector of choices (stay = 0, exit = 1)

    pol_func = zeros(n_choice, n_s)                     # policy function
    val_func = zeros(n_choice, n_s)                     # value function
    stat_dist = ones(n_choice * n_s)/(n_choice * n_s)   # stat distribution

    p = p_init  # industry price
    m = m_init  # mass of entrant

    # feed in the initial values
    prim = Primitives(β, θ, A, c_f, c_e, n_s, s, e, s_trans_mat, entrant_dist, p_lb, p_ub,
                      m_lb, m_ub, n_choice, exit_choice)

    res  = Results(pol_func, val_func, stat_dist, p, m)

    prim, res # return initialized structs
end

# ------------------------------------------------------------------------ #
#  (1) solve for entry market clearing price
# ------------------------------------------------------------------------ #

# solve value function iteration

# bellman: this function encodes the firm's value function
function bellman()
    @unpack val_func = res                       # unpack value function
    @unpack a_grid, β, α, na, s, t_matrix, ns = prim # unpack model primitives
    v_next = zeros(na, ns)                        # next guess of value function

    for (s_index, s_val) in enumerate(s)         # loop through possible employment states
        s_prob = t_matrix[s_index, :]            # get transition probabilities for current state

    end
    v_next # return next guess of value function
end

# V_iterate: is the value function iteration loop, which calls the Bellman
# function repeatedly until we reach convergence.
function V_iterate(prim::Primitives, res::Results, q::Float64, tol::Float64 = 1e-5, err::Float64 = 100.0)
    n = 0         # counter for iteration
    converged = 0 # indicator for convergence

    println("-----------------------------------------------------------------------")
    @printf "      Starting value function iteration for bond price  %.6f \n" q
    println("-----------------------------------------------------------------------")
    while converged == 0  # keep iterating until we error less than tolerance value

        v_next = Bellman(prim, res, q)                                 # call Bellman
        err = abs.(maximum(v_next.-res.val_func))/abs(maximum(v_next)) # check for error

        if err < tol          # if error less than tolerance
            converged = 1     # we have converged
        end
        res.val_func = v_next # update val func
        n += 1                # update loop counter

    end
    println("-----------------------------------------------------------------------")
    println("       Value function converged in ", n, " iterations.")
    println("-----------------------------------------------------------------------")
end

# solve entrant's value


# ------------------------------------------------------------------------ #
#  (2) solve for labor market clearing labor demand and supply
# ------------------------------------------------------------------------ #


# solve for stationary distribution

# solve for labor demand and supply


# ------------------------------------------------------------------------ #
#  (3) display and plot results
# ------------------------------------------------------------------------ #
