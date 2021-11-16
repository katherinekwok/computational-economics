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

    n_s::Int64           # number of productivity shocks
    s::Array{Float64, 1} # productivity shock on firm
    e::Array{Float64, 1} # firm employment levels given productivity shock

    s_trans_mat::Array{Float64, 2}  # transition matrix for productivity shock
    entrant_dist::Array{Float64, 1} # invariant entrant distribution

    p_lb::Float64  # lower bound on industry price
    p_ub::Float64  # upper bound on industry price

    m_lb::Float64  # lower bound on mass of entrants
    m_ub::Float64  # upper bound on mass of entrants

    n_choice::Int64                # number of firm
    exit_choice::Array{Float64, 1} # vector of choices (stay = 0, exit = 1)
end


# Results: This struct stores the results of the algorithm
mutable struct Results

    pol_func::Array{Float64, 1}  # exit policy function
    val_func::Array{Float64, 1}  # firm's value function

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

    p_lb, p_ub = [0, 1]      # lower and upper bound on industry price
    m_lb, m_ub = [0, 10]     # lower and upper bound on mass of entrants

    n_choice = 2         # number of firm choices
    exit_choice = [0, 1] # vector of choices (stay = 0, exit = 1)

    pol_func = zeros(n_s)    # policy function
    val_func = zeros(n_s)    # value function
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

# bellman: This function encodes the firm's bellman function (benchmark, without
#          any action-specific shocks).
#          For each productivity state, the firm computes the value for staying
#          vs. exiting tomorrow, and chooses the option that returns the max value.
function bellman(prim::Primitives, res::Results)
    @unpack val_func, p = res
    @unpack s_trans_mat, s, θ, β, c_f, n_choice, n_s = prim

    v_next = zeros(n_s)
    x_next = zeros(n_s)

    for (s_index, s_val) in enumerate(s)     # loop through productivity states
        s_prob = s_trans_mat[s_index, :]     # get transition probabilities for current state

        labor = max(0, (p * θ * s_val)^(1/(1-θ)))           # optimal labor choice given s
        profit = p * s_val * (labor^θ) - labor - p * c_f    # profit given s

        v_stay = profit + β * s_prob' * val_func            # value for staying tomorrow
        v_exit = profit                                     # value for exiting tomorrow

        best_choice = findmax([v_stay, v_exit])  # returns [max val, max index]
        v_next[s_index] = best_choice[1]         # store max val
        x_next[s_index] = best_choice[2] - 1     # store corresponding choice
    end

    v_next, x_next
end

# bellman_shocks: This function specifies the bellman function with action-specific
#                 shocks. It is very similar to the benchmark bellman, except
#                 the continuation value for firms that stay is defined differently.
function bellman_shocks(prim::Primitives, res::Results, α::Int64)
    @unpack val_func, p = res
    @unpack s_trans_mat, s, θ, β, c_f, n_choice, n_s = prim

    v_next = zeros(n_s)
    x_next = zeros(n_s)

    for (s_index, s_val) in enumerate(s)     # loop through productivity states
        s_prob = s_trans_mat[s_index, :]     # get transition probabilities for current state

        labor = max(0, (p * θ * s_val)^(1/(1-θ)))           # optimal labor choice given s
        profit = p * s_val * (labor^θ) - labor - p * c_f    # profit given s

        v_stay = profit + β * s_prob' * val_func            # value for staying tomorrow
        v_exit = profit                                     # value for exiting tomorrow

        # calculate ex-ante value function using log-sum-exp trick
        # we subtract and then add c because the value function could be too big
        c = max(α*v_stay, α*v_exit)
        γ = MathConstants.eulergamma
        utility = (γ/α) + (1/α)*log(exp(α * v_stay - c) + exp(α * v_exit - c)) + c

        # calculate choice probability of choosing to stay (without action specific shock, just 0 or 1)
        # we subtract c because the value function could be too big
        choice_prob = exp(α * v_stay - c)/sum(exp(α * v_stay - c) + exp(α * v_exit - c))

        v_next[s_index] = utility                # store value function
        x_next[s_index] = choice_prob            # store choice probability
    end

    v_next, x_next
end

# solve_vfi: This function calls the bellman() function to iterate the value
#            function until convergence.
function solve_vfi(prim::Primitives, res::Results, shocks::Bool, α::Int64; tol = 1e-6)
    n = 0         # counter for iteration
    converged = 0 # indicator for convergence

    while converged == 0  # keep iterating until we error less than tolerance value

        if shocks == false                         # if not including action-specific shocks
            v_next, x_next = bellman(prim, res)    # call benchmark bellman
        else
            v_next, x_next = bellman_shocks(prim, res, α) # if including shocks
        end
        v_err = sum(abs.(v_next.-res.val_func))    # get sup norm of val and pol
        x_err = sum(abs.(x_next.-res.pol_func))

        if v_err < tol && x_err < tol  # if error less than tolerance
            converged = 1              # we have converged
        end
        res.val_func = v_next # update val func
        res.pol_func = x_next
        n += 1                # update loop counter
    end
    println("-----------------------------------------------------------------------")
    println("       Value function converged in ", n, " iterations.")
    println("-----------------------------------------------------------------------")
end

# solve_entrant_val: This function encodes the entrant's valuation based on the
#                    results of solving the value function iteration.
function solve_entrant_val(prim::Primitives, res::Results)
    sum(res.val_func' * prim.entrant_dist)
end

# solve_price: This function calls the vfi (value function iteration) function
#              entrant value function to solve for the price that clears the entry
#              market.
# NOTE: By default, this function runs the benchmark version of bellman and vfi.
function solve_price(prim::Primitives, res::Results; tol::Float64 = 1e-3, shocks::Bool = false, α::Int64 = 0)
    n = 0         # counter for iteration
    converged = 0 # indicator for convergence

    while converged == 0  # keep iterating until we error less than tolerance value

        solve_vfi(prim, res, shocks, α)             # value function iteration
        entrant_val = solve_entrant_val(prim, res)  # get entrant's value

        # calculate abs diff, p * c_e is steady state entrant's value
        ss_entrant_val = res.p * prim.c_e
        abs_diff = abs(entrant_val - ss_entrant_val)
        adjustment = 200

        # use bisection method to update price if not converged

        if abs_diff < tol                              # if converged
            converged = 1                              # update convergence flag!

        elseif entrant_val > ss_entrant_val            # if current entrant val too big
            prim.p_ub = res.p                          # p too high, lower the price upper bound
            p_new = (prim.p_ub + prim.p_lb)/2          # to drop p
            println("-----------------------------------------------------------------------")
            @printf "          Entrant val too big; drop price from %.4f to %.4f \n" res.p p_new
            println("-----------------------------------------------------------------------")
            res.p = p_new
        else                                           # if current entrant val too small
            prim.p_lb = res.p                          # p too low, raise the price lower bound
            p_new = (prim.p_ub + prim.p_lb)/2          # to raise p
            println("-----------------------------------------------------------------------")
            @printf "       Entrant val too small; raise price from %.4f to %.4f \n" res.p p_new
            println("-----------------------------------------------------------------------")
            res.p = p_new
        end

        n += 1  # update loop counter
    end
    println("-----------------------------------------------------------------------")
    @printf "  Solving for industry price converged in %d iterations with p = %.4f\n" n res.p
    println("-----------------------------------------------------------------------")
end

# ------------------------------------------------------------------------ #
#  (2) solve for labor market clearing labor demand and supply
# ------------------------------------------------------------------------ #

# solve for stationary distribution

# solve for labor demand and supply


# ------------------------------------------------------------------------ #
#  (3) display and plot results
# ------------------------------------------------------------------------ #
