# Author: Katherine Kwok
# Date: November 16, 2021

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
    stat_dist = ones(n_s)/n_s   # stat distribution

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

# calc_profit: This is a function that calculates profit for a given firm
function calc_profit(p::Float64, s_val::Float64, labor::Float64, θ::Float64, c_f::Float64)
    p * s_val * (labor^θ) - labor - p * c_f
end

# calc_labor: This is a function that calculates the firm's labor demand
function calc_labor(p::Float64, s_val::Float64, θ::Float64)
    max(0, (p * θ * s_val)^(1/(1-θ)))
end

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

        labor = calc_labor(p, s_val, θ)                     # optimal labor choice given s
        profit = calc_profit(p, s_val, labor, θ, c_f)       # profit given s

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

        labor = calc_labor(p, s_val, θ)                     # optimal labor choice given s
        profit = calc_profit(p, s_val, labor, θ, c_f)       # profit given s

        v_stay = profit + β * s_prob' * val_func            # value for staying tomorrow
        v_exit = profit                                     # value for exiting tomorrow

        # calculate ex-ante value function using log-sum-exp trick
        # we subtract and then add c because the value function could be too big
        c = max(α * v_stay, α * v_exit)
        γ = MathConstants.eulergamma
        utility = (γ/α) + (1/α) * (c + log(exp(α * v_stay - c) + exp(α * v_exit - c)))

        # calculate choice probability of choosing to exit (without action specific shock, just 0 or 1)
        # we subtract c because the value function could be too big
        choice_prob = exp(α * v_exit - c)/sum(exp(α * v_stay - c) + exp(α * v_exit - c))

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
function solve_price(prim::Primitives, res::Results; tol::Float64 = 1e-6, shocks::Bool = false, α::Int64 = 0)
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

# update_stat_dist: This function iterates through today and tomorrow's productivity
#                   states to get the distribution of firms that stay and enter given
#                   the state today and tomorrow.
function update_stat_dist(prim::Primitives, res::Results)
    @unpack s, n_s, s_trans_mat, entrant_dist = prim
    @unpack pol_func, m, stat_dist = res

    stay_dist = zeros(n_s)  # initiate update distribution for staying firms
    enter_dist = zeros(n_s) # initiative update distribution for entering firms

    # loop through productivity states today
    for (s_i, s_val) in enumerate(s)
        # loop through productivity states tomorrow
        for (sp_i, sp_val) in enumerate(s)

            # transition probability of firm staying given state
            stay_dist[sp_i] += (1-pol_func[s_i])*s_trans_mat[s_i, sp_i]*stat_dist[s_i]
            # transition probability of firm entering given state
            enter_dist[sp_i] += (1-pol_func[s_i])*s_trans_mat[s_i, sp_i]*entrant_dist[s_i]
        end
    end

    # updated distribution (m is mass of entrants)
    stay_dist .+ (m .* enter_dist)
end

# solve_stat_dist: This function solves for the stationary distribution, basically
#                  by updating the distribution using the solved policy function
#                  until convergence.
function solve_stat_dist(prim::Primitives, res::Results; tol = 1e-5)
    converged = 0
    n = 0

    while converged == 0
        new_dist = update_stat_dist(prim, res) # update stationary distribution
        max_diff = maximum(abs.(new_dist .- res.stat_dist)) # get max difference between new and old dist

        if max_diff < tol        # check convergence condition
            converged = 1
        end
        res.stat_dist = new_dist # update stationary distribution
        n+=1
    end
    println("-----------------------------------------------------------------------")
    @printf "      Solving stationary distribution converged in %d iterations\n" n
    println("-----------------------------------------------------------------------")
end

# agg_labor_market: This function computes the aggregate labor demand and supply
#                   using the stationary distribution, entry distribution, and
#                   mass of entrants
function agg_labor_market(prim::Primitives, res::Results)
    @unpack s, θ, c_f, A, entrant_dist = prim
    @unpack stat_dist, p, m = res

    agg_labor_d = 0.0  # initialize aggregate labor demand by firms
    agg_profits = 0.0  # initialize aggregate profits

    for (s_index, s_val) in enumerate(s)
        # sum up labor demand by firms (staying and entering)
        labor = calc_labor(p, s_val, θ)
        agg_labor_d += labor * stat_dist[s_index]        # staying
        agg_labor_d += m * labor * entrant_dist[s_index] # entering

        # sum up profits (staying and entering)
        profit = calc_profit(p, s_val, labor, θ, c_f)
        agg_profits += profit * stat_dist[s_index]        # staying
        agg_profits += m * profit * entrant_dist[s_index] # entering
    end

    agg_labor_s = 1/A - agg_profits # aggregate labor supply (implied by FOC)

    agg_labor_d, agg_labor_s
end

# solve_mass_entrants: This function calls the functions to solve for the stationary
#                      distribution and calculate aggregate labor supply and
#                      demand. It continues to iterature until m (mass of entrants)
#                      clears the labor market (agg supply = agg demand)
function solve_mass_entrants(prim::Primitives, res::Results; tol = 1e-5)

    converged = 0   # convergence flag
    n = 0

    while converged == 0

        solve_stat_dist(prim, res)  # solve for stationary distribution

        # calculate aggregate labor supply and demand
        agg_labor_d, agg_labor_s = agg_labor_market(prim, res)
        abs_diff = abs(agg_labor_s - agg_labor_d) # get abs difference

        if abs_diff < tol                              # if converged
            converged = 1                              # update convergence flag!

        elseif agg_labor_d > agg_labor_s            # if demand > supply
            prim.m_ub = res.m                       # m too high, lower m upper bound
            m_new = (prim.m_ub + prim.m_lb)/2       # to drop m
            println("-----------------------------------------------------------------------")
            @printf "    Supply exceeds demand; drop mass of entrants from %.4f to %.4f \n" res.m m_new
            println("-----------------------------------------------------------------------")
            res.m = m_new
        else                                           # if supply > demand
            prim.m_lb = res.m                          # m too low, raise m lower bound
            m_new = (prim.m_ub + prim.m_lb)/2          # to raise m
            println("-----------------------------------------------------------------------")
            @printf "   Demand exceeds supply; raise mass of entrants from %.4f to %.4f \n" res.m m_new
            println("-----------------------------------------------------------------------")
            res.m = m_new
        end
        n+=1
        if n == 5
            converged = 1
        end
    end
    println("-----------------------------------------------------------------------")
    @printf "  Solving for mass of entrants converged in %d iterations with m = %.4f\n" n res.m
    println("-----------------------------------------------------------------------")

end

# ------------------------------------------------------------------------ #
#  (3) display and plot results
# ------------------------------------------------------------------------ #

# agg_labor: This function calculates the aggregate labor demand for incumbent (staying)
#            firms and entering firms using the stationary distribution and
#            entrant distribution.
function agg_labor(prim::Primitives, res::Results)
    @unpack s, θ, entrant_dist = prim
    @unpack stat_dist, p, m = res

    agg_labor_stay = 0.0  # initialize aggregate labor demand by staying firms
    agg_labor_enter = 0.0  # initialize aggregate labor demand by entering firms


    for (s_index, s_val) in enumerate(s)
        # sum up labor demand by firms (staying and entering)
        labor = calc_labor(p, s_val, θ)
        agg_labor_stay += labor * stat_dist[s_index]        # staying
        agg_labor_enter += m * labor * entrant_dist[s_index] # entering
    end

    agg_labor_stay, agg_labor_enter
end


# compute_moments: This function computes model moments after solving the
#                  Hopenhayn-Rogerson algorithm, and then compiles them into
#                  a neat dataframe.
function compute_moments(prim::Primitives, res::Results)
    @unpack stat_dist, pol_func, m, p = res

    # mass of incumbents, exits, entrants (recal pol_func = 1 means exit)
    mass_incumbents = (1 .- pol_func)' * stat_dist
    mass_exits = pol_func' * stat_dist
    mass_entrants = m

    # aggregate labor demand, incumbents, entrants
    demand_incumbents, demand_entrants = agg_labor(prim, res)
    demand_aggregate = demand_incumbents + demand_entrants

    # fraction of labor in entrant
    fraction_entrant = demand_entrants/demand_aggregate

    # make data frame
    moments = DataFrame(Industry_price = p,
                        Mass_of_entrants = mass_entrants,
                        Mass_of_incumbents = mass_incumbents,
                        Mass_of_exits = mass_exits,
                        Entrant_labor_demand = demand_entrants,
                        Incumbent_labor_demand = demand_incumbents,
                        Aggregate_labor_demand = demand_aggregate,
                        Fraction_of_labor_in_entrants = fraction_entrant)
    moments  # return data frame
end

function plot_decisions(prim::Primitives, res::Results)
end
