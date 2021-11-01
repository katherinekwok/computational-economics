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


##
# ------------------------------------------------------------------------ #
#   (0) set up strucs and functions to initialize (including draw shocks)
# ------------------------------------------------------------------------ #

# NOTE: Most of the structs below are just slightly modified versions of Phil's
#       reference code. The main changes I made are variable names, adding comments,
#       reducing repetitive code, and some reorganization.

# Primitives: This struct holds all the model primitives, including paramter values
#             and the capital and aggregate capital grids, and states (employment
#             and technology shock).
@with_kw struct Primitives
    β::Float64 = 0.99               # discount factor
    α::Float64 = 0.36               # capital share
    δ::Float64 = 0.025              # capital depreciation rate

    k_lb::Float64 = 0.001           # capital lower bound
    k_ub::Float64 = 25              # capital upper bound
    n_k::Int64 = 26                 # capital grid size
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0            # aggregate capital lower bound
    K_ub::Float64 = 15.0            # aggregate capital upper bound
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

    tol_vfi::Float64  = 1e-4
    tol_coef::Float64 = 1e-4
    tol_r2::Float64   = 1.0 - 1e-2      # tolerance value for overall convergence
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
    R2::Array{Float64, 1}       # R^2 values for model fit evaluation
end


# draw_shocks: This function draws a sequence of z (economy/aggregate shocks) and
#              ϵ (employment/idiosyncratic shocks)

function draw_shocks(shocks::Shocks, algo::Algorithm)
    @unpack N, T = algo
    @unpack p_gg, p_bb, π_gg, π_gb, π_bg, π_bb = shocks

    Random.seed!(12345678) # set seed
    dist = Distributions.Uniform(0, 1) # distribution to draw shocks from

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
                p_11 = π_gg[1,1] # prob of staying employed
                p_00 = π_gg[2,2] # prob of staying unemployed

            elseif z_state[t-1] == 1 && z_state[t] == 2 # if economy changes from good to bad
                p_11 = π_gb[1,1] # prob of staying employed
                p_00 = π_gb[2,2] # prob of staying unemployed

            elseif z_state[t-1] == 2 && z_state[t] == 1 # if economy changes from bad to good
                p_11 = π_bg[1,1] # prob of staying employed
                p_00 = π_bg[2,2] # prob of staying unemployed

            elseif z_state[t-1] == 2 && z_state[t] == 2 # if economy stays bad
                p_11 = π_bb[1,1] # prob of staying employed
                p_00 = π_bb[2,2] # prob of staying unemployed
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

# initialize_results: This function initializes the results struct
function initialize_results(prim::Primitives)
    @unpack n_k, n_ϵ, n_K, n_z = prim

    pol_func = zeros(n_k, n_ϵ, n_K, n_z)  # policy function for asset/savings
    val_func = zeros(n_k, n_ϵ, n_K, n_z)  # value function

    a0 = 0.095 # regression coeffients for good state; initialize with guess given by handout
    a1 = 0.999
    b0 = 0.085 # regression coefficients for bad state; initialize with guess given by handout
    b1 = 0.999
    R2 = zeros(n_z) # R^2 value for model fit evaluation (one for each regression)

    res = Results(pol_func, val_func, a0, a1, b0, b1, R2)
    res # return initilized struct
end

# initialize: This function initializes all the relevant structs for the algorithm,
#             calls the function to initialize results struct, and draws a sequence
#             of shocks.
function initialize()

    prim = Primitives()
    algo = Algorithm()
    resu = initialize_results(prim)

    shocks = Shocks()
    ϵ_seq, z_seq = draw_shocks(shocks, algo)

    prim, algo, resu, shocks, ϵ_seq, z_seq    # return all initialized structs
end

##
# ------------------------------------------------------------------------ #
#   (1) functions for value function iteration
# ------------------------------------------------------------------------ #

# get_index: This function gets the index of a value in a given grid, allowing
#            the index to be in between integers.
function get_index(val::Float64, grid::Array{Float64,1})

    n = length(grid)    # get length of grid
    index = 0

    if val <= grid[1]     # get index for value <= grid minimum
        index = 1
    elseif val >= grid[n] # get index for value >= grid maximum
        index = n
    else
        index_upper = findfirst(x->x>val, grid)   # get index for value in between
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    index
end


# calc_mean_K: This function calculates the aggregate capital K, given the z state,
#              using the aggregate capital law of motion.
function calc_K(z_state::Int64, K_today::Float64, res::Results)
    @unpack a0, a1, b0, b1 = res

    if z_state == 1
        K_tomorrow = a0 + a1*log(K_today)
    elseif z_state == 2
        K_tomorrow = b0 + b1*log(K_today)
    end
    exp(K_tomorrow)
end

# calc_L: This function calculates the aggregate L using ϵ and z today
function calc_L(z_state::Int64, ϵ_today::Float64, shocks::Shocks)
    @unpack u_g, u_b = shocks

    if z_state == 1
        L_today = ϵ_today * (1 - u_g)
    elseif z_state == 2
        L_today = ϵ_today * (1 - u_b)
    end
    L_today
end

# calc_r: This function returns the interest rate r given K, L, z
function calc_r(K::Float64, L::Float64, z::Float64, α::Float64)
    (1-α) * z * (K/L)^α
end

# calc_w: This function returns the wage rate w given K, L, z
function calc_w(K::Float64, L::Float64, z::Float64, α::Float64)
    α * z * (K/L)^(α-1)
end

# utility: This is the utility function
function utility(c::Float64)
    if c > 0
        u = log(c) # apply log if c is positive
    else
        u = -1/eps()
    end
    u
end

# interpolate_bellman: This function uses the julia interpolation package to
#                      solve the bellman function for a given set of (k, ϵ, K, z)
#
# NOTE: This was initially a part of the Bellman function provided by Phil.
#
function interpolate_bellman(shocks::Shocks, prim::Primitives, res::Results,
    row::Int64, i_Kp::Float64, r_today::Float64, w_today::Float64, k_today::Float64,
    ϵ_today::Float64)

    @unpack Π = shocks
    @unpack δ, β, k_grid = prim
    @unpack val_func = res

    k_interp = interpolate(k_grid, BSpline(Linear()))   # define interpolation function for k
    v_interp = interpolate(val_func, BSpline(Linear())) # define interpolation function for v

    # We are defining the continuation value. Notice that we are interpolating over k and K.
    v_tomorrow(i_kp) = Π[row,1]*v_interp(i_kp,1,i_Kp,1) +
                       Π[row,2]*v_interp(i_kp,2,i_Kp,1) +
                       Π[row,3]*v_interp(i_kp,1,i_Kp,2) +
                       Π[row,4]*v_interp(i_kp,2,i_Kp,2)


    # We are now going to solve the HH's problem (solve for k).
    # We are defining a function val_func as a function of the agent's capital choice.
    budget = r_today * k_today + w_today * ϵ_today + (1.0 - δ) * k_today
    val_func(i_kp) = utility(budget - k_interp(i_kp)) +  β * v_tomorrow(i_kp)

    obj(i_kp) = -val_func(i_kp)                # minimization problem
    lowerbound = 1.0
    upperbound = get_index(budget, k_grid)

    # Then, we are going to maximize the value function using an optimization routine.
    # Note: Need to call in optimize to use this package.
    opt = optimize(obj, lowerbound, upperbound)

    k_tomorrow = k_interp(opt.minimizer[1])
    v_today = -opt.minimum

    k_tomorrow, v_today
end

# bellman: This function calls helper functions to solve the household dynamic
#          programming problem, using interpolation and function minimization.
function bellman(prim::Primitives, res::Results, shocks::Shocks)

    @unpack n_k, n_ϵ, n_K, n_z, k_grid, ϵ_grid, K_grid, z_grid, α, ϵ_h = prim
    @unpack pol_func, val_func = res

    pol_func_up = zeros(n_k, n_ϵ, n_K, n_z) # initialize array to updated pol and val func
    val_func_up = zeros(n_k, n_ϵ, n_K, n_z)

    for (i_z, z_today) in enumerate(z_grid)      # for each economy/aggregate state
        for (i_K, K_today) in enumerate(K_grid)  # for each aggregate capital today

            # calculate mean K tomorrow using law of motion
            K_tomorrow = calc_K(i_z, K_today, res)
            # get index of K tomorrow in K grid (likely not an integer!)
            i_Kp = get_index(K_tomorrow, K_grid)

            for (i_ϵ, ϵ_today) in enumerate(ϵ_grid)   # for each idiosyncratic employment state
                row = i_ϵ + n_ϵ*(i_z-1)               # get index for markov transition matrix

                L_today = calc_L(i_z, ϵ_h, shocks)          # get aggregate L
                w_today = calc_w(K_today, L_today, z_today, α)  # get wage rate
                r_today = calc_r(K_today, L_today, z_today, α)  # get interest rate

                for (i_k, k_today) in enumerate(k_grid) # for each asset level

                    # use interpolation to solve Bellman for k tomorrow, v today
                    k_tomorrow, v_today = interpolate_bellman(shocks, prim, res,
                    row, float(i_Kp), r_today, w_today, k_today, ϵ_today)

                    pol_func_up[i_k, i_ϵ, i_K, i_z] = k_tomorrow
                    val_func_up[i_k, i_ϵ, i_K, i_z] = v_today

                end
            end
        end
    end
    pol_func_up, val_func_up # update pol and val functions
end

# value_function_iteration: This function solves the household dynamic programming
#                           problem using the bellman funciton and interpolation.

function value_function_iteration(prim::Primitives, res::Results, shocks::Shocks, algo::Algorithm)
    @unpack max_iters, tol_vfi = algo
    converged = 0 # convergence flag
    iters = 1     # iteration counter

    println("-----------------------------------------------------------------------")
    @printf "                 Starting value function iteration\n"
    println("-----------------------------------------------------------------------")

    while converged == 0 && iters < max_iters
        pol_func_update, val_func_update = bellman(prim, res, shocks) # solve for val and pol func using bellman
        error = maximum(abs.(val_func_update .- res.val_func))        # calculate max absolute error

        res.pol_func = pol_func_update # update val and pol func
        res.val_func = val_func_update

        if iters % 10 == 0 # update every 10 iterations
            println("-----------------------------------------------------------------------")
            @printf "       Completed %d iterations and error is %.4f. Continuing...\n" iters error
            println("-----------------------------------------------------------------------")
        end

        if error < tol_vfi # if error less than tol value, converged!
            converged = 1
            println("-----------------------------------------------------------------------")
            @printf "       Value function iteration converged after %d iterations.\n" iters
            println("-----------------------------------------------------------------------")
        end
        iters += 1 # update iteration counters
    end

    if iters == max_iters
        println("-----------------------------------------------------------------------")
        @printf "  Value function iteration did not converge. Stopped at max iter = %d.\n" max_iters
        println("-----------------------------------------------------------------------")
    end
end

##
# ------------------------------------------------------------------------ #
#   (2) functions for simulating capital path
# ------------------------------------------------------------------------ #

# simulate_capital_path: This function simulates the capital path for a random
#                        draw of ϵ and z shocks, using the policy functions
#                        computed in value funciton iteration.

function simulate_capital_path(prim::Primitives, res::Results, algo::Algorithm,
    ϵ_seq::Array{Float64, 2}, z_seq::Array{Float64, 1}; K_ss::Float64 = 11.55)

    @unpack n_k, n_ϵ, n_K, n_z, k_grid, ϵ_grid, K_grid, z_grid = prim
    @unpack N, T = algo
    @unpack pol_func = res

    K_yesterday = K_ss              # initialize K yesterday with K_ss
    K_today = 0.0                   # initialize K today
    k_yesterday = repeat([K_ss], N) # initialize k yesterday with K_ss

    K_path = zeros(T, 3) # agg capital path (colums are K today, K tomorrow, z state today)

    println("-----------------------------------------------------------------------")
    @printf "                 Starting capital path simulation.\n"
    println("-----------------------------------------------------------------------")


    for time in 1:T
        z_shock = z_seq[time]   # draw economy/aggregate shocks

        for person_index in 1:N
            ϵ_shock = ϵ_seq[person_index, time] # draw idiosyncratic shock for person and time

            pol_z_ϵ = pol_func[:, Integer(ϵ_shock), :, Integer(z_shock)]  # get policy function for z, ϵ
            k_interp = interpolate(pol_z_ϵ, BSpline(Linear()))             # define interpolation function for k

            i_k = get_index(k_yesterday[person_index], k_grid) # get index of yesterday's capital for person
            i_K = get_index(K_yesterday, K_grid)               # get index of yesterday's aggregate capital

            k_yesterday[person_index] = k_interp(i_k, i_K) # interpolate k today (store into array for tomorrow)
            K_today += k_yesterday[person_index]           # add k to aggregate K today
        end

        K_yesterday = K_today/N # update aggregate K today (store into array for tomorrow)
        K_today = 0.0           # reset K today for next time period

        K_path[time, 1] = K_yesterday # store aggregate capital and z state for today
        K_path[time, 3] = z_shock

        if time > 1 # store the next K corresponding to each K today (useful for regression)
            K_path[time - 1, 2] = K_yesterday
        end
    end

    println("-----------------------------------------------------------------------")
    @printf "                 Simulating capital path complete.\n"
    println("-----------------------------------------------------------------------")
    K_path
end

##
# ------------------------------------------------------------------------ #
#   (3) functions for estimating regression and checking for convergence
# ------------------------------------------------------------------------ #

# estimate_regression: This function estimates an AR model for good and states
#                      and bad states separately, using the simulated capita path.
function estimate_regression(K_path::Array{Float64, 2}, algo::Algorithm, res::Results)
    @unpack burn, T = algo

    # burn some amount of initial capital path, drop last K because it doesn't have a K next
    K_path = K_path[burn:T-1, :]

    # convert K_path to dataframe and subset by z state
    K_path_df = DataFrame("K_today" => K_path[:, 1], "K_tomorrow" => K_path[:, 2], "z_today" => K_path[:, 3])
    K_path_g = filter(:z_today => z_today -> z_today == 1, K_path_df) # good state
    K_path_b = filter(:z_today => z_today -> z_today == 2, K_path_df) # bad state

    # good state regression
    good_state = lm(@formula(log(K_tomorrow) ~ log(K_today)), K_path_g)
    a0_new = coef(good_state)[1]    # extra coefficients
    a1_new = coef(good_state)[2]
    res.R2[1] = adjr2(good_state)      # get model fit R^2 estimate

    # bad state regression
    bad_state = lm(@formula(log(K_tomorrow) ~ log(K_today)), K_path_b)
    b0_new = coef(bad_state)[1]    # extra coefficients
    b1_new = coef(bad_state)[2]
    res.R2[2] = adjr2(bad_state)      # get model fit R^2 estimate

    a0_new, a1_new, b0_new, b1_new # return estimates
end

# check_convergence: This function checks if the estimated coefficients are
#                    close enough. If so, end while loop. If not, update and
#                    and continue.

function check_convergence(res::Results, algo::Algorithm, a0_new::Float64,
    a1_new::Float64, b0_new::Float64, b1_new::Float64, iter::Int64)

    @unpack λ, tol_coef = algo
    @unpack a0, a1, b0, b1 = res

    # calculate difference between existing and new coefficients
    diff = abs(a0_new - a0) + abs(a1_new - a1) + abs(b0_new - b0) + abs(b1_new - b1)

    if diff < tol_coef   # if diff less than tol value, converged!
        converged = 1
        res.a0 = a0_new # update for outputting results
        res.a1 = a1_new
        res.b0 = b0_new
        res.b1 = b1_new

        println("-----------------------------------------------------------------------")
        @printf " Converged after %d iterations with R2 values %.3f (g) and %.3f (b)\n" iter res.R2[1] res.R2[2]
        @printf " Coefficients: a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f\n" res.a0 res.a1 res.b0 res.b1
        println("-----------------------------------------------------------------------")

    else           # if not, update coefficients using λ and continue
        converged = 0
        res.a0 = λ * a0_new + (1-λ) * a0
        res.a1 = λ * a1_new + (1-λ) * a1
        res.b0 = λ * b0_new + (1-λ) * b0
        res.b1 = λ * b1_new + (1-λ) * b1

        println("-----------------------------------------------------------------------")
        @printf " Continuing ... to iteration %d with R2 values %.3f (g) and %.3f (b)\n" iter+1 res.R2[1] res.R2[2]
        @printf "     Old coefficients: a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f\n" a0 a1 b0 b1
        @printf "     New coefficients: a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f\n" a0_new a1_new b0_new b1_new
        @printf " Updated coefficients: a0 = %.3f, a1 = %.3f, b0 = %.3f, b1 = %.3f\n" res.a0 res.a1 res.b0 res.b1
        println("-----------------------------------------------------------------------")
    end

    converged
end
