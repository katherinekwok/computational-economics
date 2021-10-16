# Author: Katherine Kwok
# Date: October 13, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model.
#
# The code below computes the STEADY STATES (from Problem Set 3), and is
# divided into the following sections:
#
#   //(0)\\ set up strucs and functions to initialize
#
#   //(1)\\ functions for solving dynamic programming problem
#
#   NOTE: This code in this portion is now modified from PS3 to allow for different
#         versions of backward iteration, depending on whether we are in steady
#         state or on the transition path.
#
#   //(2)\\ functions for solving for stationary distribution
#
#   //(3)\\ functions for solving for equilibrium aggregate K, L

# ---------------------------------------------------------------------------- #
#  (0) Set up strucs and functions to initialize
# ---------------------------------------------------------------------------- #

# Primitives: The primitives struc holds the model primitives. These numbers
#             are changed to test counterfactuals in part (3)
@everywhere mutable struct Primitives
    N::Int64                         # agent life span
    n::Float64                       # population growth rate
    age_retire::Int64                # retirement age
    μ::Array{Float64, 1}             # relative sizes of cohort (vector)
    μ_r::Float64                     # mass of retirees
    μ_w::Float64                     # mass of workers
    β::Float64                       # discount rate
    σ::Float64                       # coefficient of relative risk aversion
    θ::Float64                       # proportional labor income tax to finance benefits
    γ::Float64                       # weight on consumption
    α::Float64                       # capital share
    δ::Float64                       # capital depreciation rate
    w::Float64                       # initial wage
    r::Float64                       # initial interest rate
    b::Float64                       # social security payment to retirees
    na::Int64                        # number of asset grid points
    a_grid::Array{Float64,1}         # asset grid (vector)
    nz::Int64                        # number of productivity states
    z::Array{Float64, 1}             # productivity state high and low (vector)
    z_matrix::Array{Float64, 2}      # transition matrix for productivity state (2x2 matrix)
    z_initial_prob::Array{Float64}   # initial probabilies for z high and low (age = 1)
    e::Array{Float64, 2}             # worker productivities (cross product of z and η_j)
    K_0::Float64                     # aggregate K value
    L_0::Float64                     # aggregate L value
end

# Results: This results struc holds the results from solving the household
#          dynamic programming problem and stationary distribution.
@everywhere mutable struct Results
    # all the results are 2D because one row for each asset level today, and
    # one column for each age and productivity state

    val_func::Array{Float64, 2}  # value function
    pol_func::Array{Float64, 2}  # policy function
    lab_func::Array{Float64, 2}  # labor function
    ψ::Array{Float64, 2}         # stationary wealth distribution
end

# initialize_mu: This function initializes the μ vector, i.e. relative size of each
#                cohort from 1 to 66.
function initialize_mu(N::Int64, n::Float64)
    μ = ones(N)    # initialize μ vector
    μ_r = 0.0      # initialize mass of retirees
    μ_w = 0.0      # initialize mass of workers
    μ[1] = 1       # initialize μ_1, μ_1 > 0

    for i in 2:N   # iterate over N-1 to get all μ values
        μ[i] = μ[i-1]/(1+n)
    end
    μ = μ ./sum(μ) # normalize so μ sums to 1
    μ # return μ
end

# initialize_e: This function initializes the worker productivities, which is
#               the cross product of z (idiosyncratic productivity) and η_j
#               (deterministic productivity by age). η_j is from the input file
#               "ef.txt"
function initialize_e(z::Array{Float64, 1}, η_input_file::String)
    η_file = open(η_input_file)                           # open input file and read data
    η_data = parse.(Float64, strip.(readlines(η_file)))   # strip white spaces and convert to float

    e_h = z[1] .* η_data  # worker productivity with z_h
    e_l = z[2] .* η_data  # worker productivity with z_l

    [e_h e_l] # return combined worker productivity for each age and z state
end

# initialize_prims: This function initializes the values for the Primitives struc.
#                   The input paramters are set to default values, and can be
#                   changed to test counterfactuals.
function initialize_prims(;z_h::Float64=3.0, z_l::Float64=0.5, θ_input::Float64=0.11,
    γ_input::Float64=0.42, w_input::Float64=1.05, r_input::Float64=0.05,
    b_input::Float64 = 0.2, K_input = 3.3, L_input = 0.2,
    η_input_file = "code/ef.txt")

    # set model primitive values
    N = 66           # agent life span
    n = 0.011        # population growth rate
    age_retire = 46  # retirement age
    μ = initialize_mu(N, n)         # relative sizes of cohort (μ sums to 1)
    μ_r = sum(μ[age_retire:N])      # mass of retirees
    μ_w = sum(μ[1:age_retire - 1])  # mass of workers
    β = 0.97         # discount rate
    σ = 2.0          # coefficient of relative risk aversion
    θ = θ_input      # proportional labor income tax to finance benefits
    γ = γ_input      # weight on consumption
    α = 0.36         # capital share
    δ = 0.06         # capital depreciation rate
    w = w_input      # initial wage
    r = r_input      # initial interest rate
    b = b_input      # social security payment to retirees
    a_min = 0.0      # asset lower bound
    a_max = 75.0     # asset upper bound
    na = 3750        # number of asset grid points
    a_grid = collect(range(a_min, length = na, stop = a_max)) # asset grid

    nz = 2                                   # number of productivity states
    z = [z_h, z_l]                           # productivity state high and low
    z_matrix= [0.9261 0.0739; 0.0189 0.9811] # transition matrix for productivity state
    z_initial_prob = [0.2037 0.7963]         # initial probabilies for z high and low (age = 1)
    e = initialize_e(z, η_input_file)        # worker productivities (cross product of z and η_j)

    K_0 = K_input                            # initialized to default/benchmark = 3.3
    L_0 = L_input                            # initialized to default/benchmark = 0.2

    # initialize primitives
    prim = Primitives(N, n, age_retire, μ, μ_r, μ_w, β, σ, θ, γ, α, δ, w, r, b,
    na, a_grid, nz, z, z_matrix, z_initial_prob, e, K_0, L_0)
    prim # return initialized primitives
end

function initialize_results(prim::Primitives)
    @unpack na, N, age_retire, nz = prim # unpack primitives

    # number of columns for results arrays, retirees have one column at each age
    # and workers have one column for each productivity state and age
    # columns are ordered so that the first 45 * 2 = 90 columns are for workers
    # then 66 - 46 + 1 = 21 columns are for retirees
    col_num_all = (N - age_retire + 1) + (age_retire - 1)*nz

    # number of columns for labor function array; only includes workers - one column
    # for each productivity and age from 1 - 45
    col_num_worker = (age_retire - 1)*nz

    val_func = zeros(na, col_num_all)          # initial value function guess
    pol_func = zeros(na, col_num_all)          # initial policy function guess
    lab_func = zeros(na, col_num_worker)       # initial labor function
    ψ = zeros(na * nz, N)                      # initial cross-sec distribution

    res = Results(val_func, pol_func, lab_func, ψ)  # initialize results struct
    res # return initialized results
end

function worker_val_index(z_index::Int64, age::Int64, nz::Int64)
    if z_index == 1                                # get val_func index based on age and state
        val_index = (age * nz) - 1                 # high productivity
    else
        val_index = age * nz                       # low productivity
    end
    val_index
end

function retiree_val_index(age_retire::Int64, nz::Int64, age::Int64)
    (age_retire - 1)*nz + age - age_retire + 1     # get index in val func, pol func
end

# ---------------------------------------------------------------------------- #
#  (1) Functions for solving dynamic programming problem
# ---------------------------------------------------------------------------- #

# utility_retiree: this function encodes the utility function of the retiree
function utility_retiree(c::Float64, σ::Float64, γ::Float64)
    if c > 0
        u_r = (c^((1-σ)*γ))/(1-σ) # CRRA utility for retiree
    else
        u_r = -Inf
    end
    u_r # return calculation
end

# bellman_retiree: this function encodes the Bellman function for the retired
#
# NOTE: This function is different from the problem set 3 function, because it
# allows us to solve the transition path value function iteration. The default
# is set to the steady state version.
#
function bellman_retiree(prim::Primitives, res::Results, age::Int64; steady_state::Bool = true, res_next::Results = res)
    @unpack N, na, a_grid, nz, r, b, σ, γ, age_retire, β = prim
    @unpack val_func = res

    val_index = retiree_val_index(age_retire, nz, age)                  # mapping to index in val func

    if age == N # if age == N, initialize last year of life value function
        for (a_index, a_today) in enumerate(a_grid)
            c = (1 + r) * a_today + b                                   # consumption in last year of life (a' = 0)
            res.val_func[a_index, val_index] = utility_retiree(c, σ, γ) # value function for retiree given utility (v_next = 0 for last age period of life)
        end
    else # if not at end of life, compute value funaction for normal retiree
        choice_lower = 1                                                # for exploiting monotonicity of policy function

        for (a_index, a_today) in enumerate(a_grid)                     # loop through asset levels today
            max_val = -Inf                                              # initialize max val

            @sync @distributed for ap_index in choice_lower:na          # loop through asset levels tomorrow
                a_tomorrow = a_grid[ap_index]                           # get a tomorrow

                if steady_state == true                                    # NOTE: if in steady state
                    v_next_val = res.val_func[ap_index, val_index+1]       # get next age period val func given a'
                else                                                       # NOTE: if on transition path
                    v_next_val = res_next.val_func[ap_index, val_index+1]  # get next age and time period val func given a'
                end

                c = (1 + r) * a_today + b - a_tomorrow                  # consumption for retiree

                if c > 0                                                    # check for positivity of c
                    v_today = utility_retiree(c, σ, γ) + β * v_next_val     # value function for retiree

                    if max_val < v_today  # check if we have bigger value for this a_tomorrow
                        max_val = v_today                                   # update max value
                        res.pol_func[a_index, val_index] = a_tomorrow       # update asset policy
                        choice_lower = ap_index
                    end
                end
            end # end of a_tomorrow loop for standard retiree
            res.val_func[a_index, val_index] = max_val # update val function for a_today and age
        end # end of a_today loop for standard retiree
    end # end of if statement checking for whether at end of life or not
end

# utility_worker: this function encodes the utility function of the worker
function utility_worker(c::Float64, l::Float64, σ::Float64, γ::Float64)
    if c >0
        u_w = (((c^γ)*(1-l)^(1-γ))^(1-σ))/(1-σ) # CRRA utility for worker
    else
        u_w = -Inf
    end
    u_w # return calculation
end

# labor_supply: this function encodes the labor supply function
function labor_supply(γ::Float64, θ::Float64, e_today::Float64, w::Float64,
    r::Float64, a_today::Float64, a_tomorrow::Float64)

    top = γ*(1-θ)*e_today*w - (1-γ)*((1+r)*a_today - a_tomorrow)   # numerator of labor supply function
    bottom = (1-θ)*w*e_today                                       # denominator of labor supply function
    labor = top/bottom                                             # get labor supply (interior solution)

    labor = min(1, max(0, labor))                                  # check if we do not get interior solution
    labor
end

# bellman_worker: this function encodes the Bellman function for workers
#
# NOTE: This function is different from the problem set 3 function, because it
# allows us to solve the transition path value function iteration. The default
# is set to the steady state version.
#
function bellman_worker(prim::Primitives, res::Results, age::Int64; steady_state::Bool = true, res_next::Results = res)
    @unpack N, na, a_grid, nz, r, b, σ, γ, age_retire, β, z, θ, e, z_matrix, w = prim
    @unpack val_func = res

    for (z_index, z_today) in enumerate(z)                              # loop through productivity states
        e_today = e[age, z_index]                                       # get productivity for age and z_today
        z_prob = z_matrix[z_index, :]                                   # get transition probabilities given z_today
        val_index = worker_val_index(z_index, age, nz)                  # get index to val, pol, lab func

        choice_lower = 1                                                # for exploiting monotonicity of policy function

        for (a_index, a_today) in enumerate(a_grid)                     # loop through asset levels today
            max_val = -Inf                                              # initialize max val

            @sync @distributed for ap_index in choice_lower:na          # loop through asset levels tomorrow

                a_tomorrow = a_grid[ap_index]                           # get a tomorrow
                l = labor_supply(γ, θ, e_today, w, r, a_today, a_tomorrow)       # labor supply for worker
                c = w * (1-θ) * e_today * l + (1 + r) * a_today - a_tomorrow     # consumption for worker

                if c > 0 && l >= 0 && l <= 1                            # check for positivity of c and constraint on l: 0 <= l <= 1
                    if age == age_retire -1                                     # if age == 45 (retired next age period), then no need for transition probs

                        if steady_state == true                                    # NOTE: if in steady state
                            v_next_val = res.val_func[ap_index, age * nz + 1]      #       get next age period val func (just scalar) given a_tomorrow
                        else                                                       # NOTE: if on transition path
                            v_next_val = res_next.val_func[ap_index, age * nz + 1] #       get next age and period val func
                        end

                        v_today = utility_worker(c, l, σ, γ) + β * v_next_val   # value function for worker

                    else                                                                  # else if just normal worker, need transition probs

                        if steady_state == true                                                 # NOTE: if in steady state
                            v_next_val = res.val_func[ap_index, (age+1)*nz - 1:(age+1)*nz]      # get next age period val func (vector including high and low prod) given a_tomorrow
                        else                                                                    # NOTE: if on transition path
                            v_next_val = res_next.val_func[ap_index, (age+1)*nz - 1:(age+1)*nz]    # get next age and time period val func
                        end

                        v_today = utility_worker(c, l, σ, γ) + β * z_prob' * v_next_val   # value function for worker with transition probs
                    end

                    if max_val <= v_today  # check if we have bigger value for this a_tomorrow
                        max_val = v_today                                   # update max value
                        res.pol_func[a_index, val_index] = a_tomorrow       # update asset policy
                        res.lab_func[a_index, val_index] = l                # update labor policy
                        choice_lower = ap_index
                    end
                end
            end # end of a_tomorrow loop for worker
            res.val_func[a_index, val_index] = max_val # update val function for a_today and age
        end # end of a_today loop for worker
    end
end

# v_backward_iterate: is the value function iteration loop, which calls the Bellman
#                     function from end of life until the beginning (i.e. iterates backward)
#
# NOTE: This function is different from the problem set 3 function, because it
# allows us to solve the transition path value function iteration. The default
# is set to the steady state version.
#
function v_backward_iterate(prim::Primitives, res::Results; steady_state::Bool = true, res_next_input::Results = res)
    @unpack N, age_retire = prim

    for age in N:-1:1
        if age >= age_retire                   # if between retirement age and end of life
            if steady_state == true                 # do different backward iteration for steady state vs. transition
                bellman_retiree(prim, res, age;)    # call Bellman for retiree
            else
                bellman_retiree(prim, res, age; steady_state = false, res_next = res_next_input)
            end
        else                                   # else, agent is still worker
            if steady_state == true                 # do different backward iteration for steady state vs. transition
                bellman_worker(prim, res, age;)     # call Bellman for worker
            else
                bellman_worker(prim, res, age; steady_state = false, res_next = res_next_input)
            end
        end
    end
end

# ---------------------------------------------------------------------------- #
#  (2) Functions for solving for stationary distribution
# ---------------------------------------------------------------------------- #

# initialize_ψ: This function initiates the staionary distribution for the first
# age period of life, using the ergodic distribution (productivity drawn at birth)
function initialize_ψ(prim::Primitives, res::Results)
    @unpack na, z_initial_prob, nz, μ = prim
    res.ψ[1, 1] = prim.z_initial_prob[1]  * μ[1]      # distribution of high prod people
    res.ψ[na+1, 1] = prim.z_initial_prob[2] * μ[1]    # distribution of low prod people
end


# make_trans_matrix: function that creates the transition matrix that maps from
# the current state (z, a) to future state (z', a') for a given age j.
#
# NOTE: This is a modified version of the make_trans_matrix function in PS3, as
# we need to make the transition matrix differently depending on whether we are
# in the steady state or on the transition path.
#
function make_trans_matrix(prim::Primitives, pol_func::Array{Float64, 2}, age::Int64)
    @unpack a_grid, na, nz, z_matrix, z, age_retire = prim  # unpack model primitives
    trans_mat = zeros(nz*na, nz*na)                         # initiate transition matrix (zeros)

    for (z_index, z_today) in enumerate(z)                  # loop through current productivity states
        for (a_index, a_today) in enumerate(a_grid)         # loop through current asset grid

            row_index = a_index + na*(z_index - 1)          # create mapping from current a, z indices to big trans matrix ROW index (today's state)
            if age >= age_retire
                val_index = retiree_val_index(age_retire, nz, age)  # get asset index for retireee
            else
                val_index = worker_val_index(z_index, age, nz)      # get asset index for worker
            end
            a_choice = pol_func[a_index, val_index]               # get asset choice for given age

            for (zp_index, z_tomorrow) in enumerate(z)            # loop through future productivity states
                for (ap_index, a_tomorrow) in enumerate(a_grid)   # loop through future asset states

                    if a_choice == a_tomorrow                               # check if a_choice from policy function matches a tomorrow (this is the indicator in T*)
                        col_index = ap_index + na*(zp_index - 1)            # create mapping from current a, z indices to big trans matrix COLUMN index (tomorrow's state)
                        trans_mat[row_index, col_index] = z_matrix[z_index, zp_index] # enter Markov transition prob for employment state
                    end
                end
            end
        end
    end
    trans_mat # return the big transition matrix!
end


# solve_ψ: This function iterates from age 1 to N-1 (one age period before end of life)
# to find the stationary distribution for each age. The distribution tells us where
# a person will be in the distribution in the next age period, given their state in
# the current age period.
function solve_ψ(prim::Primitives, res::Results)
    @unpack N, μ, n = prim

    initialize_ψ(prim, res)                              # initialize ψ for first age period of life (age = 1)

    for age in 1:N-1
        trans_mat = make_trans_matrix(prim, res.pol_func, age)     # make transition matrix for given age
        res.ψ[:, age + 1] = trans_mat' * res.ψ[:, age] * (1/(1+n)) # get distribution for next age period
    end


    println("-----------------------------------------------------------------------")
    println("                Solved for stationary distributions.")
    println("-----------------------------------------------------------------------")
end

# ---------------------------------------------------------------------------- #
#  (3) functions for solving for equilibrium aggregate K, L
# ---------------------------------------------------------------------------- #

# F_1: This function calculates wage using the production function (derivative
# with respect to L)
function F_1(α::Float64, K::Float64, L::Float64)
    (1-α) * K^α * L^(-α)
end

# F_2: This function calculates interest rate using the production function (
# derivative with respect to K)
function F_2(α::Float64, δ::Float64, K::Float64, L::Float64)
   α * K^(α - 1) * L^(1-α) - δ
end

# b: This function calculates the social security benefit level, given w (wage),
#    L (aggregate labor supply), and μ_r (mass of retirees)
function calculate_b(θ::Float64, w::Float64, L::Float64 ,μ_r::Float64)
    (θ * w * L)/μ_r
end

# calc_aggregate: This function calculates the aggregate K and L given the results
# from solving for the decision rules (pol_func, lab_func) and stationary
# distribution (ψ)
function calc_aggregate(prim::Primitives, pol_func::Array{Float64, 2}, ψ::Array{Float64, 2}, lab_func::Array{Float64, 2})
    @unpack N, na, nz, a_grid, z, e, age_retire = prim

    agg_K = 0.0   # initiate aggregate values
    agg_L = 0.0

    for age in 1:N                                                              # for each age
        for (a_index, a_val) in enumerate(a_grid)                               # for each asset level
            for (z_index, z_val) in enumerate(z)                                # for each productivity state
                ψ_index = a_index + na*(z_index - 1)                            # get index for stationary distribution

                if age < age_retire                                             # if working, sum a and l decisions
                    lab_index = worker_val_index(z_index, age, nz)
                    agg_L += e[age, z_index] * lab_func[a_index, lab_index] * ψ[ψ_index, age]
                    agg_K += a_val * ψ[ψ_index, age]
                else                                                            # if retired, only sum a decision
                    agg_K += a_val * ψ[ψ_index, age]
                end
            end
        end
    end
    agg_K, agg_L # retired calculate aggregate values
end

# calc_prices: This function calculates the prices (r and w) and b
function calc_prices(prim::Primitives, K_0::Float64, L_0::Float64)
    @unpack α, δ, θ, μ_r = prim

    prim.w = F_1(α, K_0, L_0)
    prim.r = F_2(α, δ, K_0, L_0)
    prim.b = calculate_b(θ, prim.w, L_0, μ_r)
end

# calc_welfare: This function calculates the welfare of each specification
function calc_welfare(prim::Primitives, res::Results)
    @unpack na, nz, N, age_retire = prim
    @unpack val_func, ψ = res

    welfare = 0.0                   # initialize welfare
    for age in 1:N                  # loop through all ages
        if age < age_retire
            v_h = val_func[:, worker_val_index(1, age, nz)]  # get val func for high prod
            v_l = val_func[:, worker_val_index(2, age, nz)]  # get val func for low prod
            welfare += ψ[1:na, age][isfinite.(v_h)]' * v_h[isfinite.(v_h)]       # val func multiplied by stationary dist for high prod
            welfare += ψ[na+1:nz*na, age][isfinite.(v_l)]' * v_l[isfinite.(v_l)] # val func multiplied by stationary dist for low prod
        else
            v = val_func[:, retiree_val_index(age_retire, nz, age)] # follow similar process as above with retiree
            welfare += ψ[1:na, age][isfinite.(v)]' * v[isfinite.(v)]
            welfare += ψ[na+1:nz*na, age][isfinite.(v)]' * v[isfinite.(v)]
        end
    end
    welfare
end

# calc_wealth_var: This function calculates the coefficient of variation for the
# wealth distribution
function calc_wealth_var(prim::Primitives, res::Results)
    @unpack age_retire, a_grid, b, w, r, na, nz, N, e, θ, z = prim
    @unpack lab_func, ψ = res
    wealth = zeros(na * nz, N) # initialize wealth grid

    for age in 1:N                                      # loop through age, asset, z state
        for (a_index, a_val) in enumerate(a_grid)
            for (z_index, z_val) in enumerate(z)
                w_index = a_index + na*(z_index - 1)    # get row index in wealth matrix
                if age >= age_retire                    # wealth for retiree: asset plus benefits
                    wealth[w_index, age] = a_val * (1+r) + b
                else
                    e_t = e[age, z_index]                           # get e (productivity) given age and z
                    val_index = worker_val_index(z_index, age, nz)  # get index for lab function
                    l_t = lab_func[a_index, val_index]              # get labor supply
                    income = w * (1-θ) * e_t * l_t                  # income
                    wealth[w_index, age] = a_val * (1+r) + income   # wealth for worker: asset plus income
                end
            end
        end
    end

    mean_wealth = 0.0                               # initialize mean wealth value
    for age in 1:N                                  # loop through ages
        mean_wealth += ψ[:, age]' * wealth[:, age]  # get dot product of mass and wealth by age
    end
    var_wealth = 0.0                                                     # initialize variance value
    for age in 1:N                                                       # loop through ages
        var_wealth += ψ[:, age]' * ((wealth[:, age] .- mean_wealth).^2)  # get dot product of mass and squared error by age
    end
    std_wealth = sqrt(var_wealth) # standard deviation
    std_wealth/mean_wealth        # return coefficient of variation
end

# check_market_clearing: This function calls a helper function to calculate the
# aggregate K and L, checks if it falls within the tolerance value, and and if
# not, updates the K and L values.
function check_market_clearing(prim::Primitives, res::Results, n::Int64, λ::Float64, tol::Float64)

    K_1, L_1 = calc_aggregate(prim, res.pol_func, res.ψ, res.lab_func) # calculate aggregate K_1, L_1 after VFT and solving for ψ
    abs_diff = abs(K_1 - prim.K_0) + abs(L_1 - prim.L_0)               # calculate abs difference

    if abs_diff < tol                                 # if abs diff < tolerance val, we've converged
        converged = 1                                 # update convergence flag
        prim.K_0 = K_1                                # update with final K and L
        prim.L_0 = L_1

    else
        prim.K_0 = λ * K_1 + (1-λ) * prim.K_0         # else, we update K_0 and L_0
        prim.L_0 = λ * L_1 + (1-λ) * prim.L_0

        calc_prices(prim, prim.K_0, prim.L_0)         # update prices (r, w) and b

        println("-----------------------------------------------------------------------")
        @printf " At n=%d, new K_0 = %.5f and L_0 = %.5f, K_1 = %.5f and L_1 = %.5f\n" n prim.K_0 prim.L_0 K_1 L_1
        println("-----------------------------------------------------------------------")

        converged = 0
    end
    converged  # return convergence flag
end

# summarize_results: This function summarizes the results from solving the model
function summarize_results(prim::Primitives, res::Results, n::Int64; λ::Float64 = 0.99)
    @unpack K_0, L_0 = prim

    calc_prices(prim, K_0, L_0)             # calculate prices
    welfare = calc_welfare(prim, res)       # calculate welfare
    cv_wealth = calc_wealth_var(prim, res)  # calculate coefficient of variation for wealth

    println("-----------------------------------------------------------------------")
    @printf " Steady State Summary\n"
    println("-----------------------------------------------------------------------")
    @printf " K          = %.5f\n" K_0
    @printf " L          = %.5f\n" L_0
    @printf " w          = %.5f\n" prim.w
    @printf " r          = %.5f\n" prim.r
    @printf " b          = %.5f\n" prim.b
    @printf " welfare    = %.5f\n" welfare
    @printf " cv(wealth) = %.5f\n" cv_wealth
    @printf " iter       = %d\n" n
    println("-----------------------------------------------------------------------")

    welfare, cv_wealth
end


# solve_model: This function is a wrapper that calls each step of the algorithm
# to solve the Conesa-Krueger model.
function solve_model(;K_0::Float64 = 3.3, L_0::Float64 = 0.3, θ_0::Float64 = 0.11,
    z_h_0::Float64 = 3.0, γ_0::Float64 = 0.42, λ::Float64 = 0.99, tol::Float64 = 1.0e-3)

    converged = 0                                           # convergence flag
    n = 0                                                   # counter for iterations
    prim = initialize_prims(K_input = K_0, L_input = L_0,
                θ_input = θ_0, z_h = z_h_0, γ_input = γ_0)  # initialize benchmark prims
    res = initialize_results(prim)                          # initialize results structs
    calc_prices(prim, K_0, L_0)                             # calculate prices (r, w) and b

    while converged == 0                                         # until we reach convergence
        n+= 1                                                    # update iteration counter
        v_backward_iterate(prim, res)                            # backward iterate to get policy functions for each age
        solve_ψ(prim, res)                                       # shoot forward to get stationary distribution
        converged = check_market_clearing(prim, res, n, λ, tol)  # check market clearing condition for convergence
    end

    welf, cv_wealth = summarize_results(prim, res, n)  # print results
    prim, res, welf, cv_wealth                         # return results and prims for debugging/checking
end
