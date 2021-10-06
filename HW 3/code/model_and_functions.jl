# Author: Katherine Kwok
# Date: Sept 28, 2021

# This file contains the code for Problem Set 3, where we want to want to evaluate
# the value of the Social Security program. The program is broken down into the
# following steps:
#
#      (1) Solve the dynamic programming problem for individuals at different ages
#          (where if age >= 46, agent is retired; working if not)
#      (2) Solve for the steady-state distribution of agents over age, productivity
#          and asset holdings.
#      (3) Test counterfactuals


# ---------------------------------------------------------------------------- #
#  (0) Set up strucs and functions to initialize
# ---------------------------------------------------------------------------- #

# Primitives: The primitives struc holds the model primitives. These numbers
#             are changed to test counterfactuals in part (3)
mutable struct Primitives
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
mutable struct Results
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
    na = 5000        # number of asset grid points
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
function bellman_retiree(prim::Primitives, res::Results, age::Int64)
    @unpack N, na, a_grid, nz, r, b, σ, γ, age_retire, β = prim
    @unpack val_func = res

    val_index = retiree_val_index(age_retire, nz, age)                  # mapping to index in val func

    if age == N # if age == N, initialize last year of life value function
        for (a_index, a_today) in enumerate(a_grid)
            c = (1 + r) * a_today + b                                   # consumption in last year of life (a' = 0)
            res.val_func[a_index, val_index] = utility_retiree(c, σ, γ) # value function for retiree given utility (v_next = 0 for last period of life)
        end
    else # if not at end of life, compute value funaction for normal retiree
        choice_lower = 1                                                # for exploiting monotonicity of policy function

        for (a_index, a_today) in enumerate(a_grid)                     # loop through asset levels today
            max_val = -Inf                                              # initialize max val

            for ap_index in choice_lower:na                             # loop through asset levels tomorrow
                a_tomorrow = a_grid[ap_index]                           # get a tomorrow
                v_next_val = res.val_func[ap_index, val_index+1]        # get next period val func given a'
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
function bellman_worker(prim::Primitives, res::Results, age::Int64)
    @unpack N, na, a_grid, nz, r, b, σ, γ, age_retire, β, z, θ, e, z_matrix, w = prim
    @unpack val_func = res

    for (z_index, z_today) in enumerate(z)                              # loop through productivity states
        e_today = e[age, z_index]                                       # get productivity for age and z_today
        z_prob = z_matrix[z_index, :]                                   # get transition probabilities given z_today
        val_index = worker_val_index(z_index, age, nz)                  # get index to val, pol, lab func

        choice_lower = 1                                                # for exploiting monotonicity of policy function

        for (a_index, a_today) in enumerate(a_grid)                     # loop through asset levels today
            max_val = -Inf                                              # initialize max val

            for ap_index in choice_lower:na                             # loop through asset levels tomorrow

                a_tomorrow = a_grid[ap_index]                           # get a tomorrow
                l = labor_supply(γ, θ, e_today, w, r, a_today, a_tomorrow)       # labor supply for worker
                c = w * (1-θ) * e_today * l + (1 + r) * a_today - a_tomorrow     # consumption for worker

                if c > 0 && l >= 0 && l <= 1                            # check for positivity of c and constraint on l: 0 <= l <= 1
                    if age == age_retire -1                                     # if age == 45 (retired next period), then no need for transition probs
                        v_next_val = res.val_func[ap_index, age * nz + 1]       # get next period val func (just scalar) given a_tomorrow
                        v_today = utility_worker(c, l, σ, γ) + β * v_next_val   # value function for worker

                    else                                                                  # else, need transition probs
                        v_next_val = res.val_func[ap_index, (age+1)*nz - 1:(age+1)*nz]    # get next period val func (vector including high and low prod) given a_tomorrow
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
# function from end of life until the beginning (i.e. iterates backward)
function v_backward_iterate(prim::Primitives, res::Results)
    @unpack N, age_retire = prim

    for age in N:-1:1
        if age >= age_retire                   # if between retirement age and end of life
            bellman_retiree(prim, res, age)    # call Bellman for retiree
        else                                   # else, agent is still worker
            bellman_worker(prim, res, age)     # call Bellman for worker
        end
    end
    println("-----------------------------------------------------------------------")
    println("              Value function interation is complete.")
    println("-----------------------------------------------------------------------")
end


# plot_ex_1: This function plots all the necessary graphs for exercise 1
function plot_ex_1(prim::Primitives, res::Results)
    @unpack val_func, pol_func, lab_func = res
    @unpack a_grid, nz, age_retire = prim

    # plot value function for age 50
    index_age_50 = retiree_val_index(age_retire, nz, 50)           # mapping to index in val func
    Plots.plot(a_grid, val_func[:, index_age_50], label = "", title = "Value Function for Retired Agent at 50")
    Plots.savefig("output/age_50_value_func.png")

    # plot policy function for age 20
    index_age_20_h = worker_val_index(1, 20, nz)
    index_age_20_l = worker_val_index(2, 20, nz)
    Plots.plot(a_grid, pol_func[:, index_age_20_h] .- a_grid, label = "High productivity")
    Plots.plot!(a_grid, pol_func[:, index_age_20_l] .- a_grid, label = "Low productivity",
    title = "Savings Decisions for Worker at Age 20", legend = :topright, xlims = [0, 75], ylims = [-0.2, 1.5])
    Plots.savefig("output/age_20_savings.png")

    # plot labor supply for age 20
    Plots.plot(a_grid, lab_func[:, index_age_20_h], label = "High productivity")
    Plots.plot!(a_grid, lab_func[:, index_age_20_l], label = "Low productivity",
    title = "Labor Supply for Worker at Age 20", legend = :topright)
    Plots.savefig("output/age_20_labor.png")

end

# ---------------------------------------------------------------------------- #
#  (2) Functions for solving for stationary distribution
# ---------------------------------------------------------------------------- #

# initialize_ψ: This function initiates the staionary distribution for the first
# period of life, using the ergodic distribution (productivity drawn at birth)
function initialize_ψ(prim::Primitives, res::Results)
    @unpack na, z_initial_prob, nz, μ = prim
    res.ψ[1, 1] = prim.z_initial_prob[1]  * μ[1]      # distribution of high prod people
    res.ψ[na+1, 1] = prim.z_initial_prob[2] * μ[1]    # distribution of low prod people
end


# make_trans_matrix: function that creates the transition matrix that maps from
# the current state (z, a) to future state (z', a') for a given age j.
function make_trans_matrix(prim::Primitives, res::Results, age::Int64)
    @unpack a_grid, na, nz, z_matrix, z, age_retire = prim  # unpack model primitives
    @unpack pol_func = res                                  # unpack policy function from value function iteration
    trans_mat = zeros(nz*na, nz*na)                         # initiate transition matrix

    for (z_index, z_today) in enumerate(z)                  # loop through current productivity states
        for (a_index, a_today) in enumerate(a_grid)         # loop through current asset grid

            row_index = a_index + na*(z_index - 1)          # create mapping from current a, z indices to big trans matrix ROW index (today's state)
            if age >= age_retire
                val_index = retiree_val_index(age_retire, nz, age)    # get asset index for retireee
            else
                val_index = worker_val_index(z_index, age, nz)        # get asset index for worker
            end
            a_choice = pol_func[a_index, val_index]                   # get asset choice

            for (zp_index, z_tomorrow) in enumerate(z)                      # loop through future productivity states
                for (ap_index, a_tomorrow) in enumerate(a_grid)             # loop through future asset states

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


# solve_ψ: This function iterates from age 1 to N-1 (one period before end of life)
# to find the stationary distribution for each age. The distribution tells us where
# a person will be in the distribution in the next period, given their state in
# the current period.
function solve_ψ(prim::Primitives, res::Results)
    @unpack N, μ, n = prim

    initialize_ψ(prim, res)                              # initialize ψ for first period of life (age = 1)

    for age in 1:N-1
        trans_mat = make_trans_matrix(prim, res, age)    # make transition matrix for given age
        res.ψ[:, age + 1] = trans_mat' * res.ψ[:, age] * (1/(1+n))   # get distribution for next period
    end


    println("-----------------------------------------------------------------------")
    println("                Solved for stationary distributions.")
    println("-----------------------------------------------------------------------")
end

# ---------------------------------------------------------------------------- #
#  (3) Functions to test benchmark and counterfactuals
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
function calc_aggregate(prim::Primitives, res::Results)
    @unpack N, na, nz, a_grid, z, e, age_retire = prim
    @unpack pol_func, ψ, lab_func = res

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

# check_market_clearing: This function calls a helper function to calculate the
# aggregate K and L, checks if it falls within the tolerance value, and and if
# not, updates the K and L values.
function check_market_clearing(prim::Primitives, res::Results, n::Int64; λ::Float64 = 0.99, tol::Float64 = 1.0e-3)
    @unpack α, δ, θ, μ_r = prim

    K_1, L_1 = calc_aggregate(prim, res)                 # calculate aggregate K_1, L_1 after VFT and solving for ψ
    abs_diff = abs(K_1 - prim.K_0) + abs(L_1 - prim.L_0) # calculate abs difference

    if abs_diff < tol                                 # if abs diff < tolerance val, we've converged
        converged = 1
    else
        prim.K_0 = λ * K_1 + (1-λ) * prim.K_0         # else, we update K_0 and L_0
        prim.L_0 = λ * L_1 + (1-λ) * prim.L_0

        prim.w = F_1(α, prim.K_0, prim.L_0)           # update prices (r, w) and b
        prim.r = F_2(α, δ, prim.K_0, prim.L_0)
        prim.b = calculate_b(θ, prim.w, prim.L_0, μ_r)

        converged = 0
    end


    println("-----------------------------------------------------------------------")
    @printf " At n=%d, new K_0 = %.5f and L_0 = %.5f, K_1 = %.5f and L_1 = %.5f\n" n prim.K_0 prim.L_0 K_1 L_1
    println("-----------------------------------------------------------------------")

    converged  # return convergence flag

end


# solve_model: This function is a wrapper that calls each step of the algorithm
# to solve the Conesa-Krueger model.
function solve_model(K_0::Float64 = 3.3, L_0::Float64 = 0.3, θ_0::Float64 = 0.11,
    z_h_0::Float64 = 3.0, z_l_0::Float64 = 0.5, γ_0::Float42 = 0.42)

    converged = 0    # convergence flag
    n = 0            # counter for iterations

    prim_bm = initialize_prims(K_input = K_0, L_input = L_0, θ_input = θ_0, z_h = z_h_0, z_l = z_l_0, γ_input = γ_0)  # initialize benchmark prims
    res_bm = initialize_results(prim_bm)

    prim_bm.w = F_1(prim_bm.α, K_0, L_0)                            # calculate wage
    prim_bm.r = F_2(prim_bm.α, prim_bm.δ, K_0, L_0)                 # calculate interest rate
    prim_bm.b = calculate_b(prim_bm.θ, prim_bm.w, L_0, prim_bm.μ_r) # calculate benefit amount

    while converged == 0                                          # until we reach convergence
        n+= 1                                                     # update iteration counter
        v_backward_iterate(prim_bm, res_bm)                       # backward iterate to get policy functions for each age
        solve_ψ(prim_bm, res_bm)                                  # shoot forward to get stationary distribution
        converged = check_market_clearing(prim_bm, res_bm, n)     # check market clearing condition for convergence
    end
end
