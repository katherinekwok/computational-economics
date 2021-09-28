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
    μ[1] = 1       # initialize μ_1, μ_1 > 0

    for i in 2:N   # iterate over N to get all μ values
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
    γ_input::Float64=0.42, w_initial::Float64=1.05, r_initial::Float64=0.05,
    η_input_file = "code/ef.txt")

    # set model primitive values
    N = 66           # agent life span
    n = 0.011        # population growth rate
    age_retire = 46  # retirement age
    μ = initialize_mu(N, n) # relative sizes of cohort (μ sums to 1)
    β = 0.97         # discount rate
    σ = 2            # coefficient of relative risk aversion
    θ = θ_input      # proportional labor income tax to finance benefits
    γ = γ_input      # weight on consumption
    α = 0.36         # capital share
    δ = 0.06         # capital depreciation rate
    w = w_initial    # initial wage
    r = r_initial    # initial interest rate
    b = 0.2          # social security payment to retirees
    a_min = 0.0      # asset lower bound
    a_max = 5.0      # asset upper bound
    na = 1000        # number of asset grid points
    a_grid = collect(range(a_min, length = na, stop = a_max)) # asset grid

    nz = 2                                   # number of productivity states
    z = [z_h, z_l]                           # productivity state high and low
    z_matrix= [0.9261 0.0739; 0.9811 0.0189] # transition matrix for productivity state
    z_initial_prob = [0.2037 0.7963]         # initial probabilies for z high and low (age = 1)
    e = initialize_e(z, η_input_file)        # worker productivities (cross product of z and η_j)

    # initialize primitives
    prim = Primitives(N, n, age_retire, μ, β, σ, θ, γ, α, δ, w, r, b, na, a_grid, nz, z, z_matrix, z_initial_prob, e)
    prim # return initialized primitives
end

function initialize_results(prim::Primitives)
    @unpack na, N, age_retire, nz = prim # unpack primitives

    # number of columns for results arrays, retirees have one column at each age
    # and workers have one column for each productivity state and age
    col_num_all = (N - age_retire) + (age_retire - 1)*nz

    # number of columns for labor function array; only includes workers - one column
    # for each productivity and age from 1 - 45
    col_num_worker = (age_retire - 1)*nz

    val_func = zeros(na, col_num_all)          # initial value function guess
    pol_func = zeros(na, col_num_all)          # initial policy function guess
    lab_func = zeros(na, col_num_worker)       # initial labor function
    ψ = ones(na * nz, N)                       # initial cross-sec distribution

    res = Results(val_func, pol_func, lab_func, ψ)  # initialize results struct
    res # return initialized results
end

# ---------------------------------------------------------------------------- #
#  (1) Functions for solving dynamic programming problem
# ---------------------------------------------------------------------------- #

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
