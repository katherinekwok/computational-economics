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
#  (0) Set up primitives
# ---------------------------------------------------------------------------- #

# Primitives: The primitives struc holds the model primitives. These numbers
#             are changed to test counterfactuals in part (3)
mutable struct Primitives
    N::Int64                # agent life span
    n::Float64              # population growth rate
    age_retire::Int64       # retirement age

    β::Float64              # discount rate
    σ::Float64              # coefficient of relative risk aversion
    θ::Float64              # proportional labor income tax to finance benefits
    γ::Float64              # weight on consumption
    α::Float64              # capital share
    δ::Float64              # capital depreciation rate

    w::Float64              # initial wage
    r::Float64              # initial interest rate
    b::Float64              # social security payment to retirees

    na::Int64                 # number of asset grid points
    a_grid::Array{Float64,1}  # asset grid

    nz::Int64                          # number of productivity states
    z::Array{Float64, 1}               # productivity state high and low
    z_matrix::Array{Float64, 2}        # transition matrix for productivity state
    z_initial_prob::Array{Float64}     # initial probabilies for z high and low (age = 1)
end

# initialize_prims: This function initializes the values for the Primitives struc.
#                   The input paramters are set to default values, and can be
#                   changed to test counterfactuals.
function initialize_prims(z_h::Float64=3.0, z_l::Float64=0.5, θ_input::Float64=0.11,
    γ_input::Float64=0.42, w_initial::Float64=1.05, r_initial::Float64=0.05)
    N = 66           # agent life span
    n = 0.011        # population growth rate
    age_retire = 46  # retirement age

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

    # initialize primitives
    prim = Primitives(N, n, age_retire, β, σ, θ, γ, α, δ, w, r, b, na, a_grid, nz, z, z_matrix, z_initial_prob)
    prim # return initialized primitives
end
