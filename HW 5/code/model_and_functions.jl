# Author: Katherine Kwok (Using some reference code provided by Phil Coyle!)
# Date: October 22, 2021

# This file contains the code for Problem Set 5, where we solve the Krusell-Smith
# model.
#
# The code below is divided into the following sections:
#
#   (0) set up strucs and functions to initialize
#   (1) functions for value function iteration
#   (2) functions for simulating capital path
#   (3) functions for estimating regression and checking for convergence

# ------------------------------------------------------------------------ #
#   (0) set up strucs and functions to initialize
# ------------------------------------------------------------------------ #

@with_kw struct Parameters
    β::Float64 = 0.99               # discount factor
    α::Float64 = 0.36               # capital share
    δ::Float64 = 0.025              # capital depreciation rate
    λ::Float64 = 0.5

    N::Int64 = 5000
    T::Int64 = 11000
    burn::Int64 = 1000

    tol_vfi::Float64 = 1e-4
    tol_coef::Float64 = 1e-4
    tol_r2::Float64 = 1.0 - 1e-2
    maxit::Int64 = 10000
end

@with_kw struct Grids
    k_lb::Float64 = 0.001
    k_ub::Float64 = 20.0
    n_k::Int64 = 21
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0
    K_ub::Float64 = 15.0
    n_K::Int64 = 11
    K_grid::Array{Float64,1} = range(K_lb, stop = K_ub, length = n_K)

    eps_l::Float64 = 0.0
    eps_h::Float64 = 0.3271
    n_eps::Int64 = 2
    eps_grid::Array{Float64,1} = [eps_h, eps_l]

    z_l::Float64 = 0.99
    z_h::Float64 = 1.01
    n_z::Int64 = 2
    z_grid::Array{Float64,1} = [z_h, z_l]
end

@with_kw struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0 # Duration (Good Times)
    u_b::Float64 = 0.1 # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0 # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pgb::Float64 = 1.0 - (d_b-1.0)/d_b
    pbg::Float64 = 1.0 - (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb10::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    markov::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                                pbg*Mbg pbb*Mbb]
end

mutable struct Results
    pf_k::Array{Float64,4}
    pf_v::Array{Float64,4}

    a0::Float64
    a1::Float64
    b0::Float64
    b1::Float64

    R2::Array{Float64,1}
end
