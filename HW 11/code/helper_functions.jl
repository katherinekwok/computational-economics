# Author: Katherine Kwok
# Date: December 15, 2021

# This file contains the helper funtions and structures for Problem Set 4
# (for JF's portion) a.k.a. Problem Set 11 overall. The main program implements
# a dynamic model of inventory control.

# ---------------------------------------------------------------------------- #
# (0) Structs and definitions
# ---------------------------------------------------------------------------- #

# Primitives: This struct stores the paramter values, transition matrices, initial
#             guesses, indices, and file names.
@with_kw struct Primitives

    α::Float64 = 2          # coefficient on consumption shock
    β::Float64 = 0.99       # discount factor
    λ::Float64 = -4         # stockout penalty (when consumption shock > 0 but 0 inventory)
    i_b::Float64 = 8        # average inventory
    p_s::Float64 = 1        # sales price
    p_r::Float64 = 4        # regular price
    c::Int64 = 1            # consumption shock (0, 1)

    trans = [0.9 0.1; 0.9 0.1]  # transition matrix for prices (regular vs. sales price)

    ind_I::Int64 = 1
    ind_C::Int64 = 2
    ind_P::Int64 = 3
    ind_λ::Int64 = 1

    sim_data_file = "PS4_simdata.csv" # define fill paths
    state_space_file = "PS4_state_space.csv"
    trans_a0_file = "PS4_transition_a0.csv"
    trans_a1_file = "PS4_transition_a1.csv"

end

# DataSets: This struct stores the data sets used for the program.
mutable struct DataSets

    sim_data::Array{Any}
    S::Array{Any}
    F_0::Array{Any}
    F_1::Array{Any}
end

# process_data: This function reads and prepares the data (as arrays). It
#               returns all the data sets stored into one struct.
function process_data(data_path, prim)

    sim_data = Array(DataFrame(load(data_path*prim.sim_data_file)))[:, 2:end]
    S = Array(DataFrame(load(data_path*prim.state_space_file)))[:, 3:end]
    F_0 = Array(DataFrame(load(data_path*prim.trans_a0_file)))[:, 3:end]
    F_1 = Array(DataFrame(load(data_path*prim.trans_a1_file)))[:, 3:end]

    dataset = DataSets(sim_data, S, F_0, F_1)
    return dataset

end



# ---------------------------------------------------------------------------- #
# (1) Solve the expected value function using implicit equation
# ---------------------------------------------------------------------------- #

# payoffs: This function computes the per period pay off for a given state
function payoffs(prim, dataset)
    @unpack α, λ, ind_I, ind_C, ind_P = prim
    @unpack S = dataset

    # payoff/utility for a = 1
    U_1 = α * S[:, ind_C] .- S[:, ind_P]

    # payoff/utility for a = 0 (adding up cases if i > 0 and i = 0)
    U_0 = α * S[:, ind_C] .* (S[:, ind_I] .> 0)
    U_0 = U_0 .+ λ * (S[:, ind_C] .> 0) .* (S[:, ind_I] .== 0)

    U_0, U_1
end

# value_func: This function defines the period value for a given state
function value_func(prim, dataset, V_bar_old)
    @unpack β = prim
    @unpack F_0, F_1 = dataset

    # get payoffs
    U_0, U_1 = payoffs(prim, dataset)
    # update for V_bar function (expected value)
    V_update = hcat(U_0 .+ β * F_0 * V_bar_old, U_1 .+ β * F_1 * V_bar_old)

    return V_update
end



# e_max: This function defines the implicit equation for the expected value function
function e_max(prim, dataset, V_bar_old)

    # update using given V_bar_old
    V_update = value_func(prim, dataset, V_bar_old)
    # calculate new V_bar
    V_bar_new = log.(sum(exp.(V_update), dims = 2)) .+ Base.MathConstants.γ

    return V_bar_new
end

# val_func_iter: This function implements value function iteration to find the
#                fixed point for the e_max i.e. expected value function
function val_func_iter(prim, dataset; tol = 1e-9, max_iter = 1000)
    @unpack S = dataset

    V_bar_old = e_max(prim, dataset, zeros(size(S, 1))) # initialize guess
    err = 100
    iter = 1

    while err > tol && iter < max_iter
        V_bar_new = e_max(prim, dataset, V_bar_old)
        err = maximum(abs.(V_bar_new .- V_bar_old))
        V_bar_old = V_bar_new
        iter += 1
    end

    V_bar_old

end

# ---------------------------------------------------------------------------- #
# (2) Solve for expected value function using CCP mapping
# ---------------------------------------------------------------------------- #

# ccp_mapping: This function implements the conditional choice-probability mapping
#              for policy function iteration.
function ccp_mapping(P, prim, dataset)
    @unpack F_0, F_1 = dataset
    @unpack β = prim

    U_0, U_1 = payoffs(prim, dataset)   # get payoffs
    E_0 = Base.MathConstants.γ .- log.(1 .- P)  # write expectation analytically
    E_1 = Base.MathConstants.γ .- log.(P)

    F = F_0 .* (1 .- P) + F_1 .* P      # transition matrix multiplied by probability

    # write value function in terms of the CCP vector
    EU = (1 .- P) .* (U_0 + E_0) + P .* (U_1 + E_1)
    EV = inv(Matrix(I, size(P, 1), size(P, 1)) - β * F) * EU

    V = value_func(prim, dataset, EV)
    P_new = exp.(V[:, 2])./sum(exp.(V), dims = 2)

    EV, P_new
end

# pol_func_iter: This function implements the policy function iteration using CCP
function pol_func_iter(prim, dataset; tol = 1e-9)
    @unpack F_0, F_1, S = dataset

    P_old = fill(1/2, size(S, 1)) # initialize probability vector
    iter = 0
    norm = 100

    while norm > tol
        V_bar, P_new = ccp_mapping(P_old, prim, dataset)   # CCP mapping
        norm = maximum(abs.(P_old .- P_new))               # get norm
        iter += 1                                          # update iteration and P
        P_old = P_new
    end

    F_mat = F_0 .* (1 .- P_old) + F_1 .* P_old
    F_vec_old = fill(1/size(S, 1), size(S, 1))
    iter = 0
    norm = 100

    while norm > tol
        F_vec_new = F_mat' * F_vec_old
        norm = maximum(abs.(F_vec_old .- F_vec_new))
        F_vec_old = F_vec_new
        iter += 1
    end

    EV, P_old, F_vec_old
end

# get_P_hat: This function gets the P hat vector using the simulated data
function get_P_hat(prim, dataset)
    @unpack sim_data, S = dataset

    Y = sim_data[:, 1]          # get columns from sim data
    state_id = sim_data[:, 2]

    η = 1e-3

    N_hat = zeros(size(S, 1))   # count the number of observation with specific state
    N_mat = zeros(size(Y, 1), size(S, 1))
    for i in 1:size(S, 1)
        N_hat[i] = sum(state_id .== i-1)
        for j in 1:size(state_id, 1)
            if state_id[j] == (i-1)
                N_mat[j, i] = 1
            end
        end
    end

    P_hat = diag((Y' * N_mat) ./ N_hat)
    P_hat .= clamp.(P_hat, η, 1 - η)

    P_hat
end

# ---------------------------------------------------------------------------- #
# (3) Log-likelihood
# ---------------------------------------------------------------------------- #

# log_likelihood: This defines the log likelihood function
function log_likelihood(λ_input, prim, dataset)
    @unpack sim_data = dataset

    prim = Primitives(λ = λ_input[1])

    Y = sim_data[:, 1]          # get columns from sim data
    state_id = sim_data[:, 2]

    P_hat = get_P_hat(prim, dataset)            # get P hat from simulated data
    EV, CCP = ccp_mapping(P_hat, prim, dataset) # apply ccp mapping

    CCP_sim = zeros(size(state_id, 1))  # get CCP for sim data
    for i in 1:size(CCP_sim, 1)
        CCP_sim[i] = CCP[state_id[i]+1]
    end

    L = CCP_sim .* Y + (1 .- CCP_sim) .* (1 .- Y) # likelihood function

    log_prob = sum(log.(L), dims = 1)

    log_prob[1]
end

# ---------------------------------------------------------------------------- #
# (4) Calculate MLE using nested fixed point algorithm
# ---------------------------------------------------------------------------- #

# log_likelihood_nested: This function implements the nested fixed point version
#                        of log-likelihood.
function log_likelihood_nested(λ_input, prim, dataset; tol = 1e-9)
    @unpack sim_data, S = dataset

    prim = Primitives(λ = λ_input[1])

    Y = sim_data[:, 1]                  # get columns from sim data
    state_id = sim_data[:, 2]

    # implement CCP mapping
    P_old = get_P_hat(prim, dataset)    # get P hat from simulated data
    EV_old = zeros(size(S, 1))
    norm = 100

    while norm > tol
        EV_new, P_new = ccp_mapping(P_old, prim, dataset)
        norm = maximum(abs.(EV_old .- EV_new))
        EV_old = EV_new
        P_old = P_new
    end

    CCP_sim = zeros(size(state_id, 1))  # get CCP for sim data
    for i in 1:size(CCP_sim, 1)
        CCP_sim[i] = P_old[state_id[i]+1]
    end

    L = CCP_sim .* Y + (1 .- CCP_sim) .* (1 .- Y) # likelihood function

    log_prob = sum(log.(L), dims = 1)

    log_prob[1]
end
