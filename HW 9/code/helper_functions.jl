# Author: Katherine Kwok
# Date: November 22, 2021

# This file contains the helper functions for Problem Set 2 (for JF's portion) a.k.a.
# Problem Set 9 overall. The helper functions fall into the following categories:
#
#   (0) declare primitives and other structs
#   (1) quantrature method
#   (2) GHK method
#   (3) accept/reject method
#   (4) run maximum likelihood using quadrature method



# ---------------------------------------------------------------------------- #
#   (0) declare primitives and functions for reading data
# ---------------------------------------------------------------------------- #

# Primitives: This struct stores all the parameter values for the numerical
#             integration exercise.
@with_kw struct Primitives

    x_vars = ["score_0", "rate_spread", "i_large_loan", "i_medium_loan",
              "i_refinance", "age_r", "cltv", "dti", "cu", "first_mort_r", "i_FHA",
              "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]

    z_vars = ["score_0", "score_1", "score_2"]

    y_vars = ["i_open_0", "i_open_1", "i_open_2"]

    α0::Float64 = 0
    α1::Float64 = -1
    α2::Float64 = -1
    β::Array{Float64, 1} = zeros(size(x_vars,1))
    γ::Array{Float64, 1} = fill(0.3, size(z_vars, 1))
    ρ::Float64 = 0.5
    σ::Float64 = sqrt((1/(1-ρ)^2))
end

# read_mortgage_data: This function reads and prepares the mortgage data for the
#                     numerical integration exercise

function read_mortgage_data(file_path::String, params::Primitives)
    @unpack x_vars, z_vars, y_vars = params

    dt = DataFrame(load(file_path)) # load in mortage data set

    # select time-invariant independent variables
    X = Array(select(dt, x_vars))

    # select time-varying independent variable
    Z = Array(select(dt, z_vars))

    # select dependent variable
    Y = Array(select(dt, y_vars))

    X, Z, Y
end

# read_KPU_data: This function reads and prepares the Gaussian grid and weight
#                data for the numerical integration exercise

function read_KPU_data(file_path::String)
    KPU_data = CSV.File(file_path) |> Tables.matrix
end


# ---------------------------------------------------------------------------- #
#   (1) quantrature method
# ---------------------------------------------------------------------------- #


# transform: This function transforms the KPU spare-grid grid points into the
#            proper range using log transformations. Note that this tranform
#            function only implments the upper bound version (-∞, upperbound)
function transform(grid, upperbound)
    ρ = log.(grid) .+ upperbound
    ρ_prime = ones(size(ρ, 1), size(ρ, 2)) .* (grid.^(-1))

    ρ, ρ_prime
end


# quadrature: This function applies the quadrature method to find the choice
#             probability using the Gaussian nodes and weights.
function quadrature(params, KPU_d1, KPU_d2, x, z)
    @unpack ρ, σ, α0, α1, α2, γ, β = params

    # calculate a0, a1, a2
    a0_pos = α0 + x' * β + z[1] * γ[1] # Z_i0 = Z[1]
    a1_pos = α1 + x' * β + z[2] * γ[2] # Z_i1 = Z[2]
    a2_pos = α2 + x' * β + z[3] * γ[3] # Z_i2 = Z[3]

    a0_neg = -α0 - x' * β - z[1] * γ[1] # Z_i0 = Z[1]
    a1_neg = -α1 - x' * β - z[2] * γ[2] # Z_i1 = Z[2]
    a2_neg = -α2 - x' * β - z[3] * γ[3] # Z_i2 = Z[3]

    # choice probability of T_i = 1
    prob_1 = cdf.(Normal(), a0_neg/σ)

    # choice probability of T_i = 2
    m_ϵ0, jaco0  = transform(KPU_d1[:, 1], a0_pos) # transform grid points to proper range
    density_0 = pdf.(Normal(), m_ϵ0./σ)/σ          # get densities
    prob_2 = (cdf.(Normal(), a1_neg .- ρ.*m_ϵ0) .* density_0 .* jaco0)' * KPU_d1[:, 2]

    # transform for double integration
    m_ϵ0, jaco0  = transform(KPU_d2[:, 1], a0_pos)
    m_ϵ1, jaco1 = transform(KPU_d2[:, 2], a1_pos)

    # calculate densities for choice probability
    density_0 = pdf.(Normal(), m_ϵ0./σ)/σ
    density_1 = pdf.(Normal(), m_ϵ1 .- ρ .* m_ϵ0) .* density_0

    # choice probability of T_i = 3, 4
    prob_3 = (cdf.(Normal(), a2_neg .- ρ.*m_ϵ1) .* density_1 .* jaco0 .* jaco1)' * KPU_d2[:, 3]
    prob_4 = (cdf.(Normal(), a2_pos .- ρ.*m_ϵ1) .* density_1 .* jaco0 .* jaco1)' * KPU_d2[:, 3]

    # return choice probabilities
    probs = [prob_1, prob_2[1], prob_3[1], prob_4[1]]

end

# quadrature_wrapper: This function calls the quadrature function for each
#                     observation, and modifies the paramters as given.
function quadrature_wrapper(X, Z, KPU_d1, KPU_d2, params)

    obs = size(X, 1)           # number of rows/observations
    quad_probs = zeros(obs, 4) # initialize array to store resulting choice probs

    # call quadrature function for each observation
    for index in 1:obs
        quad_probs[index, :] = quadrature(params, KPU_d1, KPU_d2, X[index, :], Z[index, :])
    end
    return quad_probs # return probs
end


# ---------------------------------------------------------------------------- #
#   (4) run maximum likelihood using quadrature method
# ---------------------------------------------------------------------------- #


# log_likelihood: This function defines log likelihood using the Gaussian
#                 Quadrature integration method.
function log_likelihood(X, Z, Y, KPU_d1, KPU_d2, θ; T = 0)
    output = 0.0 # output log likelihood

    # set parameter values
    default_param = Primitives()

    β_start = 4
    β_end = β_start + size(default_param.x_vars, 1) -1

    γ_start = β_end + 1
    γ_end = γ_start + size(default_param.z_vars, 1) -1

    params = Primitives(α0 = θ[1], α1 = θ[2], α2 = θ[3], β = θ[β_start:β_end], γ = θ[γ_start:γ_end], ρ = θ[γ_end+1])

    # call quadrature_wrapper to get choice probabilities
    quad_probs = quadrature_wrapper(X, Z, KPU_d1, KPU_d2, params)

    # output sum of log likelihood for given T = 1 or 2 or 3
    for i in 1:size(X)[1]
        output += log(quad_probs[i, T])
    end
    output = output/size(X)[1]

    output
end
