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

    # variable definitions
    x_vars = ["score_0", "rate_spread", "i_large_loan", "i_medium_loan",
              "i_refinance", "age_r", "cltv", "dti", "cu", "first_mort_r", "i_FHA",
              "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]

    z_vars = ["score_0", "score_1", "score_2"]

    y_vars = ["i_open_0", "i_open_1", "i_open_2"]

    # initial paramter values
    α0::Float64 = 0
    α1::Float64 = -1
    α2::Float64 = -1
    β::Array{Float64, 1} = zeros(size(x_vars,1))
    γ::Array{Float64, 1} = fill(0.3, size(z_vars, 1))
    ρ::Float64 = 0.5
    σ::Float64 = (1/(1-ρ)^2)

    # simulation related variables
    n_simu::Int64 = 100 # number of simulations

end

# read_mortgage_data: This function reads and prepares the mortgage data for the
#                     numerical integration exercise

function read_mortgage_data(file_path::String, param::Primitives)
    @unpack x_vars, z_vars, y_vars = param

    dt = DataFrame(load(file_path)) # load in mortage data set

    # select time-invariant independent variables
    X = Array(select(dt, x_vars))

    # select time-varying independent variable
    Z = Array(select(dt, z_vars))

    # select dependent variable
    Y = Array(select(dt, y_vars)) # get i_open_0, i_open_1, i_open_2
    Y = 1 .- Y                    # convert to indicator is loan is paid

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

# calculate_conditions: This helper function calculates the conditions using
#                       parameters and observed data the error terms in calculating
#                       choice probabilities.
function calculate_conditions(param, x, z)
    @unpack ρ, σ, α0, α1, α2, γ, β = param

    a0_pos = α0 + x' * β + z' * γ
    a1_pos = α1 + x' * β + z' * γ
    a2_pos = α2 + x' * β + z' * γ

    a0_neg = -α0 - x' * β - z' * γ
    a1_neg = -α1 - x' * β - z' * γ
    a2_neg = -α2 - x' * β - z' * γ

    a0_pos, a1_pos, a2_pos, a0_neg, a1_neg, a2_neg

end


# quadrature: This function applies the quadrature method to find the choice
#             probability using the Gaussian nodes and weights.
function quadrature(param, KPU_d1, KPU_d2, x, z)
    @unpack ρ, σ, α0, α1, α2, γ, β = param

    # calculate a0, a1, a2
    a0_pos, a1_pos, a2_pos, a0_neg, a1_neg, a2_neg = calculate_conditions(param, x, z)

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
function quadrature_wrapper(X, Z, KPU_d1, KPU_d2, param)

    obs = size(X, 1)           # number of rows/observations
    quad_probs = zeros(obs, 4) # initialize array to store resulting choice probs

    # call quadrature function for each observation
    for index in 1:obs
        quad_probs[index, :] = quadrature(param, KPU_d1, KPU_d2, X[index, :], Z[index, :])
    end
    return quad_probs # return probs
end

# ---------------------------------------------------------------------------- #
#   (2) GHK method
# ---------------------------------------------------------------------------- #




# ---------------------------------------------------------------------------- #
#   (3) accept/reject method
# ---------------------------------------------------------------------------- #

# draw_errors: This function samples a specified number of the error terms (ϵ, η)
#              from a uniform distribution (0,1) then transforms it using the
#              inverse CDF method to the appropriate distribution. Then, ϵ_i0,
#              ϵ_i1, ϵ_i2 are computed and returned.
function draw_errors(param)
    @unpack n_simu, σ, ρ = param

    # do random draws
    ϵ_i0 = rand(Uniform(0, 1), n_simu)
    η_i1 = rand(Uniform(0, 1), n_simu)
    η_i2 = rand(Uniform(0, 1), n_simu)

    # transform from uniform to normal distributions using inverse CDF (i.e. quantile)
    ϵ_i0 = quantile.(Normal(0, σ), ϵ_i0)
    η_i1 = quantile.(Normal(0, 1), η_i1)
    η_i2 = quantile.(Normal(0, 1), η_i2)

    # compute ϵ_i1 and ϵ_i2
    ϵ_i1 = ρ*ϵ_i0 + η_i1
    ϵ_i2 = ρ*ϵ_i1 + η_i2

    return ϵ_i0, ϵ_i1, ϵ_i2
end

# accept_reject: This function implements the accept reject algorithm for a
#                single obseration in the data set.
@everywhere function accept_reject(param, x, z, ϵ_i0, ϵ_i1, ϵ_i2)
    @unpack n_simu = param

    # initialize the counters for each choice
    count_T_1 = 0
    count_T_2 = 0
    count_T_3 = 0
    count_T_4 = 0

    # calculate conditions on the error terms
    a0_pos, a1_pos, a2_pos, a0_neg, a1_neg, a2_neg = calculate_conditions(param, x, z)

    for i in 1:n_simu
        if ϵ_i0[i] < a0_neg
            count_T_1 += 1 # if satisfied condition for loan period T = 1

        elseif ϵ_i0[i] < a0_pos && ϵ_i1[i] < a1_neg
            count_T_2 += 1 # if satisfied condition for loan period T = 2

        elseif ϵ_i0[i] < a0_pos && ϵ_i1[i] < a1_pos && ϵ_i2[i] < a2_neg
            count_T_3 += 1 # if satisfied condition for loan period T = 3

        elseif ϵ_i0[i] < a0_pos && ϵ_i1[i] < a1_pos && ϵ_i2[i] < a2_pos
            count_T_4 += 1 # if satisfied condition for loan period T = 4
        end
    end

    # calculate choice probability by dividing accepted counts by total # of simulations
    probs = [count_T_1, count_T_2, count_T_3, count_T_4]./n_simu

    return probs
end

# accept_reject_wrapper: This function calls the accept_reject function for each
#                     observation, and modifies the paramters as given.
function accept_reject_wrapper(X, Z, param)

    obs = size(X, 1)                         # number of rows/observations
    a_r_probs = SharedArray{Float64}(obs, 4) # initialize array to store resulting choice probs

    ϵ_i0, ϵ_i1, ϵ_i2 = draw_errors(param)    # draw errors

    # call quadrature function for each observation
    @sync @distributed for index in 1:obs
        a_r_probs[index, :] = accept_reject(param, X[index, :], Z[index, :], ϵ_i0, ϵ_i1, ϵ_i2)
    end
    return a_r_probs # return probs
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

    param = Primitives(α0 = θ[1], α1 = θ[2], α2 = θ[3], β = θ[β_start:β_end], γ = θ[γ_start:γ_end], ρ = θ[γ_end+1])

    # call quadrature_wrapper to get choice probabilities
    quad_probs = quadrature_wrapper(X, Z, KPU_d1, KPU_d2, param)

    # output sum of log likelihood for given T = 1 or 2 or 3
    for i in 1:size(X)[1]
        product = quad_probs[i, T]^(Y[i, T]) + (1-quad_probs[i, T])^(1-Y[i, T])

        if product > 0
            output += log(product)
        end
    end

    output
end
