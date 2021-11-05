# Author: Katherine Kwok
# Date: November 8, 2021

# This file contains the code for Problem Set 1 (for JF's portion)/Problem Set 8 overall
# The main program does the following:
#
#   (1) Calculate log-likelihood, score of log-likelihood function, and Hessian on β_0 = -1 and β = 0
#   (2) Compare (1) with first and second numerical derivative of the log-likelihood
#   (3) Solve the maximum likelihood problem using a Newton-based algoritm.
#   (4) Compare with BFGS and Simplex


using Parameters, Plots, Printf, LinearAlgebra # load standard packages
using StatFiles, DataFrames                    # load packages for handling data
using FiniteDiff, Optim                        # load package for numerical derivatives and optimization
using Latexify                                 # load package for outputting results

# load in data set
dt = DataFrame(load("Mortgage_performance_data.dta"))

# ---------------------------------------------------------------------------- #
# (1) Calculate log-likelihood, score of log-likelihood function, and Hessian on β_0 = -1 and β = 0
# ---------------------------------------------------------------------------- #

# variable definition
x_vars = ["i_large_loan", "i_medium_loan", "rate_spread", "i_refinance", "age_r",
          "cltv", "dti", "cu", "first_mort_r", "score_0", "score_1", "i_FHA",
          "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]
y_vars = ["i_close_first_year"]

X = Array(select(dt, x_vars))  # select independent variables
Y = Array(select(dt, y_vars))  # select dependent variable


# log-likelihood

# logit: This function uses the logit function to calculate the probability of
#        the outcome Y = 1 given variables X and coefficients β
function logit(X, β)
    exp(X' * β)/(1 + exp(X' * β))
end

# log_likelihood: This function evaluates the log likelihood using the logit
#                 function.
function log_likelihood(X, Y, β)
    output = 0.0

    for i in 1:size(X)[1]
        product = logit(X[i, :], β)^Y[i] * (1-logit(X[i, :], β))^(1-Y[i])

        if product > 0
            output += log(product)
        end
    end
    output
end

# score: This function evaluates the score of the log-likelihood function
function score(X, Y, β)
    output = zeros(size(X)[2]) # score

    for i_person in 1:size(X)[1]
        output .+= (Y[i_person] - logit(X[i_person, :], β)) .* X[i_person, :]
    end
    output
end

# hessian: This function evaluates the hessian
function hessian(X, β)
    output = fill(0, size(X)[2], size(X)[2])

    for i in 1:size(X)[1]
        output += logit(X[i, :], β) * (1-logit(X[i, :], β)) * X[i, :] * X[i, :]'
    end
    -output
end

# Evaluate at β_0 = -1 and β = 0
β = vcat(-1, zeros(size(X)[2]-1))

test = log_likelihood(X, Y, β)
test_score = score(X, Y, β)
test_hessian = hessian(X, β)

# ---------------------------------------------------------------------------- #
# (2) Compare (1) with first and second numerical derivative of the log-likelihood
# ---------------------------------------------------------------------------- #
verify_foc = FiniteDiff.finite_difference_gradient(β -> log_likelihood(X, Y, β), β)
compare_focs = DataFrame(manual_foc = round.(test_score, digits = 2), verify_foc = round.(verify_foc, digits = 2))
latexify(compare_focs, env = :table) |> print

verify_hessian = FiniteDiff.finite_difference_hessian(β -> log_likelihood(X, Y, β), β)

verify_hessian = round.(verify_hessian, digits = 1)
latexify(verify_hessian, env = :table) |> print
test_hessian = round.(test_hessian, digits = 1)
latexify(test_hessian, env = :table) |> print


# ---------------------------------------------------------------------------- #
# (3) Solve the maximum likelihood problem using a Newton-based algoritm.
# ---------------------------------------------------------------------------- #

β_0 = vcat(-1, zeros(size(X)[2]-1)) # initial guess
β_k_prev = β_0

converged = 0   # convergence flag
tol = 10e-12    # tolerance value
s = 0.5         # adjustment step
iter = 1        # iteration counter

while converged == 0
    β_k = β_k_prev .- s * hessian(X, β_k_prev)^(-1) * score(X, Y, β_k_prev)'

    max_diff = maximum(abs.(β_k_prev .- β_k))
    if max_diff < tol
        converged = 1
        println(β_k)
    end

    println(β_k)
    β_k_prev = β_k
end

# ---------------------------------------------------------------------------- #
# (4) Compare with BFGS and Simplex
# ---------------------------------------------------------------------------- #
