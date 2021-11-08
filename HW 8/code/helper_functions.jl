# Author: Katherine Kwok
# Date: November 8, 2021

# This file contains the helper functions for Problem Set 1 (for JF's portion) a.k.a.
# Problem Set 8 overall. The helper functions fall into the following categories:
#
#   (1) Helper functions for log-likelihood, score of log-likelihood function, and Hessian.
#   (2) Helper function for implementing Newton-based algoritm.


# ---------------------------------------------------------------------------- #
# (1) Helper functions for log-likelihood, score of log-likelihood function, and Hessian.
# ---------------------------------------------------------------------------- #


# logit: This function uses the logit function to calculate the probability of
#        the outcome Y = 1 given variables X and coefficients β
#
# paramters: X - 2D array with independent variables (one row for each person)
#            β - coefficients on X variables
function logit(X, β)
    exp(X' * β)/(1 + exp(X' * β))
end

# log_likelihood: This function evaluates the log likelihood using the logit
#                 function.
# paramters: X - 2D array with independent variables (one row for each person)
#            Y - 1D array with outcome variable
#            β - coefficients on X variables
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

# score: This function evaluates the score of the log-likelihood function at a
#        given β. The returned vector is the length of the number of X variables.
#
# paramters: X - 2D array with independent variables (one row for each person)
#            Y - 1D array with outcome variable
#            β - coefficients on X variables
function score(X, Y, β)
    output = zeros(size(X)[2]) # score

    for i_person in 1:size(X)[1]
        output .+= (Y[i_person] - logit(X[i_person, :], β)) .* X[i_person, :]
    end
    output
end

# hessian: This function evaluates the hessian at a given β.
#
# paramters: X - 2D array with independent variables (one row for each person)
#            β - coefficients on X variables
function hessian(X, β)
    output = fill(0, size(X)[2], size(X)[2])

    for i in 1:size(X)[1]
        output += logit(X[i, :], β) * (1-logit(X[i, :], β)) * X[i, :] * X[i, :]'
    end
    -output
end

# ---------------------------------------------------------------------------- #
# (2) Helper function for implementing Newton-based algoritm.
# ---------------------------------------------------------------------------- #

# newton_algo: This function implements the newton-based algorithm to find the
#              optimal β for the log likelihood function.
#
# paramters: X   - 2D array with independent variables (one row for each person)
#            β   - coefficients on X variables
#            tol - tolerance value set to 10e-12 by default
#            s   - adjustment step set to 0.5 by default
function newton_algo(X, β_init; tol = 1e-10, s = 0.5)

    converged = 0       # convergence flag
    iter = 1            # iteration counter
    β_k_prev = β_init   # initial β guess
    β_k = 0.0           # initialize results

    while converged == 0
        # iterative approach to maximize L(β)
        β_k = β_k_prev .- s * hessian(X, β_k_prev)^(-1) * score(X, Y, β_k_prev)

        max_diff = maximum(abs.(β_k_prev .- β_k)) # get the max diff
        if max_diff < tol                         # check if converged
            converged = 1
            println("-----------------------------------------------------------------------")
            @printf "       Value function iteration converged after %d iterations.\n" iter
            println("-----------------------------------------------------------------------")
        end

        iter += 1
        β_k_prev = β_k
    end

    β_k # return converged β_k
end
