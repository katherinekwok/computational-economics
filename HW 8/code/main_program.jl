# Author: Katherine Kwok
# Date: November 8, 2021

# This file contains the code for Problem Set 1 (for JF's portion) a.k.a.
# Problem Set 8 overall. The main program does the following:

#   (0) Load in data
#   (1) Calculate log-likelihood, score of log-likelihood function, and Hessian on β_0 = -1 and β = 0
#   (2) Compare (1) with first and second numerical derivative of the log-likelihood
#   (3) Solve the maximum likelihood problem using a Newton-based algoritm.
#   (4) Compare with BFGS and Simplex


using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames                    # load packages for handling data
using FiniteDiff, Optim                        # load package for numerical derivatives and optimization
using Latexify                                 # load package for outputting results
include("helper_functions.jl")

# ---------------------------------------------------------------------------- #
# (0) Load in data
# ---------------------------------------------------------------------------- #

dt = DataFrame(load("Mortgage_performance_data.dta")) # load in data set

# variable definition
x_vars = ["i_large_loan", "i_medium_loan", "rate_spread", "i_refinance", "age_r",
          "cltv", "dti", "cu", "first_mort_r", "score_0", "score_1", "i_FHA",
          "i_open_year2", "i_open_year3", "i_open_year4", "i_open_year5"]
y_vars = ["i_close_first_year"]

X = Array(select(dt, x_vars))  # select independent variables
Y = Array(select(dt, y_vars))  # select dependent variable


# ---------------------------------------------------------------------------- #
# (1) Calculate log-likelihood, score of log-likelihood function, and Hessian on β_0 = -1 and β = 0
# ---------------------------------------------------------------------------- #

# Evaluate at β_0 = -1 and all other β = 0
β_init = vcat(-1, zeros(size(X)[2]-1))

test = log_likelihood(X, Y, β_init)
test_score = score(X, Y, β_init)
test_hessian = hessian(X, β_init)

# ---------------------------------------------------------------------------- #
# (2) Compare (1) with first and second numerical derivative of the log-likelihood
# ---------------------------------------------------------------------------- #

# calculate numerical first derivative
verify_foc = FiniteDiff.finite_difference_gradient(β_init -> log_likelihood(X, Y, β_init), β_init)

# output comparison with score of log likelihood
compare_focs = DataFrame(manual_foc = round.(test_score, digits = 2), verify_foc = round.(verify_foc, digits = 2))
latexify(compare_focs, env = :table) |> print

# calculate numerical second derivative
verify_hessian = FiniteDiff.finite_difference_hessian(β_init -> log_likelihood(X, Y, β_init), β_init)

# output comparison with hessian
verify_hessian = round.(verify_hessian, digits = 1)
latexify(verify_hessian, env = :table) |> print
test_hessian = round.(test_hessian, digits = 1)
latexify(test_hessian, env = :table) |> print


# ---------------------------------------------------------------------------- #
# (3) Solve the maximum likelihood problem using a Newton-based algoritm.
# ---------------------------------------------------------------------------- #

β_init = vcat(-1, zeros(size(X)[2]-1)) # initial guess
β_newton = @time newton_algo(X, β_init)

# ---------------------------------------------------------------------------- #
# (4) Compare with BFGS and Simplex
# ---------------------------------------------------------------------------- #

# NOTE: (1) Attempted to use initial values β_init and optimization fails. This
#           is because the optimization is sensitive to initialize values. The
#           optimization works if we initialize using results from Newton method
#           (β_newton).
#
#       (2) Since the optimize() function minimizes a given objective function,
#           we want to pass in the negative likelihood function.
β_simplex = @time optimize(β -> -log_likelihood(X, Y, β), β_newton, NelderMead())

β_bfgs = @time optimize(β -> -log_likelihood(X, Y, β), β_newton, BFGS())
