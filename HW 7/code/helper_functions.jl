# Author: Katherine Kwok
# Date: November 17, 2021

# This file contains the code for Problem Set 7, where we estimate the AR(1)
# process using simulated method of moments (SMM). This helper function file
# contains the helper functions that implment the SMM algorithm as follows:
#
#   (1) Get data moments from the true data generating process
#   (2) Minimize objective function where weight matrix = indentity
#   (3) Compute SE for the estimated params above
#   (4) Minimize objective function where weight matrix = variance-cov matrix (Newey-West)
#   (5) Compute SE for the estimated params above
#
# The helper functions are roughly divided into the following categories:
#
#   (1) Getting moments (simulate AR(1) given parameters; both data and model)
#   (2) Minimizing objective function
#   (3) Computing SE
#   (4) Wrapper function for full SMM algorithm

# ------------------------------------------------------------------------ #
# (1) Getting moments (simulate AR(1) given parameters; both data and model)
# ------------------------------------------------------------------------ #

function get_moments()

    # simulate H different AR(1) sequences of length T
    # compute moments given specified requests (mean, variance, autocorrelation) within each H
    # compute average across all H

end

# ------------------------------------------------------------------------ #
# (2) Minimizing objective function
# ------------------------------------------------------------------------ #

function objective_func()
    # call get_moments to get model moments
    # define J (objective function)
end

function newey_west()
    # see reference code from Phil and slides
    # need to call the gamma function twice, once for standard and once for lags
end

# ------------------------------------------------------------------------ #
# (3) Computing SE
# ------------------------------------------------------------------------ #

function compute_se()
    # calculate numerical derivative using get_moments function
end

# ------------------------------------------------------------------------ #
# (4) Wrapper function for SMM algorithm
# ------------------------------------------------------------------------ #

function solve_smm()
    # see phil's code for reference

    # get_moments for true data

    # minimize objective function with W = I

    # compute SE of parameters that minimize above

    # minimize objective function with W = S^-1

    # compute SE of parameters that minimize above

end
