# Author: Katherine Kwok
# Date: December 5, 2021

# This file contains the code for Problem Set 3 (for JF's portion) a.k.a.
# Problem Set 10 overall. The main program implements the BLP algorithm.

using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames, CSV                       # load packages for handling data
using Latexify                                         # load package for outputting results
using Distributions, Optim, FiniteDiff, Random, StatsBase

include("helper_functions.jl")                         # load helper functions

root_path = pwd()                                      # set file paths
data_path = root_path * "/data/"
output_path = root_path * "/output/"

type_data_path =  data_path * "simulated_type_distribution.dta"
car_char_path = data_path * "car_demand_characteristics_spec1.dta"
car_iv_path = data_path * "car_demand_iv_spec1.dta"

# ---------------------------------------------------------------------------- #
#   (0) load data and set up variables
# ---------------------------------------------------------------------------- #

prim = Primitives()
dataset = process_data(prim, car_char_path, car_iv_path, type_data_path)

# ---------------------------------------------------------------------------- #
#   (1) invert demand using contraction mapping and newton
# ---------------------------------------------------------------------------- #
λ_p = 0.6       # parameter value
tol = 10e-12    # tolerance for convergence

# fixed point method - contraction mapping for first year (max_T = 1)
inverse(dataset, λ_p, 0, tol, output_path, "fixed_point"; max_iter = 1000, max_T = 1)

# newton method for first year (max_T = 1)
dataset.delta_0 = dataset.delta_iia
vdelta = inverse(dataset, λ_p, 1, tol, output_path, "newton"; max_iter = 1000, max_T = 1)

# ---------------------------------------------------------------------------- #
#   (2) grid search over non-linear parameter
# ---------------------------------------------------------------------------- #


λ_p = [0.6]
p_grid = range(0, step = 0.1, stop = 1)
q_grid = zeros(size(λ_p, 1), size(p_grid, 1))

for i in 1:size(λ_p, 1)
    for j in 1:size(p_grid, 1)
        λ_p[i] = p_grid[j]
        q = gmm_obj(dataset, λ_p, 0, 0, tol, output_path, "newton")

        if isnan.(q) != 1
            q_grid[i, j] = -q[1]
        else
            q = 100
            dataset.delta_0 = dataset.delta_iia
        end
        min_val, min_ind = findmin(q_grid[i, :])
        λ_p[i] = p_grid[min_ind]
    end
    plt = plot(p_grid, q_grid[i, :])
    display(plt)
end

# ---------------------------------------------------------------------------- #
#   (3) estimate paramter using 2-step GMM
# ---------------------------------------------------------------------------- #
