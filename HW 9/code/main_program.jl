# Author: Katherine Kwok
# Date: November 22, 2021

# This file contains the code for Problem Set 2 (for JF's portion) a.k.a.
# Problem Set 9 overall. The main program does the following:

#   (0) load data, initialize
#   (1) use quadrature method to predict choice probabilities
#   (2) use GHK method to predict choice probabilities
#   (3) use accept/reject method to predict choice probabilities

using Parameters, Plots, Printf, LinearAlgebra, Printf # load standard packages
using StatFiles, DataFrames, CSV                       # load packages for handling data
using Latexify                                         # load package for outputting results
using Distributions
include("helper_functions.jl")                         # load helper functions

root_path = pwd()                                      # set file paths
data_path = root_path * "/data/"

mortgage_data_path = data_path * "mortgage_performance_data.dta"
KPU_d1_path = data_path * "KPU_d1_l20.csv"
KPU_d2_path = data_path * "KPU_d2_l20.csv"


# ---------------------------------------------------------------------------- #
#   (0) load data, initialize prims and structs
# ---------------------------------------------------------------------------- #
params = Primitives()
X, Z = read_mortgage_data(mortgage_data_path)
KPU_d1 = read_KPU_data(KPU_d1_path)
KPU_d2 = read_KPU_data(KPU_d2_path)


# ---------------------------------------------------------------------------- #
#   (1) use quadrature method to predict choice probabilities
# ---------------------------------------------------------------------------- #

obs = size(X, 1)
quad_probs = zeros(obs, 4)

for index in 1:obs
    quad_probs[index, :] = quadrature(params, KPU_d1, KPU_d2, X[index, :], Z[index, :])
end
