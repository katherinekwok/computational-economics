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
