# Edited by Katherine Kwok
# Date: Sept 14, 2021
#
# This code solves the stochastic economic growth model. This program
# calls 02_growth_model.jl, which defines, initializes, and solves for
# the equilibrium. Then, we plot the output value function and policy function.

using Parameters, Plots
include("02_Growth_model.jl") # import the functions that solve our growth model

prim, res = Initialize() # initialize primitive and results structs
time = @elapsed Solve_model(prim, res) # solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim

#----------------#
# Make plots
#----------------#

#value function
Plots.plot(k_grid, val_func[:, 1], label = "Good State")
Plots.plot!(k_grid, val_func[:, 2], label = "Bad State", title="Value Function (Julia)")
Plots.savefig("02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func[:, 1], label = "Good State")
Plots.plot!(k_grid, pol_func[:, 2], label = "Bad State", title="Policy Function (Julia)")
Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ_good = copy(pol_func[:, 1]).-k_grid
pol_func_δ_bad = copy(pol_func[:, 2]).-k_grid
Plots.plot(k_grid, pol_func_δ_good, label = "Good State")
Plots.plot!(k_grid, pol_func_δ_bad, label = "Bad State",title="Policy Functions Changes (Julia)")
Plots.savefig("02_Policy_Functions_Changes.png")

println("Completed program in $time seconds")
################################
