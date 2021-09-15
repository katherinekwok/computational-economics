# Author: Katherine Kwok
# Date: Sept 14, 2021
#
# This program plots the results from the fortran code.

using Parameters, Plots, DelimitedFiles

# Define file paths
root = "/Users/s140692/Dropbox/UW Madison/Second Year/899 - Computational/computational-economics/HW 1/"
code = root * "fortran_code/"

# Define variables and arrays to store the results
nk = 1000                       # number of capital grid points
val_func = zeros(nk, 2)    # for val func
pol_func = zeros(nk, 2)    # for pol func
k_grid   = zeros(nk)       # for k grid

# Read results from Fortran code
datafile_g = open(code*"pfs_neogrowth_good.dat","r")
    good_res = readlines(datafile_g)
    close(datafile_g)

datafile_b = open(code*"pfs_neogrowth_bad.dat","r")
    bad_res = readlines(datafile_b)
    close(datafile_b)

# Convert results to desired format
for (i, good_line) in enumerate(good_res)

    # value function
    val_func[i, 1] = parse(Float64, split(good_res[i])[4])
    val_func[i, 2] = parse(Float64, split(bad_res[i])[4])

    # policy function
    pol_func[i, 1] = parse(Float64, split(good_res[i])[3])
    pol_func[i, 2] = parse(Float64, split(bad_res[i])[3])

    # k_grid
    k_grid[i] = parse(Float64, split(good_res[i])[1])

end

# Plot fortran results

#value function
Plots.plot(k_grid, val_func[:, 1], label = "Good State")
Plots.plot!(k_grid, val_func[:, 2], label = "Bad State", title="Value Function (Fortran)")
Plots.savefig("02_Value_Functions_Fortran.png")

#policy functions
Plots.plot(k_grid, pol_func[:, 1], label = "Good State")
Plots.plot!(k_grid, pol_func[:, 2], label = "Bad State", title="Policy Function (Fortran)")
Plots.savefig("02_Policy_Functions_Fortran.png")

#changes in policy function
pol_func_δ_good = copy(pol_func[:, 1]).-k_grid
pol_func_δ_bad = copy(pol_func[:, 2]).-k_grid
Plots.plot(k_grid, pol_func_δ_good, label = "Good State")
Plots.plot!(k_grid, pol_func_δ_bad, label = "Bad State",title="Policy Functions Changes (Fortran)")
Plots.savefig("02_Policy_Functions_Changes_Fortran.png")
