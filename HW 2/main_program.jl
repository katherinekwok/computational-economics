# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the code for Problem Set 2, where we want to compute the
# cross-sectional wealth distribution.
#    (0) initializing the algorithm
#    (1) value function iteration
#    (2) solving for the stationary distribution
#    (3) finding the asset market clearing bond price
#    (4) make plots

using Distributed, SharedArrays # load package for running julia in parallel
using Parameters, Plots
include("value_function_iteration.jl") # import the functions that solves value function iteration
include("stationary_distribution.jl")     # import the functions that solves for stationary distribution


prim, res, loop = Initialize()    # initialize primitives, results, loop struct

while loop.converged == 0

      @time V_iterate(prim, res, loop.q)      #  (1) value function iteration

      @time T_star_iterate(prim, res, loop.q) #  (2) solve for the stationary distribution

      # ----------------------------------------------- #
      # (3) check asset market clearing
      # ----------------------------------------------- #
      @unpack pol_func, μ = res    # unpack policy function and stationary distribution
      @unpack s, a_grid, na, β = prim # unpack primitives
      loop.net_asset_supply = 0.0  # reset net supply variable

      loop.net_asset_supply = -sum(res.μ .* vcat(a_grid, a_grid)) # calculate net asset supply

      if abs(loop.net_asset_supply) < loop.tol    # check if converged
            loop.converged = 1
            println("---------------------------------------------------------------")
            println("          Main loop converged at bond price: ", loop.q)
            println("---------------------------------------------------------------")
      elseif loop.net_asset_supply > 0       # if agents are saving too much
                                             # we raise bond price, leading to lower interest rate
            q_hat = loop.q + (prim.β - loop.q)/2 * abs(loop.net_asset_supply)
            println("---------------------------------------------------------------")
            println("Agents saving too much; raise bond price from ", loop.q, " to ", q_hat)
            println("---------------------------------------------------------------")
            loop.q = q_hat
      elseif loop.net_asset_supply < 0   # if agents are saving too little
                                         # we lower bond price, leading to higher interest rate
            q_hat = loop.q + (1 - loop.q)/2 * abs(loop.net_asset_supply)
            println("---------------------------------------------------------------")
            println("Agents saving too little; drop bond price from ", loop.q, " to ", q_hat)
            println("---------------------------------------------------------------")
            loop.q = q_hat
      end

end

# ----------------------------------------------- #
#  (4) make plots
# ----------------------------------------------- #
@unpack pol_func, μ = res
@unpack a_grid, s, na = prim

# (0) find first point where the the decision rule equals a_grid value, i.e. a_bar
global match_index = 1
for i in 2:na
      if a_grid[i-1] < pol_func[i-1, 1] && a_grid[i] == pol_func[i, 1]
            global match_index = i
      end
end

# (1) policy functions
Plots.plot(a_grid, pol_func[:, 1], label = "Employed")
Plots.plot!(a_grid, pol_func[:, 2], label = "Unemployed", title="Policy Functions")
Plots.plot!(a_grid, a_grid, label = "45 Degree Line", linestyle = :dash)
Plots.vline!([a_grid[match_index]], label = "ā", legend = :bottomright)
xticks!(-2:1:3)
yticks!(-2:1:3)
xlims!(-2, 3)
ylims!(-2, 3)
Plots.savefig("output/Policy_Functions.png")


# (2) stationary distributions
μ_employed = μ[1:na, :]        # μ for employed
μ_unemployed = μ[na+1:na*2, :] # μ for unemployed

wealth_employed = a_grid .+ s[1]     # employed wealth (a + income)
wealth_unemployed = a_grid .+ s[2]   # unemployed wealth (a + income)

wealth = sort(unique(vcat(wealth_employed, wealth_unemployed))) # get a unique array of wealth across employed and unemployed states
wealth_mass_employed = zeros(length(wealth))              # initiate distribution of wealth for employed
wealth_mass_unemployed = zeros(length(wealth))            # initiate distribution of wealth for unemployed

for a_index in 1:na    # loop through a_grid to map the μ values for each wealth level in unique distribution
      e_index = indexin(wealth_employed[a_index], wealth)[1] # find index where wealth_employed matches unique wealth distribution
      u_index = indexin(wealth_unemployed[a_index], wealth)[1] # find index where wealth_employed matches unique wealth distribution

      wealth_mass_employed[e_index] += μ_employed[a_index]   # add μ to the mass for employed
      wealth_mass_unemployed[u_index] += μ_unemployed[a_index] # add μ to the mass for unemployed
end


Plots.plot(wealth, wealth_mass_employed, label = "Employed")
Plots.plot!(wealth, wealth_mass_unemployed, label = "Unemployed", title="Wealth Distribution")

Plots.savefig("output/Wealth_Distribution.png")


# lorenz curve
