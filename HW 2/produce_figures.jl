# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains functions that are used to produce the figures to replicate
# results from Huggett (1993) after the model is solved.

# Find_a_bar: This function returns the ā value where the employed policy function
# intersects with the 45 degree line.
function Find_a_bar(na::Int64, a_grid::Array{Float64, 1}, pol_func::Array{Float64, 2})
      match_index_high = 1               # last point of intersection
      match_index_low = 1                # first point of intersection
      pol_func_employed = pol_func[:, 1] # get pol func for employed

      for i in 2:na-1 # loop through a_grid to find the intersecting values
            if a_grid[i-1] < pol_func[i-1, 1] && a_grid[i] == pol_func_employed[i]
                  match_index_low = i
            elseif a_grid[i+1] > pol_func[i+1, 1] && a_grid[i] == pol_func_employed[i]
                  match_index_high = i
            end

      end
      # find midpoint of where 45 degree line and pol_func for employed intersect
      a_bar = (pol_func[match_index_low] + pol_func[match_index_high])/2
      a_bar # return a_bar
end

# Plot_pol_func: This function plots the policy functions, 45 degree line, a_bar
function Plot_pol_func(res::Results, prim::Primitives)
      @unpack pol_func = res
      @unpack a_grid, na = prim

      # find first point where the the decision rule equals a_grid value, i.e. a_bar
      a_bar = Find_a_bar(na, a_grid, pol_func)
      # plot policy functions
      Plots.plot(a_grid, pol_func[:, 1], label = "Employed")
      Plots.plot!(a_grid, pol_func[:, 2], label = "Unemployed", title="Policy Functions")
      # plot 45 degree line
      Plots.plot!(a_grid, a_grid, label = "45 Degree Line", linestyle = :dash)
      # set plot attributes
      Plots.plot!(xticks = (-2:1:3), yticks = (-2:1:3), xlims = (-2, 2.5), ylims = (-2, 2.5))
      # plot a bar vertical line
      Plots.vline!([a_bar], label = "ā", legend = :topleft)
      Plots.savefig("output/Policy_Functions.png")
end

# Make_wealth_dist: This function takes the a grid
function Make_wealth_dist(na::Int64, μ::Array{Float64}, a_grid::Array{Float64}, s::Array{Float64})
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

            wealth_mass_employed[e_index] += μ_employed[a_index]     # add μ to the mass for employed
            wealth_mass_unemployed[u_index] += μ_unemployed[a_index] # add μ to the mass for unemployed
      end
      wealth, wealth_mass_employed, wealth_mass_unemployed
end

# Plot_wealth_dist: This function plots the wealth distribution
function Plot_wealth_dist(res::Results, prim::Primitives)
      @unpack μ = res
      @unpack a_grid, s, na = prim

      # make steady state wealth distribution using μ and a_grid
      wealth, wealth_mass_e, wealth_mass_u = Make_wealth_dist(na, μ, a_grid, s)

      Plots.plot(wealth, wealth_mass_employed, label = "Employed") # plot wealth distribution
      Plots.plot!(wealth, wealth_mass_unemployed, label = "Unemployed", title="Wealth Distribution")
      Plots.savefig("output/Wealth_Distribution.png")

      wealth, wealth_mass_e, wealth_mass_u
end

# Calc_gini: This function calculates the Gini index
function Calc_gini(cs_wealth_mass::Array{Float64}, cs_pop_wealth::Array{Float64})
      height = cs_wealth_mass .- cs_pop_wealth
      base = cs_wealth_mass[2:size(cs_wealth_mass)[1]] - cs_wealth_mass[1:size(cs_wealth_mass)[1]-1]
      A = sum(base.*height[2:size(height)[1]])  # area between lorenz curve and 45 deg line
      B = (1/2)                                 # area under 45 deg line

      Gini = A/(A + B)                # calculate gini index
      print("Gini Index is", Gini)    # print output
end

# Plot_lorenz: This function prepares the data and makes the lorenz curve
function Plot_lorenz(wealth::Array{Float64}, wealth_mass_e::Array{Float64}, wealth_mass_u::Array{Float64})

      wealth_mass = wealth_mass_e .+ wealth_mass_u # combine masses for total population mass
      pop_wealth = wealth .* wealth_mass           # get population wealth using wealth mass

      cs_wealth_mass = cumsum(wealth_mass)         # make cummulative sum of wealth mass
      cs_pop_wealth = cumsum(pop_wealth)           # make cummulative sum of population wealth

      Plots.plot(cs_wealth_mass, cs_pop_wealth, label = "Lorenz Curve", title="Lorenz Curve")                    # plot lorenz curve
      Plots.plot!(cs_wealth_mass, cs_wealth_mass, label = "45 Degree Line", linestyle = :dash, legend =:topleft) # plot 45 degree line
      Plots.savefig("output/Lorenz_Curve.png")

      Calc_gini(cs_wealth_mass, cs_pop_wealth) # calculate gini index
end
