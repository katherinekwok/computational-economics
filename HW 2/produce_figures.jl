# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains functions that are used to produce the figures to replicate
# results from Huggett (1993) after the model is solved.

# Find_a_bar: This function returns the ā value where the employed policy function
# intersects with the 45 degree line.
#
# NOTE: for ease of reading/revising, this code has been copied to the centralized
# file for all supporting functions and strucs - "model_and_functions.jl"


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
      println("-----------------------------------------------------------------------")
      @printf "                        ā is %.4f \n" a_bar
      println("-----------------------------------------------------------------------")
      round(a_bar, digits = 3) # return a_bar rounded
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
      Plots.vline!([a_bar], label = "ā = $a_bar", legend = :topleft)
      Plots.savefig("output/Policy_Functions.png")
end

# Make_wealth_dist: This function takes the a grid
function Make_wealth_dist(na::Int64, μ::Array{Float64}, a_grid::Array{Float64}, s::Array{Float64})
      μ_employed = μ[1:na, :]        # μ for employed
      μ_unemployed = μ[na+1:na*2, :] # μ for unemployed

      wealth_employed = a_grid .+ s[1]     # employed wealth (a + income)
      wealth_unemployed = a_grid .+ s[2]   # unemployed wealth (a + income)

      wealth = sort(unique(vcat(wealth_employed, wealth_unemployed))) # get a unique array of wealth across employed and unemployed states
      wealth_mass_e = zeros(length(wealth))              # initiate distribution of wealth for employed
      wealth_mass_u = zeros(length(wealth))            # initiate distribution of wealth for unemployed

      for a_index in 1:na    # loop through a_grid to map the μ values for each wealth level in unique distribution
            e_index = indexin(wealth_employed[a_index], wealth)[1] # find index where wealth_employed matches unique wealth distribution
            u_index = indexin(wealth_unemployed[a_index], wealth)[1] # find index where wealth_employed matches unique wealth distribution

            wealth_mass_e[e_index] += μ_employed[a_index]     # add μ to the mass for employed
            wealth_mass_u[u_index] += μ_unemployed[a_index] # add μ to the mass for unemployed
      end
      wealth, wealth_mass_e, wealth_mass_u
end

# Plot_wealth_dist: This function plots the wealth distribution
function Plot_wealth_dist(res::Results, prim::Primitives)
      @unpack μ = res
      @unpack a_grid, s, na = prim

      # make steady state wealth distribution using μ and a_grid
      wealth, wealth_mass_e, wealth_mass_u = Make_wealth_dist(na, μ, a_grid, s)

      Plots.plot(wealth, wealth_mass_e, label = "Employed", legend=:topleft) # plot wealth distribution
      Plots.plot!(wealth, wealth_mass_u, label = "Unemployed", title="Wealth Distribution")
      Plots.plot!(xlims = (-2, 2.5))
      Plots.savefig("output/Wealth_Distribution.png")

      wealth, wealth_mass_e, wealth_mass_u
end

# Calc_gini: This function calculates the Gini index
function Calc_gini(cs_wealth_mass::Array{Float64}, cs_pop_wealth::Array{Float64})
      height = cs_wealth_mass .- cs_pop_wealth
      base = cs_wealth_mass[2:size(cs_wealth_mass)[1]] - cs_wealth_mass[1:size(cs_wealth_mass)[1]-1]
      A = base' * height[2:size(height)[1]]  # area between lorenz curve and 45 deg line
      B = (1/2)                              # area under 45 deg line

      Gini = A/B                      # calculate gini index
      println("-----------------------------------------------------------------------")
      @printf "                       Gini Index is %.4f \n" Gini    # print output
      println("-----------------------------------------------------------------------")

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

# Solve_w_fb: This function solves for welfare in first best allocation
function Solve_w_fb(prim::Primitives)
      @unpack t_matrix, s, α, β = prim

      invariant_dist = (t_matrix^100000)[1,:]                    # iterate forward to get invariant distribution for e vs. u
      c_fb = s[1] * invariant_dist[1] + s[2] * invariant_dist[2] # calculate consumption
      w_fb = (c_fb^(1-α) - 1)/((1-α) * (1 - β))                  # calculate welfare in first best, infinite sum of discounted utility of consumption

      println("-----------------------------------------------------------------------")
      @printf "                       W_FB is %.4f \n" w_fb    # print output
      println("-----------------------------------------------------------------------")
      w_fb
end

# Solve_λ: This function solves for the λ value
function Solve_λ(prim::Primitives, res::Results)
      @unpack α, β = prim

      w_fb = Solve_w_fb(prim)               # call function to solve for w_fb
      A = w_fb + (1/((1-α)*(1-β)))          # numerator in formula for λ
      B = res.val_func .+ (1/((1-α)*(1-β))) # denominator in formula for λ

      λ = (A./B).^(1/(1-α)) .- 1            # calculate λ using formula
      λ # return result
end

# Plot_λ: This function plots the λ for employed and unemployed
function Plot_λ(λ::Array{Float64, 2}, prim::Primitives)
      @unpack a_grid = prim

      Plots.plot(a_grid, λ[:, 1], label = "Employed", title="λ(a, s)")        # plot for employed
      Plots.plot!(a_grid, λ[:, 2], label = "Unemployed", legend =:topright)   # plot for unemployed
      Plots.savefig("output/Lambda.png")
end

# Calc_welfare: This function calculates welfare in the incomplete market
# and welfare gains going from incomplete to complete markets
function Calc_welfare(prim::Primitives, res::Results, λ::Array{Float64, 2})
      @unpack μ, val_func = res
      @unpack na, ns = prim

      w_inc = μ[1:na, ]' * val_func[:, 1] + μ[na+1:ns*na, ]' * val_func[:, 2] # welfare for incomplete market
      w_g = λ[:, 1]' * μ[1:na, ] + λ[:, 2]' * μ[na+1:ns*na, ] # welfare gain

      println("-----------------------------------------------------------------------")
      @printf "                      W_INC is %.6f \n" w_inc    # print output
      println("-----------------------------------------------------------------------")
      println("-----------------------------------------------------------------------")
      @printf "                       W_G is %.6f \n" w_g    # print output
      println("-----------------------------------------------------------------------")

      fraction = 0 # calculate fraction of people willing to go to complete market
      for a_index in 1:na
            for s_index in 1:ns
                  if λ[a_index, s_index] >= 0
                        mu_index = a_index + na*(s_index - 1)
                        fraction += μ[mu_index]
                  end
            end
      end
      println("-----------------------------------------------------------------------")
      @printf "       Fraction of population in favor of complete mkt %.4f \n" fraction    # print output
      println("-----------------------------------------------------------------------")

end
