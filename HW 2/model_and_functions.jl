# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the model, struc, and functions that support the main program.jl.
# The code below is divided into the following sections:
#    (0) initializing the algorithm
#    (1) value function iteration
#    (2) solving for the stationary distribution
#    (3) finding the asset market clearing bond price
#    (4) make plots
#    (5) welfare calculations

# ----------------------------------------------- #
# (0) strucs and functions to start the algorithm
# ----------------------------------------------- #

# Primitives: keyword-enabled structure
@with_kw struct Primitives
    β::Float64 = 0.9932      # discount rate
    α::Float64 = 1.5        # coefficient of relative risk aversion

    a_min::Float64 = -2.0   # asset lower bound
    a_max::Float64 = 5.0    # asset upper bound
    na::Int64 = 1000        # number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) # asset grid

    ns::Int64 = 2                                       # number of employment states
    s::Array{Float64, 1} = [1, 0.5]                     # employment state (e, u)
    t_matrix::Array{Float64, 2} = [0.97 0.03; 0.5 0.5] # transition matrix for employment state

end

# Results: structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2}  # value function - 2D, for employed and unemployed state
    pol_func::Array{Float64, 2}  # policy function - 2D, for employed and unemployed state
    μ::Array{Float64}            # stationary wealth distribution
end

# Loop: structure that holds main q loop parameters and variables (q is the bond
# price) This loop encompasses the smaller value function iteration and stationary
# solving algorithm.
mutable struct Loop
    tol::Float64                  # tolerance for main loop
    net_asset_supply::Float64     # initialize net asset supply value (random big number to satisfy while loop)
    q::Float64                    # q value
    converged::Float64            # converged indicator
    adjust_step::Float64          # adjustment step for adjustment method
    q_max::Float64                # q max
    q_min::Float64                # q min
end

# Initialize: function for initializing model primitives and results structs
function Initialize()
    prim = Primitives()                             # initialize primtiives
    val_func = zeros(prim.na, prim.ns)              # initial value function guess - 2D
    pol_func = zeros(prim.na, prim.ns)              # initial policy function guess - 2D
    μ = ones(prim.na*prim.ns)/(prim.na*prim.ns)     # initial wealth distribution - uniform distribution sum to 1
    res = Results(val_func, pol_func, μ)            # initialize results struct
    q_0 = (prim.β + 1)/2                            # assume 1 > q > β, so start at mid point
    tol_q = 1e-3                                    # tolerance for main loop
    q_max = 0.996                                   # q max for bisection
    q_min = prim.β                                  # q min for bisection
    loop = Loop(tol_q, 100.0, q_0, 0, 0.0, q_max, q_min)  # initialize loop variables
    prim, res, loop                                       # return deliverables
end

# ----------------------------------------------- #
#  (1) functions for value function iteration
# ----------------------------------------------- #

# Bellman: function encoding the Bellman Function, which is called repeatedly
# in the V_iterate function until convergence.
function Bellman(prim::Primitives, res::Results, q::Float64)
    @unpack val_func = res                       # unpack value function
    @unpack a_grid, β, α, na, s, t_matrix, ns = prim # unpack model primitives
    v_next = zeros(na, ns)                        # next guess of value function
    #v_next = SharedArray{Float64}(na, 2)         # next guess of value function (parallelized version)

    for (s_index, s_val) in enumerate(s)         # loop through possible employment states
        s_prob = t_matrix[s_index, :]            # get transition probabilities for current state
        choice_lower = 1                         # for exploiting monotonicity of policy function

        for a_index = 1:na                       # loop through asset grid
            a = a_grid[a_index]                  # value of a
            candidate_max = -Inf                 # initialize lowest candidate max

            # loop over possible selections of a', exploiting monotonicity of policy function
            for ap_index in choice_lower:na
                c = s_val + a - q * a_grid[ap_index]                      # consumption given a' selection

                if c > 0                                                  # check for positivity of c
                    utility = (c^(1-α) - 1)/(1 - α)                       # utility of c
                    val = utility + β * s_prob' * val_func[ap_index, :]   # compute value

                    if val > candidate_max                                # check if new value exceeds current max
                        candidate_max = val                               # if so, update max value
                        res.pol_func[a_index, s_index] = a_grid[ap_index] # update policy function for current state and asset
                        choice_lower = ap_index                           # update lowest possible choice
                    end
                end
            end
            v_next[a_index, s_index] = candidate_max # update value function
        end
    end
    v_next # return next guess of value function
end

# V_iterate: is the value function iteration loop, which calls the Bellman
# function repeatedly until we reach convergence.
function V_iterate(prim::Primitives, res::Results, q::Float64, tol::Float64 = 1e-5, err::Float64 = 100.0)
    n = 0         # counter for iteration
    converged = 0 # indicator for convergence

    println("-----------------------------------------------------------------------")
    @printf "      Starting value function iteration for bond price  %.6f \n" q
    println("-----------------------------------------------------------------------")
    while converged == 0  # keep iterating until we error less than tolerance value

        v_next = Bellman(prim, res, q)                                 # call Bellman
        err = abs.(maximum(v_next.-res.val_func))/abs(maximum(v_next)) # check for error

        if err < tol          # if error less than tolerance
            converged = 1     # we have converged
        end
        res.val_func = v_next # update val func
        n += 1                # update loop counter

    end
    println("-----------------------------------------------------------------------")
    println("       Value function converged in ", n, " iterations.")
    println("-----------------------------------------------------------------------")
end

# ----------------------------------------------- #
#  (2) functions to solve stationary distribution
# ----------------------------------------------- #


# Make_big_trans_matrix: function that creates the transition matrix that maps from
# the current state (s, a) to future state (s', a').
function Make_big_trans_matrix(prim::Primitives, res::Results)

    @unpack a_grid, na, ns, t_matrix, s = prim  # unpack model primitives
    @unpack pol_func = res                      # unpack policy function from value function iteration
    big_trans_mat = zeros(ns*na, ns*na)         # initiate transition matrix

    for (s_index, s_today) in enumerate(s)           # loop through current employment states
        for (a_index, a_today) in enumerate(a_grid)  # loop through current asset grid

            row_index = a_index + na*(s_index - 1)    # create mapping from current a, s indices to big trans matrix ROW index (today's state)
            a_choice = pol_func[a_index, s_index]     # get asset choice based on current state using policy function

            for (s_p_index, s_tomorrow) in enumerate(s)           # loop through future employment states
                for (a_p_index, a_tomorrow) in enumerate(a_grid)  # loop through future asset states

                    if a_choice == a_tomorrow    # check if a_choice from policy function matches a tomorrow (this is the indicator in T*)
                        col_index = a_p_index + na*(s_p_index - 1) # create mapping from current a, s indices to big trans matrix COLUMN index (tomorrow's state)
                        big_trans_mat[row_index, col_index] = t_matrix[s_index, s_p_index] # enter Markov transition prob for employment state
                    end
                end
            end

        end
    end
    big_trans_mat # return the big transition matrix!

end

# T_star_operator: This function defines the T star operator, which basically
# maps individuals from μ today to μ tomorrow, conditional on a' = g(a, s; q)
function T_star_operator(big_trans_mat, μ)

    μ_tomorrow = big_trans_mat' * μ   # multiply the indicator, markov transition, and current μ
    μ_tomorrow                        # return μ tomorrow

end

# T_star_iterate: This function iterates the T star operator until we converge
# at the stationary distribution
function T_star_iterate(prim::Primitives, res::Results, q::Float64, err::Float64 = 100.0)
    n = 0              # counter for iteration
    tol = 1e-5         # tolerance value
    max_iter = 1000    # limit for number of iterations
    converged = 0      # convergence indicator

    big_trans_mat = Make_big_trans_matrix(prim, res) # make big transition matrix from today to tomorrow's state

    println("-----------------------------------------------------------------------")
    @printf "        Starting T star iteration for bond price %.6f \n" q
    println("-----------------------------------------------------------------------")

    while converged == 0 && n < max_iter                       # loop until converged
        μ_new = T_star_operator(big_trans_mat, res.μ)          # apply T star operator
        err = abs.(maximum(μ_new - res.μ))/abs(maximum(μ_new)) # calculate err

        if err < tol                         # check if err within tolerance
            converged = 1
        end
        res.μ = μ_new                        # update μ
        n += 1                               # update iteration counter
    end
    println("          T star converged in ", n, " iterations.")
    println("-----------------------------------------------------------------------")
end

# ----------------------------------------------- #
# (3) functions for  asset market clearing
# ----------------------------------------------- #

# Check_asset_clearing: This function checks for the asset market clearing condition
# for the main loop that searches for the correct q (bond price). This loops
# over the value function iteration and stationary distribution solving algorithm.
function Check_asset_clearing(prim::Primitives, res::Results, loop::Loop)
    @unpack pol_func, μ = res    # unpack policy function and stationary distribution
    @unpack s, a_grid, na, β = prim # unpack primitives
    loop.net_asset_supply = 0.0  # reset net supply variable

    loop.net_asset_supply = res.μ' * vcat(a_grid, a_grid) # calculate net asset supply

    if abs(loop.net_asset_supply) < loop.tol    # check if converged
        loop.converged = 1
        println("-----------------------------------------------------------------------")
        @printf "          Main loop converged at bond price: %.6f \n" loop.q
        println("-----------------------------------------------------------------------")
    elseif loop.net_asset_supply > 0       # if agents are saving too much
                                           # we raise bond price, leading to lower interest rate
        q_hat = loop.q + (loop.q_max - loop.q)/2*abs(loop.net_asset_supply)
        println("-----------------------------------------------------------------------")
        @printf "Agents saving too much; raise bond price from %.6f to %.6f \n" loop.q q_hat
        println("-----------------------------------------------------------------------")
        loop.q = q_hat
    elseif loop.net_asset_supply < 0   # if agents are saving too little
                                       # we lower bond price, leading to higher interest rate
        q_hat = loop.q + (loop.q_min - loop.q)/2*abs(loop.net_asset_supply)
        println("-----------------------------------------------------------------------")
        @printf "Agents saving too little; drop bond price from %.6f to %.6f \n" loop.q q_hat
        println("-----------------------------------------------------------------------")
        loop.q = q_hat
    end
end

# ----------------------------------------------- #
#   (4) functions for making plots
# ----------------------------------------------- #

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
      Plots.plot!(xticks = (-2:1:3), yticks = (-2:1:3), xlims = (-2, 2.5), ylims = (-2, 2.5), xlabel = "a (assets today)", ylabel = "a' (assets tomorrow)")
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
      Plots.plot!(xlims = (-2, 2.5), xlabel = "Wealth", ylabel = "Fraction of population")
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
      Plots.plot!(xlabel = "Fraction of agents", ylabel = "Fraction of wealth")
      Plots.savefig("output/Lorenz_Curve.png")

      Calc_gini(cs_wealth_mass, cs_pop_wealth) # calculate gini index
end

# ----------------------------------------------- #
#   (5) functions for welfare calculations
# ----------------------------------------------- #


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
function Plot_λ(λ::Array{Float64, 2}, prim::Primitives, res::Results)
      @unpack a_grid = prim

      λ = Solve_λ(prim, res)

      Plots.plot(a_grid, λ[:, 1], label = "Employed", title="Consumption Equivalence Estimates")        # plot for employed
      Plots.plot!(a_grid, λ[:, 2], label = "Unemployed", legend =:topright, xlabel = "a (assets today)")   # plot for unemployed
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
