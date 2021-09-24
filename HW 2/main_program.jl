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
include("stationary_distribution.jl")  # import the functions that solves for stationary distribution


# ----------------------------------------------- #
#  (0) initialize things for algorithm
# ----------------------------------------------- #

prim, res, loop = Initialize()    # initialize primitives, results, loop struct

while loop.converged == 0

      loop.q = (loop.q_max + loop.q_min)/2 # use bisection method to update q
      #loop.adjust_step = 0.01 * loop.q

      # ----------------------------------------------- #
      #  (1) value function iteration
      # ----------------------------------------------- #

      @time V_iterate(prim, res, loop.q)

      # ----------------------------------------------- #
      #  (2) solve for the stationary distribution
      # ----------------------------------------------- #

      @time T_star_iterate(prim, res, loop.q)

      # ----------------------------------------------- #
      # (3) check asset market clearing
      # ----------------------------------------------- #
      @unpack pol_func, μ = res    # unpack policy function and stationary distribution
      @unpack s, a_grid, na = prim # unpack primitives
      loop.net_asset_supply = 0.0  # reset net supply variable

      for (s_index, s) in enumerate(s)                  # loop through current employment states
            for (a_index, a) in enumerate(a_grid)       # loop through current asset grid
                  μ_index = a_index + na*(s_index - 1)                         # get mapping to μ index from s, a
                  loop.net_asset_supply += pol_func[a_index, s_index] * μ[μ_index]  # sum to net asset supply
            end
      end

      if abs(loop.net_asset_supply) < loop.tol    # check if converged
            loop.converged = 1
            println("---------------------------------------------------------------")
            println("          Main loop converged at bond price: ", loop.q)
            println("---------------------------------------------------------------")
      elseif loop.net_asset_supply > 0       # if agents are saving too much
                                             # we raise bond price, leading to lower interest rate
            loop.q_min = loop.q             # for bisection method
            #loop.q = loop.q + loop.adjust_step
            println("---------------------------------------------------------------")
            println("          Agents saving too much; raise bond price")
            println("---------------------------------------------------------------")
      elseif loop.net_asset_supply < 0   # if agents are saving too little
                                         # we lower bond price, leading to higher interest rate
            loop.q_max = loop.q         # for bisection method
            #loop.q = loop.q - loop.adjust_step
            println("---------------------------------------------------------------")
            println("          Agents saving too little; lower bond price")
            println("---------------------------------------------------------------")
      end
end

# ----------------------------------------------- #
#  (4) make plots
# ----------------------------------------------- #
