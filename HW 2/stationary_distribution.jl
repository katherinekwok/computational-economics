# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the code for solving for the stationary wealth distribution.
# Given the policy function from the value function iteration, we use the T star
# operator to find the stationary distribution μ.

include("value_function_iteration.jl") # import the functions that solves value function iteration


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
    tol = 1e-6         # tolerance value
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


# Check_asset_clearing: This function checks for the asset market clearing condition
# for the main loop that searches for the correct q (bond price). This loops
# over the value function iteration and stationary distribution solving algorithm.
function Check_asset_clearing(prim::Primitives, res::Results, loop::Loop)
    @unpack pol_func, μ = res    # unpack policy function and stationary distribution
    @unpack s, a_grid, na, β = prim # unpack primitives
    loop.net_asset_supply = 0.0  # reset net supply variable

    loop.net_asset_supply = sum(res.μ .* vcat(a_grid, a_grid)) # calculate net asset supply

    if abs(loop.net_asset_supply) < loop.tol    # check if converged
        loop.converged = 1
        println("-----------------------------------------------------------------------")
        @printf "          Main loop converged at bond price: %.6f \n" loop.q
        println("-----------------------------------------------------------------------")
    elseif loop.net_asset_supply > 0       # if agents are saving too much
                                         # we raise bond price, leading to lower interest rate
        q_hat = loop.q + (1 - loop.q)/2* abs(loop.net_asset_supply)
        println("-----------------------------------------------------------------------")
        @printf "Agents saving too much; raise bond price from %.6f to %.6f \n" loop.q q_hat
        println("-----------------------------------------------------------------------")
        loop.q = q_hat
    elseif loop.net_asset_supply < 0   # if agents are saving too little
                                     # we lower bond price, leading to higher interest rate
        q_hat = loop.q + (prim.β - loop.q)/2* abs(loop.net_asset_supply)
        println("-----------------------------------------------------------------------")
        @printf "Agents saving too little; drop bond price from %.6f to %.6f \n" loop.q q_hat
        println("-----------------------------------------------------------------------")
        loop.q = q_hat
    end
end
