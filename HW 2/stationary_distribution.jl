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
