# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the Value Function Iteration (VFT) code. This program iteratively
# calls the Bellman Function to solve the Household problem, in order to find
# the value maximizing policy function.
#
# NOTE: for ease of reading/revising, this code has been copied to the centralized
# file for all supporting functions and strucs - "model_and_functions.jl"

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

##############################################################################