# Author: Katherine Kwok
# Date: Sept 22, 2021

# This file contains the Value Function Iteration (VFT) code. This program iteratively
# calls the Bellman Function to solve the Household problem, in order to find
# the value maximizing policy function.

# Primitives: keyword-enabled structure
@with_kw struct Primitives
    β::Float64 = 0.9932     # discount rate
    α::Float64 = 1.5        # coefficient of relative risk aversion

    a_min::Float64 = -2.0   # asset lower bound
    a_max::Float64 = 5.0    # asset upper bound
    na::Int64 = 701         # number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max)) # asset grid

    ns::Int64 = 2                                       # number of employment states
    s::Array{Float64, 1} = [1, 0.5]                     # employment state (e, u)
    t_matrix::Array{Float64, 2} = [0.97 0.03; 0.05 0.5] # transition matrix for employment state

end

# Results: structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2}  # value function - 2D, for employed and unemployed state
    pol_func::Array{Float64, 2}  # policy function - 2D, for employed and unemployed state
    μ::Array{Float64}            # stationary wealth distribution
end

# Loop: structure that holds model loop parameters and variables
mutable struct Loop
    tol::Float64                  # tolerance for main loop
    net_asset_supply::Float64     # initialize net asset supply value (random big number to satisfy while loop)
    q_min::Float64                # lower bound for q for bisection method
    q_max::Float64                # upper bound for q for bisection method
    q::Float64                    # q value
    converged::Float64            # converged indicator
    adjust_step::Float64          # adjustment step for adjustment method
end

# Initialize: function for initializing model primitives and results structs
function Initialize()
    prim = Primitives()                             # initialize primtiives
    val_func = zeros(prim.na, 2)                    # initial value function guess - 2D
    pol_func = zeros(prim.na, 2)                    # initial policy function guess - 2D
    μ = ones(prim.na*prim.ns)/(prim.na*prim.ns)     # initial wealth distribution - uniform distribution sum to 1
    res = Results(val_func, pol_func, μ)            # initialize results struct
    loop = Loop(1e-5, 100.0, 0.0, 0.9, 0.9, 0, 0.0) # initialize loop variables
    prim, res, loop                                 # return deliverables
end

# Bellman Operator
function Bellman(prim::Primitives, res::Results, q::Float64)
    @unpack val_func = res                       # unpack value function
    @unpack a_grid, β, α, na, s, t_matrix = prim # unpack model primitives
    v_next = zeros(na, 2)                        # next guess of value function
    #v_next = SharedArray{Float64}(na, 2)         # next guess of value function (parallelized version)

    for (s_index, s_val) in enumerate(s)         # loop through possible employment states
        s_prob = t_matrix[s_index, :]            # get transition probabilities for current state
        choice_lower = 1                         # for exploiting monotonicity of policy function

        for a_index = 1:na                       # loop through asset grid
            a = a_grid[a_index]                  # value of a
            candidate_max = -Inf                 # initialize lowest candidate max

            # loop over possible selections of a', exploiting monotonicity of policy function
            for ap_index in choice_lower:na
                c = s_val + a - q * a_grid[ap_index]                 # consumption given a' selection
                if c > 0                                             # check for positivity
                    utility = (c^(1-α) - 1)/(1 - α)                   # utility of consumption
                    val = utility + β * s_prob' * val_func[ap_index, :] # compute value

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

# Value function iteration
function V_iterate(prim::Primitives, res::Results, q::Float64, tol::Float64 = 1e-4, err::Float64 = 100.0)
    n = 0 # counter for iteration
    converged = 0 # indicator for convergence

    println("---------------------------------------------------------------")
    println("      Starting value function iteration for bond price ", q)
    println("---------------------------------------------------------------")
    while converged == 0  # keep iterating until we error less than tolerance value
        v_next = Bellman(prim, res, q)
        diff_v_e = maximum(abs.(v_next[1, :] - res.val_func[1, :]))
        diff_v_u = maximum(abs.(v_next[2, :] - res.val_func[2, :]))
        max_diff = maximum([diff_v_e, diff_v_u])

        if max_diff < tol
            converged = 1
        end

        if mod(n, 50) == 0
            println("Iteration = ", n)
            println("Max Difference = ", max_diff)
        end
        res.val_func = v_next
        n += 1

    end
    println("       Value function converged in ", n, " iterations.")
    println("---------------------------------------------------------------")
end

##############################################################################
