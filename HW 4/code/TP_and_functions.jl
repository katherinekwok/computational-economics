# Author: Katherine Kwok
# Date: October 13, 2021

# This file contains the code for Problem Set 4, where we solve for transition
# paths from eliminating social security, using the Conesa-Krueger Model.
#
# The code below computes the TRANSITIONS PATHS, and is divided into the
# following sections:
#
#   (0) set up strucs and functions to initialize
#   (1) functions for shooting backward
#   (2) functions for shooting forward
#   (3) functions for checking for convergence

include("models_and_functions_SS.jl") # use functions that find steady state solution
                                      # if function referenced in this file is not defined here
                                      # then it is in the SS file

##
# ---------------------------------------------------------------------------- #
#  (0) Set up strucs and functions to initialize
# ---------------------------------------------------------------------------- #

# Results_TP: This struct holds the variables for transition path
@everywhere mutable struct TransitionPaths
    TPs::Int64           # number of transition periods
    K_TP::Array{Float64, 1}     # transition path of K
    L_TP::Array{Float64, 1}     # transition path of L
    θ_TP::Array{Float64, 1}     # transition path of θ
    r_TP::Array{Float64, 1}     # transition of r
    w_TP::Array{Float64, 1}     # transition of w
    b_TP::Array{Float64, 1}     # transition of b

    val_func_TP::Array{Float64, 3} # value function transition
    pol_func_TP::Array{Float64, 3} # policy function transition (asset)
    lab_func_TP::Array{Float64, 3} # labor function transition
    ψ_TP::Array{Float64, 3}        # cross-sec distribution transition

end

# initialize_TP: This function solves for the steady states, then initializes the
#                transition path variables (stored in the Results_TP struc).
function initialize_TP()
    p0, r0, w0, cv0 = solve_model()           # solve θ = 0.11 model - with soc security
    pT, rT, wT, cvT = solve_model(θ_0 = 0.0)  # solve θ = 0 model - no soc security

    TPs = 30 # number of transition periods

    @unpack α, δ, μ_r, na, nz, N, age_retire = p0 # unpack prims from θ = 0.11 model (same for θ = 0)
    θ_TP = repeat([pT.θ], TPs) # θ along the transition path

    K_TP = collect(range(p0.K_0, step = ((pT.K_0 - p0.K_0)/TPs), stop = pT.K_0))[2:TPs+1] # transition path of K
    L_TP = collect(range(p0.L_0, step = ((pT.L_0 - p0.L_0)/TPs), stop = pT.L_0))[2:TPs+1] # transition path of L
    r_TP = F_1.(α, K_TP, L_TP)     # transition path of r using K_TP, L_TP
    w_TP = F_2.(α,δ, K_TP, L_TP)   # transition path of w using K_TP, L_TP
    b_TP = calculate_b.(θ_TP, w_TP, L_TP, μ_r)

    col_num_all = (N - age_retire + 1) + (age_retire - 1)*nz # number of indices corresponding to age + state for everyone (carry-over from pset 3 set up)
    col_num_worker = (age_retire - 1)*nz                     # number of indices corresponding to age + state for workers  (carry-over from pset 3 set up)
    val_func_TP = zeros(TPs, na, col_num_all)          # initial value function for all periods
    pol_func_TP = zeros(TPs, na, col_num_all)          # initial policy function for all periods
    lab_func_TP = zeros(TPs, na, col_num_worker)       # initial labor function for all periods
    ψ_TP = zeros(TPs, na * nz, N)                      # initial cross-sec distribution for all periods

    tp = TransitionPaths(TPs, K_TP, L_TP, θ_TP, r_TP, w_TP, b_TP, val_func_TP,
    pol_func_TP, lab_func_TP, ψ_TP) # transition path of b

    tp, p0, r0, pT, rT # return initialized struc
end

##
# ---------------------------------------------------------------------------- #
#  (1) Shoot backward: Solve HH dynamic programming problem along TP backwards
# ---------------------------------------------------------------------------- #

function shoot_backward(pt::Primitives, tp::TransitionPaths, r0::Results, rT::Results)
    @unpack TPs = tp # unpack toilet paper :-) (i.e. transition path)

    tp.val_func_TP[TPs, :, :] = rT.val_func     # store the initialize steady state into first period (t=1)
    tp.pol_func_TP[TPs, :, :] = rT.pol_func
    tp.lab_func_TP[TPs, :, :] = rT.lab_func
    rt_next = rT                              # initialize next results (contains value, pol, lab fun)

    for t in TPs-1:-1:1 # iterate from T back to 0

        rt = initialize_results(pt) # initialize results

        # set up primitives for this backward induction of household dynamic programming problem
        # accessing using index + 1 because julia does not do 0-based indexing
        pt.θ = tp.θ_TP[t]     # θ
        pt.r = tp.r_TP[t]     # interest rate
        pt.w = tp.w_TP[t]     # wage
        pt.b = tp.b_TP[t]     # social security benefit
        pt.K_0 = tp.K_TP[t]   # aggregate capital
        pt.L_0 = tp.L_TP[t]   # aggregate labor supply

        v_backward_iterate(pt, rt; steady_state = false, res_next_input = rt_next) # backward iterate

        tp.val_func_TP[t, :, :] = rt.val_func # store the val, pol, lab funcs into transition path
        tp.pol_func_TP[t, :, :] = rt.pol_func
        tp.lab_func_TP[t, :, :] = rt.lab_func

        rt_next = rt # update next results (next period, next age) to use in next iteration of for loop

        if t % 2 == 0 # give status update at every 2 transition periods
            println("-----------------------------------------------------------------------")
            @printf "                Shooting backward; at period %d now \n" t
            println("-----------------------------------------------------------------------")
        end
    end
end

##
# ---------------------------------------------------------------------------- #
#  (2) Shoot forward: Solve cross-sec distribution and new TP forwards
# ---------------------------------------------------------------------------- #

# solve_ψ_TP: This function solves the stationary distribution for each age and
# time period. First, we make the transition matrix that gives us the probability
# of an individual in a given age and time period to be an a particular asset state
# in the next age and time period. Then, we apply the transition matrix to the
# distribution for each time and age period, to get the distribution for
# the next time and age period.
function solve_ψ_TP(t::Int64, pt::Primitives, tp::TransitionPaths)
    @unpack N, n, na, z_initial_prob, nz, μ = pt # unpack paramters

    # distribution for model age = 1
    tp.ψ_TP[t, 1, 1] = pt.z_initial_prob[1]  * μ[1]    # distribution of high prod people (at birth)
    tp.ψ_TP[t, na+1, 1] = pt.z_initial_prob[2] * μ[1]  # distribution of low prod people (at birth)

    for age in 1:N-1 # loop through ages

        # make transition matrix for given period and age
        trans_mat = make_trans_matrix(pt, tp.pol_func_TP[t, :, :], age)
        # use transition matrix and current distribution for next period and age distribution
        tp.ψ_TP[t+1, :, age+1] = trans_mat' * tp.ψ_TP[t, :, age] * (1/(1+n))

    end

end


# shoot_forward: This function shoots forward from 0 to T, in order to find
# the cross-sec distribution from one age and time period to the next age and
# time period. Then, we can calculate the new transition path based on the
# the policy functions and cross-sec distributions.
function shoot_forward(pt::Primitives, tp::TransitionPaths, r0::Results, K_TP_1::Array{Float64, 1}, L_TP_1::Array{Float64, 1})
    @unpack TPs = tp # unpack toilet paper :-) (i.e. transition path)

    tp.ψ_TP[1, :, :] = r0.ψ                    # fill first period with the initial steady state dist

    for t in 1:TPs-1                           # iterate forward from 0 to T-1
        solve_ψ_TP(t, pt, tp)                  # solve for distribution by age and time period
        update_path(t, pt, tp, K_TP_1, L_TP_1) # update aggregate K and L

        if t % 2 == 0 # give status update at every 2 transition periods
            println("-----------------------------------------------------------------------")
            @printf " Shooting forward; t=%d, K_TP_1 = %.4f, L_TP_1 = %.4f, K_TP_0 = %.4f, L_TP_0 = %.4f\n" t+1 K_TP_1[t+1] L_TP_1[t+1] tp.K_TP[t+1] tp.L_TP[t+1]
            println("-----------------------------------------------------------------------")
        end
    end
end

##
# ---------------------------------------------------------------------------- #
#  (3) Check for convergence
# ---------------------------------------------------------------------------- #

# update_path: This function updates the transition path, by calculating the
# aggregate capital (K) and labor supply (K)
function update_path(t::Int64, pt::Primitives, tp::TransitionPaths, K_TP_1::Array{Float64, 1}, L_TP_1::Array{Float64, 1})
    @unpack TPs = tp

    # calculate aggregate capital and labor supply in next time period
    K_next, L_next = calc_aggregate(pt, tp.pol_func_TP[t+1, :, :], tp.ψ_TP[t+1, :, :], tp.lab_func_TP[t+1, :, :])
    # update the new transition paths
    K_TP_1[t+1] = K_next
    L_TP_1[t+1] = L_next

end


# display_progress: This function plots the new and old transition paths for
# troubleshooting
function display_progress(tp::TransitionPaths, K_TP_1::Array{Float64}, L_TP_1::Array{Float64},
    p0::Primitives, pT::Primitives)
    @unpack TPs = tp

    display(plot([tp.K_TP K_TP_1 repeat([p0.K_0], TPs) repeat([pT.K_0], TPs)],
            label = ["Old TP" "New TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Capital", legend = :topleft), )

    display(plot([tp.L_TP L_TP_1 repeat([p0.L_0], TPs) repeat([pT.L_0], TPs)],
            label = ["Old TP" "New TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Labor", legend = :topleft))
end


# check_convergence_TP: This function checks if the maximum, absolute difference
# between K and L transition paths combined is less than the tolerance value.
# If so, we have converged, and if not, we update the transition path and repeat.
function check_convergence_TP(pt::Primitives, tp::TransitionPaths, K_TP_1::Array{Float64}, L_TP_1::Array{Float64};
    tol::Float64 = 1.0e-3, λ::Float64 = 0.5)
    @unpack α, δ, μ_r = pt
    @unpack TPs, K_TP, L_TP, θ_TP = tp

    K_diffs = abs.(K_TP_1 .- K_TP) # calculate differences between two K transition paths
    L_diffs = abs.(L_TP_1 .- L_TP) # calculate differences between two L transition paths
    max_diff = maximum(K_diffs .+ L_diffs)               # get max diff of K and L combined

    if max_diff > tol # if max diff is above tolerance, we update

        tp.K_TP = λ .* K_TP_1 .+ (1-λ) .* K_TP # adjust using λ paramter
        tp.L_TP = λ .* L_TP_1 .+ (1-λ) .* L_TP

        tp.r_TP = F_1.(α, tp.K_TP, tp.L_TP)                  # transition path of r
        tp.w_TP = F_2.(α, δ, tp.K_TP, tp.L_TP)               # transition path of w
        tp.b_TP = calculate_b.(θ_TP, tp.w_TP, tp.L_TP, μ_r)  # transition path of θ

        converged = 0 # update convergence flag
    else
        converged = 1
    end
    converged # return convergence flag
end

function check_convergence_outer()
end
