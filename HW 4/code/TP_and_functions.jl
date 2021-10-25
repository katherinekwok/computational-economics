# Author: Katherine Kwok
# Date: October 23, 2021

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
#   (4) functions for running the main algorithm and producing results + plots

include("model_and_functions.jl") # use functions and structs that solve household
                                  # dynamic programming problem and stationary
                                  # distribution

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
#
# NOTE: By default, this function initializes a transtion path for unanticipated
#       social security shock.
#

function initialize_TP(p0::Primitives, pT::Primitives, TPs::Int64, date_implemented::Int64)

    println("-----------------------------------------------------------------------")
    @printf "          Starting Outer While Loop with %d transition periods \n" TPs
    println("-----------------------------------------------------------------------")

    @unpack α, δ, μ_r, na, nz, N, age_retire = p0 # unpack prims from θ = 0.11 model

    θ_TP = vcat(repeat([p0.θ], date_implemented - 1), repeat([pT.θ], TPs - date_implemented + 1)) # transition path of θ

    K_TP = collect(range(p0.K_0, step = ((pT.K_0 - p0.K_0)/(TPs-1)), stop = pT.K_0)) # transition path of K
    L_TP = collect(range(p0.L_0, step = ((pT.L_0 - p0.L_0)/(TPs-1)), stop = pT.L_0)) # transition path of L

    # NOTE: the first aggregate K, L is the initial SS K, L with θ = 0.11, because
    #       at t = 1, households are still in old SS asset allocations.

    w_TP = F_1.(α, K_TP, L_TP)                  # transition path of w using K_TP, L_TP
    r_TP = F_2.(α, δ, K_TP, L_TP)               # transition path of r using K_TP, L_TP
    b_TP = calculate_b.(θ_TP, w_TP, L_TP, μ_r)  # transition path of b

    col_num_all = (N - age_retire + 1) + (age_retire - 1)*nz # number of indices corresponding to age + state for everyone (carry-over from pset 3 set up)
    col_num_worker = (age_retire - 1)*nz                     # number of indices corresponding to age + state for workers  (carry-over from pset 3 set up)

    val_func_TP = zeros(TPs, na, col_num_all)                # initial value function for all periods
    pol_func_TP = zeros(TPs, na, col_num_all)                # initial policy function for all periods
    lab_func_TP = zeros(TPs, na, col_num_worker)             # initial labor function for all periods
    ψ_TP = zeros(TPs, na * nz, N)                            # initial cross-sec distribution for all periods

    tp = TransitionPaths(TPs, K_TP, L_TP, θ_TP, r_TP, w_TP, b_TP, val_func_TP, pol_func_TP, lab_func_TP, ψ_TP)
    tp # return initialized struc
end

##
# ---------------------------------------------------------------------------- #
#  (1) Shoot backward: Solve HH dynamic programming problem along TP backwards
# ---------------------------------------------------------------------------- #

# shoot_backward: This function shoots backward along the transition path to
# solve the household dynamic programming problem, given the aggregate K,L, and
# the corresponding prices (r, w) and benefit level (b).
#
# NOTE: The pol, val, lab functions for the last period in transition is always
#       the same as that of the steady state in which θ = 0 (no social security).

function shoot_backward(pt::Primitives, tp::TransitionPaths, r0::Results, rT::Results)
    @unpack TPs = tp # unpack toilet paper :-) (i.e. transition path)

    tp.val_func_TP[TPs, :, :] = rT.val_func     # at t = T, the val/pol/lab func is the same at SS with θ = 0
    tp.pol_func_TP[TPs, :, :] = rT.pol_func
    tp.lab_func_TP[TPs, :, :] = rT.lab_func
    rt_next = rT                                # initialize next results (contains value, pol, lab fun)

    for t in TPs-1:-1:1 # iterate from T-1 back to 1

        # set up primitives for this backward induction of household dynamic programming problem
        pt.θ = tp.θ_TP[t]     # θ
        pt.r = tp.r_TP[t]     # interest rate
        pt.w = tp.w_TP[t]     # wage
        pt.b = tp.b_TP[t]     # social security benefit
        pt.K_0 = tp.K_TP[t]   # aggregate capital
        pt.L_0 = tp.L_TP[t]   # aggregate labor supply

        rt = initialize_results(pt) # initialize results

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
    tp.ψ_TP[t+1, 1, 1] = pt.z_initial_prob[1]  * μ[1]    # distribution of high prod people (at birth)
    tp.ψ_TP[t+1, na+1, 1] = pt.z_initial_prob[2] * μ[1]  # distribution of low prod people (at birth)

    for age in 1:N-1 # loop through ages

        # make transition matrix for given period and age
        trans_mat = make_trans_matrix(pt, tp.pol_func_TP[t, :, :], age)
        # use transition matrix and current distribution for next period and age distribution
        tp.ψ_TP[t+1, :, age+1] = trans_mat' * tp.ψ_TP[t, :, age] * (1/(1+n))
    end
end

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


# shoot_forward: This function shoots forward across the transition, in order to find
# the cross-sec distribution from one age and time period to the next age and
# time period. Then, we can calculate the new transition path based on the
# the policy functions and cross-sec distributions.
#
# NOTE: The t = 0 period distribution is always the same as that of the steady
#       state in which θ = 0.11 (when there is social security).

function shoot_forward(pt::Primitives, tp::TransitionPaths, r0::Results, K_TP_1::Array{Float64, 1}, L_TP_1::Array{Float64, 1})
    @unpack TPs = tp # unpack toilet paper :-) (i.e. transition path)

    tp.ψ_TP[1, :, :] = r0.ψ                    # at t = 1, the distribution is the same as SS with θ = 0.11
    update_path(0, pt, tp, K_TP_1, L_TP_1)     # update aggregate K and L for t = 1

    for t in 1:TPs-1                           # iterate forward from 1 to T-1
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


# display_progress: This function plots the new and old transition paths for
#                   troubleshooting. If save = true, we save the plot for
#                   record-keeping.
function display_progress(tp::TransitionPaths, K_TP_1::Array{Float64}, L_TP_1::Array{Float64},
    p0::Primitives, pT::Primitives, experiment::String; save::Bool = false)
    @unpack TPs = tp

    # plot aggregate capital
    K_plot = plot([tp.K_TP K_TP_1 repeat([p0.K_0], TPs) repeat([pT.K_0], TPs)],
            label = ["Old TP" "New TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Aggregate Capital Transition Path", legend = :bottomright,
            ylims = (3.25, 4.75))
    display(K_plot)

    # plot aggregate labor
    L_plot = plot([tp.L_TP L_TP_1 repeat([p0.L_0], TPs) repeat([pT.L_0], TPs)],
            label = ["Old TP" "New TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Aggregate Labor Transition Path", legend = :bottomright,
            ylims = (0.34, 0.38))
    display(L_plot)

    if save == true
        savefig(K_plot, "output/K_TP_"*experiment*"_"*string(TPs)*".png")
        savefig(L_plot, "output/L_TP_"*experiment*"_"*string(TPs)*".png")
    end
end


# check_convergence_TP: This function checks if the maximum, absolute difference
# between K and L transition paths combined is less than the tolerance value.
# If so, we have converged, and if not, we update the transition path and repeat.
# This is for the inner loop.

function check_convergence_TP(iter::Int64, pt::Primitives, tp::TransitionPaths,
    K_TP_1::Array{Float64}, L_TP_1::Array{Float64}, experiment::String;
    tol::Float64 = 1.0e-3, λ::Float64 = 0.5)

    @unpack α, δ, μ_r = pt
    @unpack TPs, K_TP, L_TP, θ_TP = tp

    K_diffs = abs.(K_TP_1 .- K_TP) # calculate differences between two K transition paths
    L_diffs = abs.(L_TP_1 .- L_TP) # calculate differences between two L transition paths
    max_diff = maximum(K_diffs .+ L_diffs)               # get max diff of K and L combined

    if max_diff > tol # if max diff is above tolerance, we update

        display_progress(tp, K_TP_1, L_TP_1, p0, pT, experiment) # plot current transition path

        tp.K_TP = λ .* K_TP_1 .+ (1-λ) .* K_TP # adjust using λ paramter
        tp.L_TP = λ .* L_TP_1 .+ (1-λ) .* L_TP

        converged = 0    # convergence flag still 0

        println("-----------------------------------------------------------------------")
        @printf "       Completed %d iterations of inner while loop; continuing...\n" iter
        println("-----------------------------------------------------------------------")

    else

        display_progress(tp, K_TP_1, L_TP_1, p0, pT, experiment; save = true) # plot!

        tp.K_TP = K_TP_1 # store the converged transition paths
        tp.L_TP = L_TP_1

        converged = 1    # update convergence flag!

        println("-----------------------------------------------------------------------")
        @printf "           Inner while loop converged after %d iteration\n" iter
        println("-----------------------------------------------------------------------")
    end

    # update prices and benefit level
    tp.w_TP = F_1.(α, tp.K_TP, tp.L_TP)                  # transition path of w
    tp.r_TP = F_2.(α, δ, tp.K_TP, tp.L_TP)               # transition path of r
    tp.b_TP = calculate_b.(θ_TP, tp.w_TP, tp.L_TP, μ_r)  # transition path of b

    converged # return convergence flag
end

# check_convergence_SS: This function checks if the aggregate K and L at the end
# of the transition periods are close enough to the steady state without social
# security. If not, we lengthen the transition periods. This is for the outer loop.

function check_convergence_SS(pT::Primitives, tp::TransitionPaths, TPs::Int64, iter::Int64; tol::Float64 = 1.0e-3)

    # calculate diff between K and L in T (last period of transition) and steady state
    diff = abs(tp.K_TP[TPs] - pT.K_0) + abs(tp.L_TP[TPs] - pT.L_0)

    if diff > tol                 # if difference greater than tolerance
        update_TPs = TPs + 20     # lengthen transition period by 20
        converged = 0
    else
        converged = 1      # update convergence flag!
        println("-----------------------------------------------------------------------")
        @printf "           Outer while loop converged after %d iteration\n" iter
        println("-----------------------------------------------------------------------")
    end

    converged, update_TPs # return convergence flag and updated transition period
end

##
# ---------------------------------------------------------------------------- #
#   (4) functions for running the main algorithm and producing results + plots
# ---------------------------------------------------------------------------- #

# solve_algorithm: This function runs the outer and inner while loops that solves
#                  the Conesa-Krueger model along the transition paths.
#
#   (1) The inner loop takes a given transition period length (TPs), and shoots
#       backward and forward to solve the HH problem and stationary distribution
#       until the transition paths converged.
#
#   (2) The outer loop checks if the aggregate K, L last period of the transition
#       is close enough to the ending steady state (no social security). If it's
#       not close enough, the transition period is lengthened.

function solve_algorithm(experiment::String, TPs::Int64, p0::Primitives, pT::Primitives,
    r0::Results, rT::Results; date_imple_input::Int64 = 1)

    converged_outer = 0          # convergence flag for outer while loop
    iter_outer = 1               # iteration counter for outer while loop

    while converged_outer == 0   # outer loop

        tp = initialize_TP(p0, pT, TPs, date_imple_input) # initialize transition path variables
        pt = initialize_prims()  # initialize mutatable struc primitives for current period (t)

        K_TP_1 = zeros(tp.TPs)   # initialize arrays for new transition path
        L_TP_1 = zeros(tp.TPs)
        converged_inner = 0      # convergence flag for inner while loop
        iter_inner = 1           # iteration counter for inner while loop

        while converged_inner == 0 # inner loop

            @time shoot_backward(pt, tp, r0, rT) # shoot backwards to solve dynamic household problem
            @time shoot_forward(pt, tp, r0, K_TP_1, L_TP_1) # shoot forwards to solve cross-sec distribution

            converged_inner = check_convergence_TP(iter_inner, pt, tp, K_TP_1, L_TP_1, experiment) # check progress and convergence
            iter_inner += 1   # update iteration counter
        end

        converged_outer, update_TPs = check_convergence_SS(pT, tp, TPs, iter_outer) # check outer convergence, update
        TPs = update_TPs # update number of transition periods
        iter_outer += 1
    end

    tp, pt # return converged transition paths and other stuff
end

# calculate_EV: This function calculates the consumption equivalence of the
#               policy change relative to the steady state with social security.

function calculate_EV(pt::Primitives, tp::TransitionPaths, r0::Results)
    @unpack nz, na, N, γ, σ = pt
    EV = zeros(nz, na, N)        # initialize EV array by age, states, asset levels
    EV_age = zeros(age)          # initialize EV array by age
    voters_in_favor = zeros(age) # initialize array of voters in favor of policy change by age

    for age in 1:N               # for each age
        for z_index in 1:nz         # for each state
            for a_indez in 1:na         # for each asset level

                if age >= age_retire
                    val_index = retiree_val_index(age_retire, nz, age)  # get asset index for retireee
                else
                    val_index = worker_val_index(z_index, age, nz)      # get asset index for worker
                end
                ψ_index = a_index + na*(z_index - 1)                    # get index for stationary distribution


                v0_TP = tp.val_func_tp[1, a_index, val_index]   # numerator (use t = 1 b/c household still at old asset level)
                v0_SS = r0.val_func[a_index, val_index]         # denominator

                EV[z_indez, a_indez, age] = (v0_TP/v0_SS)^(1/(γ*(1-σ)))         # EV formula
                EV[age] += EV[z_index, a_index, age] * ψ_index[ψ_index, age]    # EV by age (weighted by distribution)

                # assume voters will be willing to support policy if EV >= 0 (weighted by distribution)
                voters_in_favor += (EV[z_index, a_index, age] >= 0) * ψ_index[ψ_index, age]
            end
        end
    end
    EV, EV_age, voters_in_favor # return completed calculations
end

# summarize_results: This function takes the output from running the main algorithm
#                    and produces plots and the consumption equivalent variations.

function summarize_results(experiment::String, tp::TransitionPaths, pt::Primitives, r0::Results)

    EV, EV_age, voters_in_favor = calculate_EV(pt, tp, r0) # calculate consumption equivalence variation objects

    # display share of voters supporting policy (sum across ages)
    println("-----------------------------------------------------------------------")
    @printf "            The share of voters supporting policy is %d\n" sum(voters_in_favor)
    println("-----------------------------------------------------------------------")

    # make plots for K, L, r, w by transition period

    K_plot = plot([tp.K_TP repeat([p0.K_0], TPs) repeat([pT.K_0], TPs)],
            label = ["K TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Aggregate Capital Transition Path", legend = :bottomright,
            ylims = (3.25, 4.75), xlabel = "Time")
    savefig(K_plot, "output/K_TP_"*experiment*"_"*string(TPs)*".png")

    L_plot = plot([tp.L_TP repeat([p0.L_0], TPs) repeat([pT.L_0], TPs)],
            label = ["L TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Aggregate Labor Transition Path", legend = :bottomright,
            ylims = (0.34, 0.38), xlabel = "Time")
    savefig(L_plot, "output/L_TP_"*experiment*"_"*string(TPs)*".png")

    r_plot = plot([tp.r_TP*100 repeat([p0.r*100], TPs) repeat([pT.r*100], TPs)],
            label = ["r TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Interest Rate Transition Path (%)", legend = :topright,
            ylims = (1, 3), xlabel = "Time")
    savefig(r_plot, "output/r_TP_"*experiment*"_"*string(TPs)*".png")

    w_plot = plot([tp.w_TP repeat([p0.w], TPs) repeat([pT.w], TPs)],
            label = ["w TP" "SS w/ θ > 0" "SS w/ θ = 0"],
            title = "Wage Transition Path", legend = :bottomright,
            ylims = (1.4, 1.6), xlabel = "Time")
    savefig(w_plot, "output/w_TP_"*experiment*"_"*string(TPs)*".png")

    # make plots for EV_age and voters in favor of policy

    EV_plot = plot([EV_age], title = "Consumption Equivalence",
              ylims = (0.1, 0.7), xlabel = "Agent Age")
    savefig(EV_plot, "output/EV_TP_"*experiment*"_"*string(TPs)*".png")

    voters_plot = plot([voters_in_favor], title = "Fraction of Voters Supporting Policy",
                  ylims = (0, 1), xlabel = "Agent Age")
    savefig(w_plot, "output/voters_TP_"*experiment*"_"*string(TPs)*".png")

end
