# Author: Katherine Kwok
# Date: November 17, 2021

# This file contains the code for Problem Set 7, where we estimate the AR(1)
# process using simulated method of moments (SMM). This helper function file
# contains the helper functions that implment the SMM algorithm as follows:
#
#   (1) Get data moments from the true data generating process
#   (2) Minimize objective function where weight matrix = indentity
#   (3) Compute SE for the estimated params above
#   (4) Minimize objective function where weight matrix = variance-cov matrix (Newey-West)
#   (5) Compute SE for the estimated params above
#
# The helper functions are roughly divided into the following categories:
#
#   (0) Declaring structs
#   (1) Getting moments (simulate AR(1) given parameters; both data and model)
#   (2) Minimizing objective function
#   (3) Computing SE
#   (4) Wrapper function for full SMM algorithm

# ------------------------------------------------------------------------ #
# (0) Declaring structs
# ------------------------------------------------------------------------ #

@with_kw struct Algorithm

    T::Int64 = 200      # length of AR(1) sequences
    H_true::Int64 = 1   # number of true AR(1) sequences
    H_model::Int64 = 10 # number of model AR(1) sequences

    # settings for number of moments
    mean_opt::Bool = true   # indicator for whether or not to get mean
    var_opt::Bool = true    # indicator for whether or not to get variance
    ac_opt::Bool = true     # indicator for whether or not to get first order auto correlation
    n_moms::Int64 = 3       # number of moments

    # paramters for true data generation
    ρ_0::Float64 = 0.5 # coefficient on term for previous period in AR(1)
    σ_0::Float64 = 1   # standard deviation for error term ε in AR(1)

    # paramteres for both true and model data generation
    μ_0::Float64 = 0   # mean for error term ε in AR(1)
    x_0::Float64 = 0   # initial term for AR(1)

    seed_0::Int64 = 12032020   # set seed for random draws
    lag_length::Int64 = 4      # lag length for Newey-West method
    nudge::Float64 = 1e-10     # "nudge" used to calculate numerical derivative for computing SE

end


# ------------------------------------------------------------------------ #
# (1) Getting moments (simulate AR(1) given parameters; both data and model)
# ------------------------------------------------------------------------ #

# simulate_data: This function simulates H different AR(1) sequences of length T
function simulate_data(ρ::Float64, σ::Float64, H::Int64, algo::Algorithm)
    @unpack μ_0, x_0, T, seed_0 = algo

    Random.seed!(seed_0)

    sim_data = zeros(H, T)        # intialize array to store true data
    sim_data[:, 1] = fill(x_0, H) # initialize the x_0 values

    for i_H in 1:H              # generate H different AR(1) sequences
        for i_T in 1:T-1        # each AR(1) sequence is length T (already initiated first value)

            error_draw = rand(Normal(μ_0, σ^2))                        # draw error term
            sim_data[i_H, i_T+1] = ρ * sim_data[i_H, i_T] + error_draw # get next period value
        end
    end
    sim_data
end

# get_mean: This function gets the means of different AR(1) sequences and returns
#           a vector of means.
function get_mean(input_data::Array{Float64, 2}, H::Int64)
    means = zeros(H) # initialize array to store output

    # calculate means across H sequences
    for i_H in 1:H
        means[i_H] = mean(input_data[i_H, :])
    end
    means # return
end

# get_variance: This function gets the variances of different AR(1) sequences
#               an returns a vector of variances.
function get_variances(input_data::Array{Float64, 2}, H::Int64)
    variances = zeros(H) # initialize array to store output

    # calculate variances across H sequences
    for i_H in 1:H
        variances[i_H] = var(input_data[i_H, :])
    end
    variances # return
end

# get_auto_corr: This function gets the first-order auto correlation of different
#                AR(1) sequences and returns a vector for auto correlation values.
function get_auto_corr(input_data::Array{Float64, 2}, algo::Algorithm, σ::Float64, H::Int64)
    @unpack T = algo

    k = 1                # degree of autocorrelation
    auto_corr = zeros(H) # initialize array to store output

    for i_H in 1:H                          # loop across H sequences
        m = mean(input_data[i_H, :])        # get mean for given sequence
        for i_T in 1:T-k                    # loop across values in sequence
            auto_corr[i_H] += (input_data[i_H, i_T] - m) * (input_data[i_H, i_T+k] - m)
        end
    end
    auto_corr = auto_corr ./((T-k) * σ^2)   # divide by observation number - k * variance
    auto_corr
end

# get_moments: This function gets relevant moments for the exercise. It can compute:
#              mean, variance, autocorrelation. NOTE: By default, this function
#              gets one moment (mean).
function get_moments(ρ::Float64, σ::Float64, H::Int64, algo::Algorithm; return_data::Bool = false)
    @unpack mean_opt, var_opt, ac_opt = algo

    # simulate H different AR(1) sequences of length T
    sim_data = simulate_data(ρ, σ, H, algo)

    # compute average moments across each H by specification

    # (1) get mean and variance only
    if mean_opt == true && var_opt == true && ac_opt == false
        output_moments = [mean(get_mean(sim_data, H)),mean(get_variances(sim_data, H))]

    # (2) get variance and auto correlation only
    elseif mean_opt == false && var_opt == true && ac_opt == true
        output_moments = [mean(get_variances(sim_data, H)),mean(get_auto_corr(sim_data, algo, σ, H))]

    # (3) get all moments
    elseif mean_opt == true && var_opt == true && ac_opt == true
        output_moments = [mean(get_mean(sim_data, H)),
                          mean(get_variances(sim_data, H)),
                          mean(get_auto_corr(sim_data, algo, σ, H))]
    end

    # output moments and/or data depending on request
    if return_data == true
        output_moments, sim_data
    else
        output_moments
    end
end

# ------------------------------------------------------------------------ #
# (2) Minimizing objective function
# ------------------------------------------------------------------------ #

# obj_func: This function defines the objective function to be minimized in the
#           SMM algorithm.
function obj_func(ρ::Float64, σ::Float64, algo::Algorithm, targ::Array{Float64}, W::Array{Float64, 2})

    @unpack H_model = algo

    # call get_moments to get model moments
    model_mom = get_moments(ρ, σ, H_model, algo)

    # define J (objective function)
    g = targ .- model_mom           # vector of distance between data and model moments
    J = g' * W * g                  # objective function

    return J
end

# newey_west: This function estimates the asymptotic variance-covariance matrix
#             using the Newey-West method.
function newey_west(algo::Algorithm, sim_mom::Array{Float64}, sim_data::Array{Float64, 2})
    @unpack lag_length, H_model, n_moms = algo

    # need to call the gamma function twice, once at 0 and once for lags
    gamma_0 = get_gamma(algo, sim_mom, sim_data, 0)

    # loop over lags to sum gamma_j for each lag period
    sum_gamma_j = zeros(n_moms, n_moms)
    for j = 1:lag_length
        gamma_j = get_gamma(algo, sim_mom, sim_data, j)
        sum_gamma_j += (1-(j/(lag_length + 1))) .* (gamma_j + gamma_j')
    end

    # calculate asymptotic variance-covariance matrix
    S = (1 + (1/H_model)) .* (gamma_0 + sum_gamma_j)

    return S
end

# get_obs_moment: This function gets the moments for a given data point. Note that
#                 it is different from the get_moments function, because the
#                 get_moments function is applied to a full AR(1) sequence
#                 rather than a single data point.
function get_obs_moment(sim_obs::Float64, sim_obs_tm1::Float64, sim_data::Array{Float64, 2},
    i_H::Int64, algo::Algorithm)

    @unpack mean_opt, var_opt, ac_opt = algo

    # get mean, variance, auto correlation for given observation
    sim_mean = mean(sim_data[i_H, :])
    sim_var = (sim_obs - sim_mean)^2
    sim_ac = (sim_obs - sim_mean) * (sim_obs_tm1 - sim_mean)

    # select appropriate moments depending on user input
    if mean_opt == true && var_opt == true && ac_opt == false     # get mean and variance only
        obs_mom = [sim_mean, sim_var]
    elseif mean_opt == false && var_opt == true && ac_opt == true # get variance and auto correlation only
        obs_mom = [sim_var, sim_ac]
    elseif mean_opt == true && var_opt == true && ac_opt == true  # get all moments
        obs_mom = [sim_mean, sim_var, sim_ac]
    end

    obs_mom
end


# get_gamma: This function computes gamma, which is used for calculating the
#            asymptotic variance-covariance matrix. See handout for formula.
function get_gamma(algo::Algorithm, sim_mom::Array{Float64}, sim_data::Array{Float64, 2}, lag::Int64)
    @unpack H_model, T, n_moms, mean_opt, var_opt, ac_opt = algo

    gamma_tot = zeros(n_moms, n_moms)

    # calculate gamma for a given number of lagged periods (between 0 to lag length)
    for i_T = (1+lag):T
        for i_H = 1:H_model

            # (1) get data point in sequence
            sim_obs = sim_data[i_H, i_T]

            if i_T == 1    # if at first time period, set previous observation to 0
                sim_obs_tm1 = 0.0
            else           # else, get the previous observation
                sim_obs_tm1 = sim_data[i_H, i_T-1]
            end

            # get moments for given data point then compare to simulated moments
            obs_mom = get_obs_moment(sim_obs, sim_obs_tm1, sim_data, i_H, algo)
            mom_obs_diff = obs_mom .- sim_mom

            # (2) get lagged data point
            sim_lag = sim_data[i_H, i_T-lag]

            if i_T - lag == 1 # if i_T - lag is first time period, set previous observation to 0
                sim_lag_tm1 = 0.0
            else              # else, get the previous observation
                sim_lag_tm1 = sim_data[i_H, i_T-lag-1]
            end

            # get moments for lagged data point then compare to simulated moments
            lag_mom = get_obs_moment(sim_lag, sim_lag_tm1, sim_data, i_H, algo)
            mom_lag_diff = lag_mom .- sim_mom

            # (3) multiply obs diff with lagged diff and add to gamma
            gamma_tot .+= mom_obs_diff * mom_lag_diff'
        end
    end
    # multiply summed gamma with 1/(T * H)
    gamma = (1/(T * H_model)) .* gamma_tot
    return gamma
end


# ------------------------------------------------------------------------ #
# (3) Computing SE
# ------------------------------------------------------------------------ #

# compute_se: This function computes the standard errors for the b_2 coefficients
#             using numerical derivatives.
function compute_se(algo::Algorithm, ρ::Float64, σ::Float64, S::Array{Float64, 2})
    @unpack nudge, H_model, T, n_moms = algo

    # call get moments at ρ, σ, ρ - nudge and σ - nudge
    m = get_moments(ρ, σ, H_model, algo)
    m_ρ_nudge = get_moments(ρ - nudge, σ, H_model, algo)
    m_σ_nudge = get_moments(ρ, σ - nudge, H_model, algo)

    # calculate numerical derivative using get_moments results
    ρ_derivative = (m_ρ_nudge .- m)/nudge
    σ_derivative = (m_σ_nudge .- m)/nudge

    # combine derivatives
    output_deriv = [ρ_derivative σ_derivative]

    # variance-covariance matrix for ρ and σ
    var_cov = (1/T) * inv(output_deriv' * inv(S) * output_deriv)

    # get standard errors
    se = [sqrt(Diagonal(var_cov))[1, 1], sqrt(Diagonal(var_cov))[2, 2]]

    # return jacobian, variance covariance matrix, se
    output_deriv, var_cov, se

end

# ------------------------------------------------------------------------ #
# (4) Wrapper function for SMM algorithm and output formatting
# ------------------------------------------------------------------------ #

# solve_smm: This is a wrapper function that implements the SMM algorithm using
#            helper functions defined above. NOTE: By default, this function
#            does not do bootstrapping.
function solve_smm(mean_opt_i::Bool, var_opt_i::Bool, ac_opt_i::Bool, n_moms_i::Int64;
    bootstrap_opt::Bool = false)

    # initialize algorithm settings: if bootstrapping, then draw a random seed
    # if not, we will use the same seed (defined in Algorithm struct)
    if bootstrap_opt == true
        rand_seed = rand(1000000:2000000)
        algo = Algorithm(mean_opt = mean_opt_i, var_opt = var_opt_i, ac_opt = ac_opt_i, n_moms = n_moms_i, seed_0 = rand_seed)
    else
        algo = Algorithm(mean_opt = mean_opt_i, var_opt = var_opt_i, ac_opt = ac_opt_i, n_moms = n_moms_i)
    end
    b_init = [algo.ρ_0, algo.σ_0] # initial parameter values

    # get_moments for true data
    @unpack H_true, H_model, T, n_moms = algo
    targ = get_moments(algo.ρ_0, algo.σ_0, H_true, algo)

    # minimize objective function with W = I
    W = Matrix{Float64}(I, n_moms, n_moms)
    b_1 = optimize(b -> obj_func(b[1], b[2], algo, targ, W), b_init).minimizer

    # compute SE of parameters that minimize above
    b_1_mom, b_1_data = get_moments(b_1[1], b_1[2], H_model, algo; return_data = true)
    S = newey_west(algo, b_1_mom, b_1_data)
    jacob_1, var_cov_1, se_1 = compute_se(algo, b_1[1], b_1[2], S)

    # minimize objective function with W = S^-1
    W_opt = inv(S)
    b_2 = optimize(b -> obj_func(b[1], b[2], algo, targ, W_opt), b_init).minimizer

    # compute SE of parameters that minimize above
    b_2_mom, b_2_data = get_moments(b_2[1], b_2[2], H_model, algo; return_data = true)
    S = newey_west(algo, b_2_mom, b_2_data)
    jacob_2, var_cov_2, se_2 = compute_se(algo, b_2[1], b_2[2], S)

    # conduct J-test
    J_b_2 = obj_func(b_2[1], b_2[2], algo, targ, W)
    chi_square = T * (H_model/(1+H_model)) * J_b_2

    # print summary if not bootstrapping
    if bootstrap_opt == false
        print_summary(algo, targ, b_1, b_2, jacob_1, jacob_2, var_cov_1, var_cov_2, se_1, se_2, chi_square)
    end

    # return results
    [algo, targ, b_1, b_2, jacob_1, jacob_2, var_cov_1, var_cov_2, se_1, se_2, chi_square]

end

# bootstrap_smm: This function implements a bootstrapped version of the smm algo
function bootstrap_smm(mean_opt_i::Bool, var_opt_i::Bool, ac_opt_i::Bool, n_moms_i::Int64;
    n_bootstrap::Int64 = 10)

    # initialize output from bootstrapping with first iteration
    smm_result = solve_smm(mean_opt_i, var_opt_i, ac_opt_i, n_moms_i; bootstrap_opt = true)
    bootstrap_output = smm_result[3:size(smm_result, 1)]

    # call smm solver function over number of iterations
    for i_boot in 2:n_bootstrap

        # call smm solver with option to sample a random seed
        smm_result = solve_smm(mean_opt_i, var_opt_i, ac_opt_i, n_moms_i; bootstrap_opt = true)

        # add results together
        for res_i in 1:size(bootstrap_output, 1) - 1
            bootstrap_output[res_i] .+= smm_result[res_i + 2]
        end
        bootstrap_output[9] += smm_result[11]

    end

    # get average across bootstraps
    for res_i in 1:size(bootstrap_output, 1)
        bootstrap_output[res_i] = bootstrap_output[res_i]/n_bootstrap
    end

    # summary results
    print_summary(smm_result[1], smm_result[2], bootstrap_output[1], bootstrap_output[2],
                  bootstrap_output[3], bootstrap_output[4], bootstrap_output[5],
                  bootstrap_output[6], bootstrap_output[7], bootstrap_output[8],
                  bootstrap_output[9])
end


# print_summary: This function prints a summary of the results
function print_summary(algo, targ, b_1, b_2, jacob_1, jacob_2, var_cov_1, var_cov_2, se_1, se_2, chi_square)
    @unpack mean_opt, var_opt, ac_opt = algo

    # select correct summary of option
    if mean_opt == true && var_opt == true && ac_opt == false     # get mean and variance only
        moments = "mean and variance"
    elseif mean_opt == false && var_opt == true && ac_opt == true # get variance and auto correlation only
        moments = "variance and first order autocorrelation"
    elseif mean_opt == true && var_opt == true && ac_opt == true  # get all moments
        moments = "mean, variance and first order autocorrelation with bootstrapping"
    end

    # print σ^2 rather than just σ
    targ[2] = targ[2]^2
    b_1[2] = b_1[2]^2
    b_2[2] = b_2[2]^2

    @printf "+------------------------------------------------------------+\n"
    @printf " Computed moments: %s\n" moments
    @printf "+------------------------------------------------------------+\n"
    println("              target = ", round.(targ, digits = 4))
    println("                 b_1 = ", round.(b_1, digits = 4))
    println("                 b_2 = ", round.(b_2, digits = 4))
    println("    jacobian for b_1 = ", round.(jacob_1, digits = 4))
    println("     var-cov for b_1 = ", round.(var_cov_1, digits = 4))
    println(" std. errors for b_1 = ", round.(se_1, digits = 4))
    println("    jacobian for b_2 = ", round.(jacob_2, digits = 4))
    println("     var-cov for b_2 = ", round.(var_cov_2, digits = 4))
    println(" std. errors for b_2 = ", round.(se_2, digits = 4))
    println(" chi_square (J-test) = ", round.(chi_square, digits = 10))
    @printf "+------------------------------------------------------------+\n"

end
