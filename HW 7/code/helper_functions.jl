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

    # paramters for true data generation
    ρ_0::Float64 = 0.5 # coefficient on term for previous period in AR(1)
    σ_0::Float64 = 1   # standard deviation for error term ε in AR(1)

    # paramteres for both true and model data generation
    μ_0::Float64 = 0   # mean for error term ε in AR(1)
    x_0::Float64 = 0   # initial term for AR(1)

    seed_0::Int64 = 1234  # set seed for random draws

    lag_length::Int64 = 4 # lag length for Newey-West method

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

            error_draw = rand(Normal(μ_0, σ^2))                      # draw error term
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
function get_moments(ρ::Float64, σ::Float64, H::Int64, algo::Algorithm;
    mean_opt::Bool = true, var_opt::Bool = false, ac_opt::Bool = false)

    # simulate H different AR(1) sequences of length T
    sim_data = simulate_data(ρ, σ, H, algo)

    # compute average moments across each H by specification

    # (1) get mean and variance only
    if mean_opt == true && var_opt == true && ac_opt == false
        output_moments = [mean(get_mean(sim_data, H)), mean(get_variances(sim_data, H))]
    # (2) get variance and auto correlation only
    elseif mean_opt == false && var_opt == true && ac_opt == true
        output_moments = [mean(get_variances(sim_data, H)), mean(get_auto_corr(sim_data, algo, σ, H))]
    # (3) get all moments
    elseif mean_opt == true && var_opt == true && ac_opt == true
        output_moments = [mean(get_mean(sim_data, H)), mean(get_variances(sim_data, H)),
                        mean(get_auto_corr(sim_data, algo, σ, H))]
    end

    output_moments
end

# ------------------------------------------------------------------------ #
# (2) Minimizing objective function
# ------------------------------------------------------------------------ #

# obj_func: This function defines the objective function to be minimized in the
#           SMM algorithm.
function obj_func(ρ::Float64, σ::Float64, algo::Algorithm, targ::Array{Float64},
    W::Array{Float64, 2}; mean_opt_i::Bool = true, var_opt_i::Bool = false, ac_opt_i::Bool = false)

    @unpack H_model = algo

    # call get_moments to get model moments
    model_mom = get_moments(ρ, σ, H_model, algo;
                mean_opt = mean_opt_i, var_opt = var_opt_i, ac_opt = ac_opt_i)

    # define J (objective function)
    g = targ .- model_mom           # vector of distance between data and model moments
    J = g' * W * g                  # objective function

    return J
end

# newey_west: This function estimates the asymptotic variance-covariance matrix
#             using the Newey-West method.
function newey_west(algo::Algorithm, sim_mom::Array{Float64, 2})
    @unpack lag_length, H_model = algo

    # need to call the gamma function twice, once at 0 and once for lags
    gamma_0 = get_gamma(algo, sim_mom, 0)

    # loop over lags to sum gamma_j for each lag period
    sum_gamma_j = 0.0
    for j = 1:lag_legnth
        gamma_j = get_gamma(algo, sim_mom, j)
        sum_gamma_j += (1-(j/(lag_length + 1))) .* (gamma_j + gamma_j')
    end

    # calculate asymptotic variance-covariance matrix
    S = (1 + (1/H_model)) .* (gamma_0 + sum_gamma_j)

    return S
end

# get_gamma
function get_gamma(algo::Algorithm, res_sim::Array{Any}, lag::Int64)
    @unpack H, T = prim

    """
    """
    mom_sim = [res_sim[1], res_sim[2], res_sim[3]]
    data_sim = res_sim[4]

    gamma_tot = zeros(length(mom_inx),length(mom_inx))

    for t = (1+lag):T
        for h = 1:H
            # No Lagged
            avg_obs = data_sim[t,h]
            if t > 1
                avg_obs_tm1 = data_sim[t-1,h]
            else
                avg_obs_tm1 = 0
            end
            avg_h = mean(data_sim[:,h])
            var_obs = (avg_obs - avg_h)^2
            auto_cov_obs = (avg_obs - avg_h)*(avg_obs_tm1 - avg_h)

            mom_obs_diff = [avg_obs, var_obs, auto_cov_obs] - mom_sim
            mom_obs_diff = mom_obs_diff

            # Lagged
            avg_lag = data_sim[t-lag,h]
            if t - lag > 1
                avg_lag_tm1 = data_sim[t-lag-1,h]
            else
                avg_lag_tm1 = 0
            end
            avg_h = mean(data_sim[:,h])
            var_lag = (avg_lag - avg_h)^2
            auto_cov_lag = (avg_lag - avg_h)*(avg_lag_tm1 - avg_h)


            mom_lag_diff = [avg_lag, var_lag, auto_cov_lag] - mom_sim
            mom_lag_diff = mom_lag_diff

            gamma_tot += mom_obs_diff*mom_lag_diff'
        end
    end

    gamma = (1/(T*H)).*gamma_tot

    return gamma
end


# ------------------------------------------------------------------------ #
# (3) Computing SE
# ------------------------------------------------------------------------ #

function compute_se()
    # calculate numerical derivative using get_moments function
end

# ------------------------------------------------------------------------ #
# (4) Wrapper function for SMM algorithm
# ------------------------------------------------------------------------ #

function solve_smm()
    algo = Algorithm()

    # get_moments for true data
    @unpack H_true, H_model = algo
    targ = get_moments(algo.ρ_0, algo.σ_0, H_true, algo; mean_opt = true, var_opt = true)

    # minimize objective function with W = I
    W = Matrix{Float64}(I, 2, 2)
    b_1 = optimize(b -> obj_func(b[1], b[2], algo, targ, W; mean_opt_i = true,
          var_opt_i = true), [0.5, 1.0]).minimizer

    # compute SE of parameters that minimize above
    b_1_mom = get_moments(b_1[1], b_1[2], H_model, algo; mean_opt = true, var_opt = true)
    S = newey_west(algo, b_1_mom)

    # minimize objective function with W = S^-1
    W_opt = inv(S)
    b_2 = optimize(b -> obj_func(b[1], b[2], algo, targ, W_opt; mean_opt_i = true,
          var_opt_i = true), [0.5, 1.0]).minimizer


    # compute SE of parameters that minimize above


end
