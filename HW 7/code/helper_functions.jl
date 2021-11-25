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

    seed_0::Int64 = 1234 # set seed for random draws

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

function newey_west(algo::Algorithm, )
    # see reference code from Phil and slides
    # need to call the gamma function twice, once for standard and once for lags

    """
    Phil's code
    """
    lag_max = 4
    Sy = GammaFunc(prim, res_sim, 0)

    # loop over lags
    for i = 1:lag_max
        gamma_i = GammaFunc(prim, res_sim, i)
        Sy += (gamma_i + gamma_i').*(1-(i/(lag_max + 1)))
    end
    S = (1 + 1/prim.H).*Sy

    return S
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
    # see phil's code for reference

    # get_moments for true data
    algo = Algorithm()
    targ = get_moments(algo.ρ_0, algo.σ_0, 1, algo; mean_opt = true, var_opt = true)

    # minimize objective function with W = I
    W = Matrix{Float64}(I, 2, 2)
    b_1 = optimize(b -> obj_func(b[1], b[2], algo, targ, W; mean_opt_i = true,
          var_opt_i = true), [0.5, 1.0]).minimizer

    # compute SE of parameters that minimize above

    # minimize objective function with W = S^-1

    # compute SE of parameters that minimize above

    """
    Phil's version
    """
    @unpack T, H = prim

    # Step 2: solve for b_2
    res_sim = sim_moments(b_1[1], b_1[2], prim, targ)
    S = NeweyWest(prim, res_sim)
    W_opt = inv(S)
    b_2 = optimize(b->obj_func(b[1], b[2], prim, targ, W_opt), [0.5, 1.0]).minimizer

end
