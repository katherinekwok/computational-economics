# --------------------------------------------------------------------------
# File Name: smm_model.jl
# Author: Philip Coyle
# -------------------------------------------------------------------------

# Main SMM Function
function SMM(prim::Primitives, targ::Targets)
    @unpack T, H = prim
    # @unpack data_true, avg, variance, shocks = targ

    # Step 1 solve for b_1
    W = Matrix{Float64}(I)
    b_1 = optimize(b->obj_func(b[1], b[2], prim, targ, W), [0.5, 1.0]).minimizer

    # Step 2: solve for b_2
    res_sim = sim_moments(b_1[1], b_1[2], prim, targ)
    S = NeweyWest(prim, res_sim)
    W_opt = inv(S)
    b_2 = optimize(b->obj_func(b[1], b[2], prim, targ, W_opt), [0.5, 1.0]).minimizer
end


function NeweyWest(prim::Primitives, res_sim::Array{Any})
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

function GammaFunc(prim::Primitives, res_sim::Array{Any}, lag::Int64)
    @unpack H, T = prim

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
