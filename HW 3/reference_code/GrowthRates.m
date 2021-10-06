%--------------------------------------------------------------------------
% File Name: GrowthRates.m
% Author: Philip Coyle
% Date Created: 09/29/2021
% cd /Users/philipcoyle/cuments/School/University_of_Wisconsin/ThirdYear/Fall_2021/TA - Computation/ProblemSets/PS3
%--------------------------------------------------------------------------
% Sample code to construct age cohort weights

clear all
close all
clc

%% Housekeeping
cGR = 0.011; % Growth Rate
grate = 1/(1 + cGR); % Growth Rate of Cohort

n_t = 66; % Deterministic age which agents die
n_r = 46; % Deterministic age which agents retire


%% Pick inial cohort mass. Get total mass of all cohorts
% Assume that measure of born agents = 1. (We will normalize later)

% Approach 1: Compute with for loop
tot_size_1 = 0;
tot_worker_1 = 0;
tot_ret_1 = 0;
for t = 1:n_t
    tot_size_1 = tot_size_1 + grate^(t-1);
    if t >= n_r
        tot_ret_1 = tot_ret_1 + grate^(t-1);
    else
        tot_worker_1 = tot_worker_1 + grate^(t-1);
    end
end


% Approach 2: Compute using geometric sum formula
tot_size_2 = (1-grate^(n_t))/(1-grate);
tot_worker_2 = (1-grate^(n_r - 1))/(1-grate);
tot_ret_2 = (grate^(n_r-1))*(1-grate^(n_t-n_r+1))/(1-grate);

% Check Comparison
A =[tot_size_1, tot_size_2;
    tot_worker_1, tot_worker_2;
    tot_ret_1, tot_ret_2];

disp(A)

%% Construct cohort weights
AgeWeights = zeros(n_t,1);

for t = 1:n_t
    if t == 1
        AgeWeights(t) = 1/tot_size_2;
    else
        AgeWeights(t) = AgeWeights(t-1)*grate;
    end
end

% Check that mass of all age cohorts sum to 1
disp(sum(AgeWeights))
