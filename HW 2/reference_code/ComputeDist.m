%--------------------------------------------------------------------------
% File Name: ComputeDist.m
% Author: Philip Coyle
% Date Created: 09/17/2021
% cd /Users/philipcoyle/Documents/School/University_of_Wisconsin/ThirdYear/Fall_2021/TA - Computation/ProblemSets/PS2
%--------------------------------------------------------------------------
% Sample code to construct the transition matrix (pi) compute the stationary distribution (mu).

clear all
close all
clc

%% Housekeeping
% I am making up a fake asset grid just for a working example.
cALPHA = 1;
a_grid = [-cALPHA, 0, cALPHA];
n_a = length(a_grid);

s_grid = [1, 0.5];
n_s = length(s_grid);

% Probabilities
Pee = 0.97;
Pue= 1 - Pee;

Puu = 0.5;
Peu = 1 - Puu;

markov = [Pee, Pue; Peu, Puu];

%% Hypothetical PFs
% See Dean's slides

pf_a = zeros(n_a,n_s);
pf_a(1,1) =  0; % TODAY's STATE: (-cALPHA, e)
pf_a(2,1) =  cALPHA; % (0, e)
pf_a(3,1) =  cALPHA; % (cALPHA, e)

pf_a(1,2) =  -cALPHA; % TODAY's STATE: (-cALPHA, u)
pf_a(2,2) =  -cALPHA; % (0, u)
pf_a(3,2) =  0; % (cALPHA, u)

%% Construct Big Transition Matrix (Pi in Dean's Slides)
% The reason we do this first is because Pi does not depend at all on the
% distribution. So we can construct this first and then repeatedly use it
% when we iterate over the distribution.

% Allocate space
trans_mat = zeros(n_a*n_s);

%intuitively, what is trans_mat?
%trans_mat is a big transition matrix that tells you if you are in a
%certain state today (some asset level and employment status), where are
%you transitioning to tomorrow. The (row col) combo tells you where you are
%today (row) and where you are going tomorrow (col) with some probability
%(the entry in trans_mat). that probability is determined by the markov
%transition matrix.

for i_s = 1:n_s
    s_today = s_grid(i_s);
    for i_a = 1:n_a
        a_today = a_grid(i_a);

        % trans_mat is dimension (n_a*n_s x n_a*n_s). We need a way to map
        % counting up from from i_a = 1:n_a and i_s = 1:n_s into a single
        % number that runs from 1:n_a*n_s.
        row = i_a + n_a*(i_s - 1); % defines row position in trans_mat. Do you see how your state today is encoded in row?
        a_tomorrow = pf_a(i_a, i_s); % read off asset choice based on policy function


        for i_sp = 1:n_s
            for i_ap = 1:n_a

                if a_tomorrow == a_grid(i_ap) % This is one way to construct the indicator function Dean references in his slides.
                    col = i_ap + n_a*(i_sp - 1); % defines col position in trans_mat. Do you see how your state tomorrow is encoded in col?
                    trans_mat(row, col) = markov(i_s, i_sp);
                end
            end
        end
    end
end

%% Solve For Stationary Distribution
% Apply T* operator. That is, iterate on the cross-sectional distribution
% until convergence.

% Ininitial guess for dist_mu
dist_mu=ones(n_a*n_s,1)/(n_a*n_s);

% start iteration
it = 1;
converged = 0;
tol = 1e-5;
maxit = 1000;

while (converged == 0 && it < maxit)
    dist_mu_up = trans_mat' * dist_mu;

    % Calculate the Supnorm
    max_diff = sum(abs(dist_mu_up - dist_mu));
    if max_diff < tol
        converged = 1;
        dist_mu_out = dist_mu_up;
    end

    it=it+1;
    fprintf('Iteration # %2d \tSupnorm: %g \n',it,max_diff);

    % update cross sectional distribution
    dist_mu = dist_mu_up;
end

% disp(dist_mu)
