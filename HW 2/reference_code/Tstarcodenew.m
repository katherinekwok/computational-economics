% Sample code for the invariant distribution iteration in a deterministic environment 
clc;
clear;
close;
%%
% suppose you solve the VFI problem over a grid of 100 point as follow
agrid=(0:0.1:9.9);
na=size(agrid,2);

% suppose further that the solution of policy function for a is as follows
ga(1:80)=agrid(2:81); 
ga(81:100)=agrid(82);

ga_ind(1:80)=(2:81); % indicies of the gird implied by the policy function
ga_ind(81:100)=82;

% take an initial guess of the cross section distribution 
mu0=ones(na,1);
mu0=mu0./na;

% start iteration 
iter=1;
supnorm=1;

while (supnorm>0.0001)
    % iterate on the distribution using policy function
    mu1=zeros(na,1);
    for ai=1:na % current state
        mu1(ga_ind(ai))=mu1(ga_ind(ai))+mu0(ai);
    end
    
    % caculate the supnorm
    supnorm=norm(mu0-mu1);
    iter=iter+1;
    fprintf('Iteration # %2d \tSupnorm: %g \n',iter,supnorm);
    % update cross sectional distribution
    mu0=mu1;
end

% take a look at mu and think about why all mass concentrate on one point