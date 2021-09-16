function [Q]=Qsample(I)
%This subroutine computes Q
%Input matrix I is the index of decision rules
    %e.g. I(1,1) is the decision of a (a=-2,s=e) hh
    
a=-2:.01:30; %Grids for a

pos=find(a==1); %Position of 1 in grid a
%Assume blo=-0.5 and bhi=0.5 are the borders
blo=0.5;bhi=1.5;

%Calculationg probabilities for each s'
pr1=(a(I(pos,1))>=blo-1 & a(I(pos,1))<=bhi-1);
pr2=(a(I(pos,1))>=blo-0.5 & a(I(pos,1))<=bhi-0.5);

%Summing over s'
Q=pr1*0.97+pr2*0.03;

end