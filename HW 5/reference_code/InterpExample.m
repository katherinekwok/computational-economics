%--------------------------------------------------------------------------
% File Name: InterpExample.m
% Author: Philip Coyle
% Date Created: 09/29/2021
%--------------------------------------------------------------------------
% Sample function and code on how to do linear interpolation in 1 and 2
% dimensions.

clear all
close all
clc

%% Housekeeping
n_grid = 11;
n_fine = 1001;

%% 1 dimensional example
% Interpolate off grid for exp(x).
x_grid = linspace(0,10,n_grid);
y_out = exp(x_grid);

%x_grid is [0, 1, 2, ..., 10]. What if we want to evaluate exp(0.5)? We
%need to interpolate. 
y_val = lin_interp_1d(x_grid, 0.5, y_out);

x_grid_fine = linspace(0,10,n_fine);
y_truth = exp(x_grid_fine);
y_interp = zeros(n_fine, 1);
for i = 1:n_fine
    interp_point = x_grid_fine(i);
    y_interp(i) = lin_interp_1d(x_grid, interp_point, y_out);
end

% Plot Results
figure(1);
subplot(2,2,1)
box on
grid on
hold on
plot(x_grid_fine, y_truth,'-k','LineWidth',2);
xlabel('x','fontsize',15)
ylabel('e(x)','fontsize',15)
title('True Function','Fontweight','normal','fontsize',15)

subplot(2,2,2)
box on
grid on
hold on
plot(x_grid_fine, y_interp,'-k','LineWidth',2);
xlabel('x','fontsize',15)
ylabel('e(x)','fontsize',15)
title('Interpolated Function','Fontweight','normal','fontsize',15)

%% 2 dimensional example
% Interpolate off grid for exp(x).
x_grid = linspace(0,10,n_grid);
y_grid = linspace(-10,10,n_grid);

x_grid_mesh = zeros(n_grid,n_grid);
y_grid_mesh = zeros(n_grid,n_grid);
z_out = zeros(n_grid,n_grid);

for i_y = 1:n_grid
    y = y_grid(i_y);
    for i_x = 1:n_grid
        x = x_grid(i_x);
        z = x^2*y;
        
        x_grid_mesh(i_x, i_y) = x;
        y_grid_mesh(i_x, i_y) = y;
        z_out(i_x, i_y) = z;
    end
end

%x_grid and y_grid are coarse. What if we want to evaluate function at x = 0.5 and y = 2.5? 
%We need to interpolate. 
z_val_interp = lin_interp_2d(x_grid, y_grid, 0.5, 2.5, z_out);

x_grid_fine = linspace(0,10,n_fine);
y_grid_fine = linspace(-10,10,n_fine);

x_grid_mesh_fine = zeros(n_fine,n_fine);
y_grid_mesh_fine = zeros(n_fine,n_fine);
z_truth = zeros(n_fine,n_fine);
z_interp = zeros(n_fine,n_fine);

for i_y = 1:n_fine
    y = y_grid_fine(i_y);
    for i_x = 1:n_fine
        x = x_grid_fine(i_x);
        z = x^2*y;
        z_interp_val = lin_interp_2d(x_grid, y_grid, x, y, z_out);
        
        x_grid_mesh_fine(i_x, i_y) = x;
        y_grid_mesh_fine(i_x, i_y) = y;
        
        z_truth(i_x, i_y) = z;
        z_interp(i_x, i_y) = z_interp_val;
    end
end
     
% Plot Results
figure(2);
subplot(2,2,1)
surf(x_grid_mesh_fine, y_grid_mesh_fine, z_truth)
xlabel('x','fontsize',15)
ylabel('y','fontsize',15)
zlabel('x^2y','fontsize',15)
title('True Function','Fontweight','normal','fontsize',15)

subplot(2,2,2)
surf(x_grid_mesh_fine, y_grid_mesh_fine, z_interp)
xlabel('x','fontsize',15)
ylabel('y','fontsize',15)
zlabel('x^2y','fontsize',15)
title('Interpolated Function','Fontweight','normal','fontsize',15)

%% Linear Interpolation over 1 dimension
function [o1] = lin_interp_1d(x1, x1i, pf1) 
% Inputs:
%   x1  : Grids
%   x1i : Point to evaluate
%   pf1 : Policy function interpolating over
% Outputs:
%   o* : Interpolated/extrapolated values of dimension x*ipts

% Grid lengths
nx1 = length(x1);

% Number of stochastic realizations
x1ipts = length(x1i);

% Preallocate output
o1 = zeros(x1ipts,1);

for i1 = 1:x1ipts
    
    s1 = x1(2) - x1(1);
    x1i_min = x1i(i1) - x1(1);
    loc1 = min(nx1 - 1, max(1,floor(x1i_min/s1) + 1));
    
    xi = x1i(i1);
    xi_left = x1(loc1);
    xi_right = x1(loc1+1);
    
    w_2 = (xi - xi_left)./(xi_right - xi_left);
    w_1 = 1 - w_2;
    w1 = [w_1 w_2];
    
    for m1 = 0:1
        o1(i1) = o1(i1) + w1(m1+1)*pf1(loc1+m1);
    end
    
end

end



%% Linear Interpolation over 2 dimensions
function [o1] = lin_interp_2d(x1, x2, x1i, x2i, pf1)
% Inputs:
%   x1, x2  :   Grids
%   x1i, x1i:   Point to evaluate
%   pf1     :   Policy function interpolating over
% Outputs:
%   o*      :   Interpolated/extrapolated values of dimension x*ipts

% Grid lengths
nx1 = length(x1);
nx2 = length(x2);

% Number of stochastic realizations
x2ipts = length(x2i);

% Preallocate output
o1 = zeros(x2ipts,1);

s1 = x1(2) - x1(1);
x1i_min = x1i - x1(1);
loc1 = min(nx1-1,max(1,floor(x1i_min/s1) + 1));

for i2 = 1:x2ipts
    s2 = x2(2) - x2(1);
    x2i_min = x2i(i2) - x2(1);
    loc2 = min(nx2-1,max(1,floor(x2i_min/s2) + 1));

    xi = [x1i x2i(i2)];
    xi_left = [x1(loc1) x2(loc2)];
    xi_right = [x1(loc1+1) x2(loc2+1)];

    w_2 = (xi - xi_left)./ (xi_right - xi_left);
    w_1 = 1 - w_2;
    w1 = [w_1(1) w_2(1)];
    w2 = [w_1(2) w_2(2)];
    
    for m2 = 0:1
        for m1 = 0:1
            o1(i2) = o1(i2) + w1(m1+1)*w2(m2+1)*pf1(loc1+m1,loc2+m2);
        end
    end
end
end
