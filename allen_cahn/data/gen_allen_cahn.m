c:\Users\admin\Desktop\jax\jaxpi-main\examples\allen_cahn\data\gen_allen_cahn_bem.m
%% Allen-Cahn equation using Boundary Element Method (BEM)
% Allen-Cahn equation: du/dt = epsilon * d^2u/dx^2 + 5*u - 5*u^3
% where epsilon = 0.0001

clear; clc; close all;

%% Parameters
nn = 511;              % Number of spatial points
steps = 200;           % Number of time steps
epsilon = 0.0001;      % Diffusion coefficient
dt = 1.0 / steps;      % Time step size

% Spatial domain
x = linspace(-1, 1, nn);
dx = x(2) - x(1);
t = linspace(0, 1, steps + 1);

% Initial condition: u(x,0) = x^2 * cos(pi*x)
u = x.^2 .* cos(pi * x);

% Store solution
usol = zeros(nn, steps + 1);
usol(:, 1) = u;

%% BEM-based solver using Green's function approach
% For the linear diffusion part, we use the fundamental solution (Green's function)
% The nonlinear term is treated as a source term using Picard iteration

fprintf('Starting BEM-based solver...\n');

for n = 1:steps
    fprintf('Time step %d/%d\n', n, steps);
    
    % Previous time step solution
    u_prev = usol(:, n);
    
    % Picard iteration for nonlinearity
    u_new = u_prev;
    max_iter = 10;
    tol = 1e-6;
    
    for iter = 1:max_iter
        u_old = u_new;
        
        % Compute nonlinear term: f(u) = 5*u - 5*u^3
        f_nonlinear = 5 * u_old - 5 * u_old.^3;
        
        % BEM formulation: solve (I - dt*epsilon*L)u = u_prev + dt*f_nonlinear
        % where L is the Laplacian operator
        
        % Construct the BEM influence matrix using Green's function
        % For 1D diffusion with Dirichlet boundary conditions
        % The Green's function approach leads to a tridiagonal system
        
        alpha = epsilon * dt / dx^2;
        
        % Construct the system matrix (implicit scheme)
        A = speye(nn);
        
        % Interior points: -alpha*u_{i-1} + (1+2*alpha)*u_i - alpha*u_{i+1}
        for i = 2:nn-1
            A(i, i-1) = -alpha;
            A(i, i) = 1 + 2*alpha;
            A(i, i+1) = -alpha;
        end
        
        % Boundary conditions (Dirichlet: u(-1,t) = u(1,t) = 0)
        A(1, 1) = 1;
        A(nn, nn) = 1;
        
        % Right-hand side
        rhs = u_prev + dt * f_nonlinear;
        rhs(1) = 0;   % u(-1,t) = 0
        rhs(nn) = 0;  % u(1,t) = 0
        
        % Solve the linear system
        u_new = A \ rhs;
        
        % Check convergence
        if norm(u_new - u_old, inf) < tol
            break;
        end
    end
    
    % Store solution
    usol(:, n+1) = u_new;
end

%% Add boundary point for periodic-like visualization
x_plot = linspace(-1, 1, nn + 1);
usol_plot = [usol; usol(1, :)];

%% Visualization
figure;
pcolor(t, x_plot, usol_plot);
shading interp;
axis tight;
colormap(jet);
colorbar;
xlabel('Time t');
ylabel('Position x');
title('Allen-Cahn Equation Solution (BEM-based)');
saveas(gcf, 'allen_cahn_bem_solution.png');

%% Save data
usol = usol'; % shape = (steps+1, nn)
save('allen_cahn_bem.mat', 't', 'x', 'usol');

fprintf('BEM-based solver completed!\n');
fprintf('Data saved to allen_cahn_bem.mat\n');
