function [output] = nat_poisson_Usub(xi,v_tilde,D,alpha_Y,u_start)
QUIET    = 0;
MAX_ITER = 2;

alpha = 0.1;
BETA  = 0.5;

n = length(xi);
% Storage of output
u = zeros(n, MAX_ITER);
z = zeros(n, MAX_ITER);

xit = xi';  % xit is a column vector
u(:,1) = log(xit+1e-5);
u(:,1) = u_start;

v_tilde = v_tilde';  % v_tilde is a column vector

% ADMM
% tic
% ADMM update
for k = 1:MAX_ITER-1  %%% One-step update
        
    gradient =  alpha_Y * (exp(u(:,k)) -  xit)  + D' * (D *u(:,k)-v_tilde);
    
    % hessian = diag( exp(u(:,k)) ) + D'*D;
    hessian = eye(n);
    % backtracking
    v = - hessian\gradient;
    fx = nat_poisson_bc_obj(xit,D,v_tilde,u(:,k),alpha_Y);
    %    
    t = 1;
         
     while nat_poisson_bc_obj(xit,D,v_tilde,u(:,k) + t*v,alpha_Y) > fx + alpha*t* gradient.' * v
            t = BETA*t;
     end
     % disp(t);
    
     u(:,k+1) = u(:,k) +  t * v ;
    
    % u = u - hessian\gradient ;
                 
%     if norm(u(:,k+1) - u(:,k)) < 1e-5
%         break
%     end
    
    % if norm(obj(k+1) - obj(k)) < 1e-5
    
end
% h.admm_toc = toc;


output = u(:,k+1);
    

end