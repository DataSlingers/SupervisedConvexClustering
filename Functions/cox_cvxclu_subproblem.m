function [obj] = cox_cvxclu_subproblem(delta,surv_time,D,v_tilde,U,alpha_Y)

logit = exp(U) ; % row vector

temp = log(flipud(cumsum(flipud(logit')))); % column vector
     

obj = alpha_Y * (- sum( delta .* U) + sum( delta .* temp')) +  1/2 * norm(U * D' - v_tilde ).^2 ;
    
end

