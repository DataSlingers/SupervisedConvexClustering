function [ obj ] = nat_poisson_bc_obj(xit,D,v_tilde,u,alpha_Y)

obj = alpha_Y * sum(sum(- u .* xit  + exp(u))) + 1/2 * norm(D*u-v_tilde).^2;


end

