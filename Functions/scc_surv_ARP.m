function [U_output,Z_output,class_id_final] = scc_surv_ARP(X,Y,w,target_K)
%%% Matrix Update

[p1,n] = size(X);

alpha_X = 2/ norm(X - mean(X,2),'fro')^2;

UU = 2*ones(1,n);
logit = exp(UU) ; % row vector
temp = log(flipud(cumsum(flipud(logit'))));
delta = Y(2,:);
null_dev_Y = - sum( delta .* UU) + sum( delta .* temp') ;
null_dev_Y = null_dev_Y * 2;

alpha_Y = 1 / null_dev_Y;

alpha2 = 0.5;

alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;

surv_time = Y(1,:);  % row vector
delta = Y(2,:);     % row vector

[p2,n] = size(surv_time);

[x,y] = meshgrid(1:n, 1:n);
A = [x(:) y(:)];

A = A(y(:)>x(:),:);
A_whole = A;
w_whole = w;

active = find(w~=0);
A = A(active,:);

[len_l,~] = size(A);

% Remove Redundant edges
w = w(w~=0);

l1_mat_org = zeros(len_l,n);
l2_mat_org = zeros(len_l,n);

for i = 1:n
    l1_mat_org(:,i) = (A(:,1) == i);
end

for i = 1:n
    l2_mat_org(:,i) = (A(:,2) == i);
end


% D: len_l * n
D = l1_mat_org - l2_mat_org;

% Stroage of Output
MAX_ITER  = 500;

U = zeros(p1,n);
V = zeros(p2,n);
V = delta;

Z1 = zeros(p1,len_l);
Z2 = zeros(p2,len_l);
Z = zeros(p1+p2,len_l);

Lambda1 = zeros(p1,len_l);
Lambda2 = zeros(p2,len_l);

M1 = inv(alpha_X * eye(n) + D' * D);
alpha = 0.1;
BETA  = 0.5;

gamma = 0.0001;
pen_t = 1.1;

k = 1;

U_output = rand(p1+p2,n,MAX_ITER);
Z_output = rand(p1+p2,len_l,MAX_ITER);
U_output(:,:,1) = [X ; delta];
Z_output(:,:,1) = rand(p1+p2,len_l);

ORACLE_APPEAR = false;
NO_ORACLE = false;

class_id_final = [];

while norm(Z_output(:,:,k)) ~= 0

    Z_output(:,:,1) = zeros(p1+p2,len_l);

    if k < 5
        MAX_ITER_INNER = 50;
    else
        MAX_ITER_INNER = 50;
    end

for m = 1:MAX_ITER_INNER


    % U update
    U = (alpha_X * X + (Z1+Lambda1) * D) * M1;

    % V-update      
    logit = exp(V);

    denom = cumsum(flipud(logit'));

    denom = 1 ./ denom; % column vector

    denom_delta = flipud(delta') .* denom;

    denom_delta_cum = cumsum(flipud(denom_delta));  % column vector

    gradient = logit' .* denom_delta_cum;  % column vector

    gradient =  alpha_Y * (- delta' + gradient) + ((V * D' - (Z2 + Lambda2))*D)';

    % backtracking

    v_tilde = Z2 + Lambda2;

    fx = cox_cvxclu_subproblem(delta,surv_time,D,v_tilde,V,alpha_Y);      

    v = - gradient;
    t = .01;

    while cox_cvxclu_subproblem(delta,surv_time,D,v_tilde,V + t * v',alpha_Y) > fx + alpha*t* gradient.' * v
        t = BETA*t;
    end
    V = V +  t * v' ;            

    % Z-update and Lambda-update:
    for l = 1:len_l
        tmp = A(l,1);
        tmp2 = A(l,2);

        Z1(:,l) = U(:,tmp) - U(:,tmp2) - Lambda1(:,l);
        Z2(:,l) = V(:,tmp) - V(:,tmp2) - Lambda2(:,l); 

        Z(:,l) = group_soft_threshold([Z1(:,l);Z2(:,l)], gamma * w(l));

        Z1(:,l) = Z(1:p1,l);
        Z2(:,l) = Z((p1+1):(p1+p2),l);

        Lambda1(:,l) = Lambda1(:,l) + (Z1(:,l) - U(:,tmp) + U(:,tmp2));
        Lambda2(:,l) = Lambda2(:,l) + (Z2(:,l) - V(:,tmp) + V(:,tmp2));

    end


end

    [no_class,class_id] = group_assign_vertice(Z,w_whole,n);

    if ORACLE_APPEAR == false
        class_id_final = class_id;
    end

    if no_class == target_K
        class_id_final = class_id;
        ORACLE_APPEAR = true;
        U_output(:,:,k+1) = [U;V];
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma_lower = gamma;
        gamma = gamma * pen_t;
        gamma_upper = gamma;
    elseif no_class < target_K && ORACLE_APPEAR == true  && NO_ORACLE == false
        U_output(:,:,k+1) = [U;V];
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma = gamma * pen_t;
    elseif no_class < target_K && ORACLE_APPEAR == false && NO_ORACLE == false
        gamma = (gamma_lower + gamma_upper)/2;
        gamma_upper = gamma;
        Z1 = Z1_old;
        Z2 = Z2_old;
        Lambda1 = Lambda1_old;
        Lambda2 = Lambda2_old;

        if abs(gamma - gamma_lower) < 1e-3
            NO_ORACLE = true;
            gamma = gamma * pen_t;
        end

    elseif no_class < target_K && ORACLE_APPEAR == false && NO_ORACLE == true
        class_id_final = class_id;
        ORACLE_APPEAR = true;
        U_output(:,:,k+1) = [U;V];
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma = gamma * pen_t;
    elseif no_class < target_K && ORACLE_APPEAR == true && NO_ORACLE == true
        U_output(:,:,k+1) = [U;V];
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma = gamma * pen_t;
    elseif no_class > target_K
        U_output(:,:,k+1) = [U;V];
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma_lower = gamma;
        gamma = gamma * pen_t;
        gamma_upper = gamma;
        Z1_old = Z1;
        Z2_old = Z2;
        Lambda1_old = Lambda1;
        Lambda2_old = Lambda2;
    end




end

U_output = U_output(:,:,1:k);
Z_output = Z_output(:,:,1:k);


end