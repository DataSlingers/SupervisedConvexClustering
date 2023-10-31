function [U_output,V_output,Z_output,class_id_final] = scc_category_ARP(X,Y,w,target_K)

%%% X: p by n
%%% Y: n by K

[p1,n] = size(X);
[n,K] = size(Y);


alpha_X = 2/ norm(X - mean(X,2),'fro')^2;

logit_vec = log ( mean(Y,1) ./  (1 -  mean(Y,1)) ) ; 

null_dev_Y = sum(sum( - Y .* logit_vec )) + n * sum(log(  1 + exp( logit_vec) ) )  ; 

alpha_Y = 1 / null_dev_Y;

alpha2 = 0.5;

alpha_scale = min([alpha_X,alpha_Y]);
alpha_X =  (1-alpha2) * alpha_X / alpha_scale;
alpha_Y =  alpha2 * alpha_Y / alpha_scale;


[x,y] = meshgrid(1:n, 1:n);
E = [x(:) y(:)];

E = E(y(:)>x(:),:);
E_whole = E;
w_whole = w;

active = find(w~=0);
E = E(active,:);

[len_l,~] = size(E);

% Remove Redundant edges
w = w(w~=0);

l1_mat_org = zeros(len_l,n);
l2_mat_org = zeros(len_l,n);

for i = 1:n
    l1_mat_org(:,i) = (E(:,1) == i);
end

for i = 1:n
    l2_mat_org(:,i) = (E(:,2) == i);
end


% D: len_l * n
D = l1_mat_org - l2_mat_org;

% A matrix in multinomial
A2 = kron(D,eye(1));
A3 = kron(eye(K),A2);
A = A3;

% Stroage of Output
MAX_ITER  = 1000;

U = zeros(p1,n);
V = zeros(n,K);

v = V(:);

Z1 = zeros(p1,len_l);
Z2 = zeros(1*K,len_l);
Z = zeros(p1+1*K,len_l);

Lambda1 = zeros(p1,len_l);
Lambda2 = zeros(1*K,len_l);

M1 = inv(alpha_X * eye(n) + D' * D);

gamma = 0.1;
pen_t = 1.1;
k = 1;

U_output = rand(p1,n,MAX_ITER);
V_output = rand(n,K,MAX_ITER);


Z_output = rand(p1+1*K,len_l,MAX_ITER);
U_output(:,:,1) = X;
V_output(:,:,1) = Y;


Z_output(:,:,1) = rand(p1+1*K,len_l);

ORACLE_APPEAR = false;
NO_ORACLE = false;
MAX_ITER_V_sub = 1;
class_id_final = [];
hessian_inv = inv(alpha_Y * 1/2 * eye(K*n) + A'*A) ;



while norm(Z_output(:,:,k)) ~= 0
    
    Z_output(:,:,1) = zeros(p1+1*K,len_l);
    
    if k < 5
        MAX_ITER_INNER = 50;
    else
        MAX_ITER_INNER = 20;
    end
    
    for m = 1:MAX_ITER_INNER
        
        % U update
        U = (alpha_X * X + (Z1+Lambda1) * D) * M1;
        
        % V update       
        
        for l = 1:MAX_ITER_V_sub
            
            Z_tilde = Lambda2 + Z2; % K * len_l
            
            Zt_resize = zeros(len_l,K);
            
            for r = 1:K
                
                Zt_resize(:,r) = Z_tilde( (r-1)*1 + 1 : r*1,:);
                
            end
            
            
            Z_Kbind = [];
            
            for q = 1:K
                Z_Kbind = [Z_Kbind  Zt_resize(:,q)]; % len_l * K
            end
            
            z_tilde = Z_Kbind(:);
            
            % Compute gradient
            
            logit_mat = exp(V) ./ sum(exp(V),2); % V: n * K
            
            gradient = alpha_Y * (logit_mat(:) - Y(:)) +  A' * (A*v-z_tilde);
            norm(gradient);
            
            for s = 1:K
                temp = logit_mat(:,s);
                logit_collapse(:,s) = temp(:);
            end
            
            % logit_collapse n * K            
            
            v = v -  hessian_inv * (gradient) ;
            
            V = reshape(v,[n K]);
                        
            V_Kbind = V';
            
        end
        
        
        % Z-update and Lambda-update:
        for l = 1:len_l
            tmp = E(l,1);
            tmp2 = E(l,2);
            
            Z1(:,l) = U(:,tmp) - U(:,tmp2) - Lambda1(:,l);
            Z2(:,l) = V_Kbind(:,tmp) - V_Kbind(:,tmp2) - Lambda2(:,l);
            
            
            % V(:,l) = soft_threshold(V(:,l), w(l));
            Z(:,l) = group_soft_threshold([Z1(:,l);Z2(:,l)], gamma * w(l));
            
            Z1(:,l) = Z(1:p1,l);
            Z2(:,l) = Z((p1+1):(p1+1*K),l);
            
            Lambda1(:,l) = Lambda1(:,l) + (Z1(:,l) - U(:,tmp) + U(:,tmp2));
            Lambda2(:,l) = Lambda2(:,l) + (Z2(:,l) - V_Kbind(:,tmp) + V_Kbind(:,tmp2));
            
        end
        
        
    end
    
    [no_class,class_id] = group_assign_vertice(Z,w_whole,n);
    
    if ORACLE_APPEAR == false
        class_id_final = class_id;
    end
    
    if no_class == target_K
        class_id_final = class_id;
        ORACLE_APPEAR = true;
        U_output(:,:,k+1) = U;
        V_output(:,:,k+1) = V;
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma_lower = gamma;
        gamma = gamma * pen_t;
        gamma_upper = gamma;
    elseif no_class < target_K && ORACLE_APPEAR == true  && NO_ORACLE == false
        U_output(:,:,k+1) = U;
        V_output(:,:,k+1) = V;
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
        U_output(:,:,k+1) = U;
        V_output(:,:,k+1) = V;
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma = gamma * pen_t;
    elseif no_class < target_K && ORACLE_APPEAR == true && NO_ORACLE == true
        U_output(:,:,k+1) = U;
        V_output(:,:,k+1) = V;
        Z_output(:,:,k+1) = Z;
        k = k + 1;
        gamma = gamma * pen_t;
    elseif no_class > target_K
        U_output(:,:,k+1) = U;
        V_output(:,:,k+1) = V;
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
V_output = V_output(:,:,1:k);
Z_output = Z_output(:,:,1:k);


end