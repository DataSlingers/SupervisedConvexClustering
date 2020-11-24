function [ class_id ] = scc_category( X,Y,K )

[p1 n] = size(X);
%% Dummy Coding
K2 = max(Y);
Y_mat = zeros(n,K2);

for j = 1:n
    Y_mat(j,Y(j)) = 1;
end

%%
eucli_dis = norm(X - mean(X,2),'fro')^2;

logit_vec = log ( mean(Y_mat,1) ./  (1 -  mean(Y_mat,1)) ) ; 
null_dev_Y = sum(sum( - Y_mat .* logit_vec )) + n * sum(log(  1 + exp( logit_vec) ) )  ; 
null_dev_Y = 2 * null_dev_Y;


ratio =  (eucli_dis) / ( (null_dev_Y) + (eucli_dis ));
fs_weight = [ones(1,p1)* ratio/ (p1) 1- ratio];

%% Select K and phi in weights
[K_best,phi,w] = select_K_target_gower( [X;Y], fs_weight,@knn_weight_gower_weighted_dense_cat,@knn_weight_gower_concat_2data_weighted_tune_cat,1);
w = w ./ sum(w);


%% [U_output,Z_output] = intclu_l2_logll_CARP_SOK(X,Y,w,3);
% [U_output,V_output,Z_output,class_id_final] = intclu_l2_multinomial_matrix_CARP_SOK(X,Y_mat,w,3);
[U_output,V_output,Z_output,class_id_final] = scc_category_ARP(X,Y_mat,w,K);
%% Rand Index

V_round = round(Z_output,3);
[class_id,iter_cut] = get_cluster_assignment(V_round,w,n,K);



len_V = size(V_round,3);
if length(unique(class_id)) == 1
    class_id_mat = zeros(len_V,n);
    class_no_vec = zeros(len_V,1);
    for i = 1:len_V 
        [class_no_vec(i), class_id_mat(i,:)] = group_assign_vertice(V_round(:,:,i),w,n);
    end

    iter_cut = max(find(class_no_vec > 1));
    class_id = class_id_mat(iter_cut,:);
end





end

