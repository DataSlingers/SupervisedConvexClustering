function [ class_id ] = scc_gaus( X,Y,K )

[p1,n] = size(X);

eucli_dis1 = norm(X - mean(X,2),'fro')^2;
eucli_dis2 = norm(Y - mean(Y,2),'fro')^2;

ratio =  (eucli_dis1) / ( (eucli_dis2) + (eucli_dis1 ));
fs_weight = [ones(1,p1)* ratio/ (p1) 1- ratio];

%%% Select K and phi in weights
[K_best,phi,w] = select_K_target_gower( [X;Y], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_2data_weighted_tune,1);
w = w ./ sum(w);

[U_output,Z_output] = scc_gaus_ARP(X,Y,w,K);
% Same as
% [U_output,Z_output] = scc_gaus_ARP_v2(X,Y,w,K);

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

