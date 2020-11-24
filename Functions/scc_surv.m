function [ class_id_unsort ] = scc_surv( X,y,K )

X = X';  % X now n by p 
y = y';  % y now n by 2; the first column is survival time, the second column is censoring indicator

n = size(X,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sort the data based on survival time
data_sorted = sort(y(:,1));
[~, rnk] = ismember(data_sorted,y(:,1));
y = y(rnk,:);
Y = y';

%%% Kaplan_meiern curve for all data
[f_all,x_all] = ecdf(y(:,1),'censoring',1-y(:,2));
% ecdf(y(:,1),'censoring',1-y(:,2),'function','survivor');
% stairs(x_all, 1-f_all, ':y', 'LineWidth',2)

cdf_y = zeros(1,n);
for i = 1:n
    index = find( abs(y(i,1) - x_all) == min( abs(y(i,1) - x_all)));
    index = max(index);
    cdf_y(i) = f_all(index);
end

%% Supervised data, also need to rank X
X = X(rnk,:);
X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p1 n] = size(X);

eucli_dis = norm(X - mean(X,2),'fro')^2;

UU = 2*ones(1,n);
logit = exp(UU) ; % row vector
temp = log(flipud(cumsum(flipud(logit'))));
delta = Y(2,:);
null_dev_Y = - sum( delta .* UU) + sum( delta .* temp') ;
null_dev_Y = null_dev_Y * 2;


ratio =  (eucli_dis) / ( (null_dev_Y) + (eucli_dis ));
fs_weight = [ones(1,p1)* ratio/ (p1) 1- ratio];

%%% Select K and phi in weights
[K_best,phi,w] = select_K_target_gower( [X;cdf_y], fs_weight,@knn_weight_gower_weighted_dense,@knn_weight_gower_concat_2data_weighted_tune,1);
w = w ./ sum(w);

[U_output,Z_output,class_id_final] = scc_surv_ARP(X,Y,w,K);


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


% https://blogs.mathworks.com/loren/2007/08/21/reversal-of-a-sort/
% Transform back to unsorted cluster label
class_id_unsort(rnk) = class_id;


end

