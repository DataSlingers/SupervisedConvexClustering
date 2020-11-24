function [ kmeans_ri, hclust_ri ] = exist_methods( X,K,truth,distance_metric )

if strcmp(distance_metric,'squaredeuclidean') 
    distance_metric1 = 'sqeuclidean';
else
    distance_metric1 = distance_metric;
end


%% Clustering on X1
%%% K-means
trials = 10;
kmeans_ri_ind = zeros(1,trials);
for r = 1:trials
    idx = kmeans(X',K,'Distance',distance_metric1);
    kmeans_ri_ind(r) = rand_index(idx',truth,'adjusted');
end
kmeans_ri = mean(kmeans_ri_ind);

% Hierarchical Clustering
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    Z = linkage(X','complete',distance_metric);
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(Z,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

hclust_ri.complete = mean(hclust_ri_ind);


hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    Z = linkage(X','single',distance_metric);
    idx3 = cluster(Z,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

hclust_ri.single = mean(hclust_ri_ind);


%%% Average
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    Z = linkage(X','average',distance_metric);
    idx3 = cluster(Z,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

hclust_ri.average = mean(hclust_ri_ind);


% Ward
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    Z = linkage(X','ward');
    idx3 = cluster(Z,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

hclust_ri.ward = mean(hclust_ri_ind);







end

