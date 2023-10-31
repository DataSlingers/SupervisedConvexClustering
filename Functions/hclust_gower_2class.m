function [ conca_hclust_ri, agg_hclust_ri ] = hclust_gower_2class( X,Y,K,truth )

trials = 10;
% The following are the same:
% Z1 = linkage(X','single','cityblock');

% Y = pdist(X','cityblock');
% Z2 = linkage(Y,'single');
% Z1 = Z2

XY = [X ; Y];
% Hierarchical Clustering using concatenate gower distance
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower(XY));
    ZZ = linkage(D,'complete');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

conca_hclust_ri.complete = mean(hclust_ri_ind);


hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower(XY));
    ZZ = linkage(D,'single');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

conca_hclust_ri.single = mean(hclust_ri_ind);


%%% Average
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower(XY));
    ZZ = linkage(D,'average');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

conca_hclust_ri.average = mean(hclust_ri_ind);


% Ward
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower(XY));
    ZZ = linkage(D,'ward');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

conca_hclust_ri.ward = mean(hclust_ri_ind);



%% Hierarchical Clustering using Aggregate gower distance
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower_2data(X,Y));
    ZZ = linkage(D,'complete');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

agg_hclust_ri.complete = mean(hclust_ri_ind);


hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower_2data(X,Y));
    ZZ = linkage(D,'single');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

agg_hclust_ri.single = mean(hclust_ri_ind);


%%% Average
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower_2data(X,Y));
    ZZ = linkage(D,'average');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

agg_hclust_ri.average = mean(hclust_ri_ind);


% Ward
hclust_ri_ind = zeros(1,trials);
for r = 1:trials
    D = squareform(gower_2data(X,Y));
    ZZ = linkage(D,'ward');
    % Z = linkage(X','average','cityblock');
    idx3 = cluster(ZZ,'maxclust',K);
    hclust_ri_ind(r) = rand_index(idx3',truth,'adjusted');
end

agg_hclust_ri.ward = mean(hclust_ri_ind);











end

