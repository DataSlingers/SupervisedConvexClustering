function [kmeans_ri_X,hclust_ri_X,kmeans_ri_Y,hclust_ri_Y, kmeans_ri_XY,hclust_ri_XY,conca_hclust_ri, agg_hclust_ri] = icc_methods_2data( X,Y,K,truth )
% function [kmeans_ri_X,hclust_ri_X,kmeans_ri_Y,hclust_ri_Y, kmeans_ri_XY,hclust_ri_XY,conca_hclust_ri, agg_hclust_ri] = icc_methods_2data( X,Y,K,truth )

%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% X: p1 * n
% Y: p2 * n
distance_metric  = 'squaredeuclidean';
%% K-means and Hclust
[ kmeans_ri, hclust_ri ] = exist_methods( X,K,truth,distance_metric );
kmeans_ri_X = kmeans_ri;
hclust_ri_X = hclust_ri;

[ kmeans_ri, hclust_ri ] = exist_methods( Y,K,truth,distance_metric );
kmeans_ri_Y = kmeans_ri;
hclust_ri_Y = hclust_ri;



XY = [X ; Y];
[ kmeans_ri, hclust_ri ] = exist_methods( XY ,K,truth,distance_metric );
kmeans_ri_XY = kmeans_ri;
hclust_ri_XY = hclust_ri;


%% Hclust using Gower distances

[conca_hclust_ri, agg_hclust_ri ] = hclust_gower_2class( X,Y,K,truth );





end

