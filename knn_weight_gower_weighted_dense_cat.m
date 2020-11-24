function [w] = knn_weight_gower_weighted_dense_cat(X,gamma,fs_weight,phi)
        
[p,n] = size(X);
    
tmp = gower_weighted_with_cat(X(1:(p-1),:),X(p,:),fs_weight);
        
% calculate w based on distance 'tmp'
% use gaussian kernel
w = gamma *  exp(-phi*tmp);
    
    

end