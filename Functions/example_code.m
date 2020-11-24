%%% Assume oracle numeber of clusters and features
% Input:
X = ; y = ; Z = ;
type = "";
% X: Continuous/Gaussian data: p * n
% y: Supervising Auxiliary Variable:    
% 1 * n for Gaussian,binary,categorical,count data
% 2 * n for survival data; the first column is survival time, the second column is censoring indicator
% Z: Additional covariate, n * p_cov
% type: type of Supervising Auxiliary Variable
% can be "gaussian","binary","categorical","count" and "survival".
% Z_cov: additional covariates
K = ;  % Desired Number of Clusters
% SCC
[class_id] = scc(X,y,type,K,Z_cov); 
% Output:
% class_id: cluster assignment


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Example Code; Example data from simulation in paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Without covariates
%%% Spherical Data + Gaussian Supervising Auxiliary Variable

load('gaus_S1_X_R_1.mat')
load('gaus_S1_Y_R_1.mat')

% X: p * n, Y: 1 * n
K = 3;
[class_id] = scc(X,Y,"gaussian",K);

%%% Visualize Clustering result
% subplot(1,3,1)
[coeff,score,latent] = pca(X');
n = length(class_id);
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X")



%%% Spherical Data + Categorical Supervising Auxiliary Variable
load('cate_S1_X_R_1.mat')
load('cate_S1_Y_R_1.mat')

% X: p * n, Y: 1 * n
K = 3;
[class_id] = scc(X,Y,"categorical",K);

%%% Visualize Clustering result
subplot(1,2,1)
[coeff,score,latent] = pca(X');
n = length(class_id);
c = zeros(n,3); 
c(class_id == 1,1) = 1; c(class_id == 2,2) = 1; c(class_id == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X, colors are estimated cluster labels")


subplot(1,2,2)
[coeff,score,latent] = pca(X');
n = length(class_id);
c = zeros(n,3); 
c(Y == 1,1) = 1; c(Y == 2,2) = 1; c(Y == 3,3) = 1;
sz = 25; 
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
title("PCA plot in X, colors are labels from supervising auxiliary variable")



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% With covariates
load('gaus_cov_X_R_1.mat')
load('gaus_cov_Y_R_1.mat')
load('gaus_cov_Z_R_1.mat')

K = 3;
[class_id] = scc(X,Y,"gaussian",K,Z_cov); 





