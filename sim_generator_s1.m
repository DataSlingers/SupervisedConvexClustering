function [ X,Y,truth ] = sim_generator_s1(  )

n = 120;
%%% Chunk One: Gaussian
p1 = 30;

% SNR1 = 25;
% MU1 = repmat([8,10],[1 p1/2]);
% X1 = mvnrnd(MU1, SNR1 * eye(p1) ,n/3);
% 
% MU2 = repmat([10,0],[1 p1/2]);
% X2 = mvnrnd(MU2, SNR1 * eye(p1) ,n/3);
% 
% MU3 = repmat([12,10],[1 p1/2]);
% X3 = mvnrnd(MU3,  SNR1 * eye(p1) ,n/3);
% 
% X = [X1;X2;X3];

SNR1 = 1;
MU1 = repmat([1.6,2],[1 p1/2]);
X1 = mvnrnd(MU1, SNR1 * eye(p1) ,n/3);

MU2 = repmat([2,0],[1 p1/2]);
X2 = mvnrnd(MU2, SNR1 * eye(p1) ,n/3);

MU3 = repmat([2.4,2],[1 p1/2]);
X3 = mvnrnd(MU3,  SNR1 * eye(p1) ,n/3);

X = [X1;X2;X3];



[coeff,score,latent] = pca(X);
c = zeros(n,3); c(1:n/3,1) = 1; c(n/3+1:n/3*2,2) = 1; c(n/3*2+1:n,3) = 1;
sz = 25;
scatter(score(:,1),score(:,2),sz,c,'filled');
labels = cellstr(num2str([1:n]'));
% text(score(:,1),score(:,2),labels);

%% Chunk Three: Binary
%%% Simulat from Copula ?
rng(123); % good
%%% Chunk Two
% n = 30;
%%% Simulate Binary Data using different Bernoulli Distribution

n1 = n/3;
n2 = n/3;
n3 = n/3;
p2 = 1;
% Generate Data for the first class
Z1 = zeros(n1,p2);
MU1 = [.85];
for s = 1:p2
    Z1(:,s) = binornd(1,MU1(s),[1 n1]);
end

% Generate Data for the 2nd class
Z2 = zeros(n2,p2);
MU2 = [.5];
for s = 1:p2
    Z2(:,s) = binornd(1,MU2(s),[1 n2]);
end


% Generate Data for the 3rd class
Z3 = zeros(n3,p2);
MU3 = [.15];
for s = 1:p2
    Z3(:,s) = binornd(1,MU3(s),[1 n3]);
end

Z_binary = [Z1 ;Z2; Z3];

Z = Z_binary;

% X = bsxfun(@minus, X, mean(X));
% X = bsxfun(@rdivide, X, std(X));
[coeff,score,latent] = pca(Z);

c = zeros(n,3); c(1:n/3,1) = 1; c(n/3+1:n/3*2,2) = 1; c(n/3*2+1:n,3) = 1;
% c = zeros(n,3); c(1:n,1) = 1;
% sz = 25;
% scatter(score(:,1),score(:,2),sz,c,'filled');
% labels = cellstr(num2str([1:n]'));
% text(score(:,1),score(:,2),labels);






%%
truth = [ones(1,n/3) 2*ones(1,n/3) 3*ones(1,n/3)];
K = 3;


X = X';
Y = Z';



end

