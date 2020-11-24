function dd = gower_with_cat(X)
% X: p by n

[p n] = size(X);
X_raw = X';
D1 = zeros(n,n);

for i = 1:p
    d1 = pdist(X_raw(:,i),'hamming')/max(pdist(X_raw(:,i),'hamming'));
    D1 = squareform(d1) + D1;
end



dd = squareform(D1)/p ;