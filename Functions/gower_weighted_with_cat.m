function dd = gower_weighted_with_cat(X,Y,fs_weight)
% X Y are p by n

[p1 n] = size(X);
[p2 n] = size(Y);

X_raw = X';
Y_raw = Y';

D1 = zeros(n,n);

for i = 1:p1
    d1 = pdist(X_raw(:,i),'cityblock')/max(pdist(X_raw(:,i),'cityblock'));
    d1 = d1 *  fs_weight(i);
    D1 = squareform(d1) + D1;
end

for i = 1:p2
    d1 = pdist(Y_raw(:,i),'hamming')/max(pdist(Y_raw(:,i),'hamming'));
    d1 = d1 *  fs_weight(p1+i);
    D1 = squareform(d1) + D1;
end



dd = squareform(D1)/(p1+p2) ;