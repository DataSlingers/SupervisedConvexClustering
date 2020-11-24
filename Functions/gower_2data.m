function dd = gower_2data(X,Y)
% X Y are p by n


sc1 = max(pdist(X(1,:)','cityblock'));
sc2 = max(pdist(Y(1,:)','cityblock'));

scale_factor = sc2 / sc1;
% scale_factor = 1;

d1 = gower(X);
d2 = gower(Y);

dd = (d1 + d2/scale_factor ) / 2;
