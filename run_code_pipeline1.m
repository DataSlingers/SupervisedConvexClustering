%%% Pipeline
B = 10;
para1 = zeros(1,B);
para2 = zeros(1,B);
para3 = zeros(1,B);
para4 = zeros(1,B);

% exist methods
kmeans_ri_X = zeros(1,B);
kmeans_ri_Y = zeros(1,B);
kmeans_ri_XY = zeros(1,B);

X_single = zeros(1,B);
X_complete = zeros(1,B);
X_average = zeros(1,B);
X_ward = zeros(1,B);

Y_single = zeros(1,B);
Y_complete = zeros(1,B);
Y_average = zeros(1,B);
Y_ward = zeros(1,B);

XY_single = zeros(1,B);
XY_complete = zeros(1,B);
XY_average = zeros(1,B);
XY_ward = zeros(1,B);

conca_single = zeros(1,B);
conca_complete = zeros(1,B);
conca_average = zeros(1,B);
conca_ward = zeros(1,B);

rng(1);
for b = 1:B
    [X,Y,truth] = sim_generator_s1(); % hard case but win much
    % [X,Y,truth] = sim_generator_s1_3(); % easy case all does well, not
    % all does well?
    % [X,Y,truth] = sim_generator_s3();
%     if mod(b,2) == 1
%         % Y(41:80) = 1 - Y(41:80);
%         Y(61:70) = 1 - Y(61:70); % works
%     end
    % rng('shuffle')
    rng(b);
    % Y(61:70) = binornd(1,0.5,[1 10]); % works
    Y(51:60) = binornd(1,0.5,[1 10]); % works
    % Y(41:80) = binornd(1,0.5,[1 size(Y,2)/3]); % might only have 2 clusters
    K = max(truth);
    class_id = scc_binary( X,Y,K );
    para1(b) = cluster_rand_group(class_id,truth);
    disp(para1(b));
    % para2(b) = scc_binary_para( X,Y,truth,2 );
    % para3(b) = scc_binary_para( X,Y,truth,3 );
    % para4(b) = scc_binary_para( X,Y,truth,4 );
    % Exist Method
    [kmeans_ri_X(b),hclust_ri_X,kmeans_ri_Y(b),hclust_ri_Y, kmeans_ri_XY(b),hclust_ri_XY,conca_hclust_ri, agg_hclust_ri] = icc_methods_2data( X,Y,K,truth );

    X_single(b) = hclust_ri_X.single;
    X_complete(b) = hclust_ri_X.complete;
    X_average(b) = hclust_ri_X.average;
    X_ward(b) = hclust_ri_X.ward;
    
    Y_single(b) = hclust_ri_Y.single;
    Y_complete(b) = hclust_ri_Y.complete;
    Y_average(b) = hclust_ri_Y.average;
    Y_ward(b) = hclust_ri_Y.ward;       
    
    XY_single(b) = hclust_ri_XY.single;
    XY_complete(b) = hclust_ri_XY.complete;
    XY_average(b) = hclust_ri_XY.average;
    XY_ward(b) = hclust_ri_XY.ward;  
    
    
    conca_single(b) = conca_hclust_ri.single;
    conca_complete(b) = conca_hclust_ri.complete;
    conca_average(b) = conca_hclust_ri.average;
    conca_ward(b) = conca_hclust_ri.ward;
    
%     filenameX = ['logi_S1_X_R_', int2str(b),'.mat'];
%     save(filenameX,'X');
%     
%     filenameY = ['logi_S1_Y_R_', int2str(b),'.mat'];
%     save(filenameY,'Y');    
    
end




% result = [ mean(para1), mean(para2), mean(para3), mean(para4),
%            mean(X_single), mean(X_complete), mean(X_average), mean(X_ward),
%            mean(Y_single), mean(Y_complete), mean(Y_average), mean(Y_ward),
%            mean(XY_single), mean(XY_complete), mean(XY_average), mean(XY_ward),
%            mean(conca_single), mean(conca_complete), mean(conca_average), mean(conca_ward)];



result = [ mean(X_single), mean(X_complete), mean(X_average), mean(X_ward),
           mean(Y_single), mean(Y_complete), mean(Y_average), mean(Y_ward),
           mean(XY_single), mean(XY_complete), mean(XY_average), mean(XY_ward),
           mean(conca_single), mean(conca_complete), mean(conca_average), mean(conca_ward),
           mean(para1), mean(para2), mean(para3), mean(para4)];



sd = [sqrt(var(X_ward))/sqrt(B);
sqrt(var(Y_ward))/sqrt(B);
sqrt(var(XY_ward))/sqrt(B);
sqrt(var(conca_ward))/sqrt(B);
sqrt(var(para1))/sqrt(B)];



result = [ mean(X_single), mean(X_complete), mean(X_average), mean(X_ward),
           mean(Y_single), mean(Y_complete), mean(Y_average), mean(Y_ward),
           mean(XY_single), mean(XY_complete), mean(XY_average), mean(XY_ward),
           mean(conca_single), mean(conca_complete), mean(conca_average), mean(conca_ward),
           mean(para1), mean(para2), mean(para3), mean(para4)];

% sd = [ mean(X_single), mean(X_complete), mean(X_average), mean(X_ward),
%            mean(Y_single), mean(Y_complete), mean(Y_average), mean(Y_ward),
%            mean(XY_single), mean(XY_complete), mean(XY_average), mean(XY_ward),
%            mean(conca_single), mean(conca_complete), mean(conca_average), mean(conca_ward),
%            mean(para1), mean(para2), mean(para3), mean(para4)];

sd = [sqrt(var(X_single))/sqrt(B),sqrt(var(X_complete))/sqrt(B),sqrt(var(X_average))/sqrt(B),sqrt(var(X_ward))/sqrt(B);
sqrt(var(Y_single))/sqrt(B),sqrt(var(Y_complete))/sqrt(B),sqrt(var(Y_average))/sqrt(B),sqrt(var(Y_ward))/sqrt(B);
sqrt(var(XY_single))/sqrt(B),sqrt(var(XY_complete))/sqrt(B),sqrt(var(XY_average))/sqrt(B),sqrt(var(XY_ward))/sqrt(B);
sqrt(var(conca_single))/sqrt(B),sqrt(var(conca_complete))/sqrt(B),sqrt(var(conca_average))/sqrt(B),sqrt(var(conca_ward))/sqrt(B);
sqrt(var(para1))/sqrt(B),sqrt(var(para1))/sqrt(B),sqrt(var(para1))/sqrt(B),sqrt(var(para1))/sqrt(B)];
           
           
           
       
% sd = [sqrt(var(X_ward))/sqrt(B);
% sqrt(var(Y_ward))/sqrt(B);
% sqrt(var(XY_ward))/sqrt(B);
% sqrt(var(conca_ward))/sqrt(B);
% sqrt(var(para1))/sqrt(B)];






