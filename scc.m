function [class_id] = scc(X,y,type,K,Z)

if nargin == 4  % no covariate
    if type == "gaussian"
        [ class_id ] = scc_gaus( X,y,K );
    elseif type == "binary"
        [ class_id ] = scc_binary( X,y,K );
    elseif type == "categorical"
        [ class_id ] = scc_category( X,y,K );
    elseif type == "count"
        [ class_id ] = scc_count( X,y,K );
    elseif type == "survival"
        [ class_id ] = scc_surv( X,y,K);
    end
end

if nargin == 5 % with covariate
    if type == "gaussian"
        [ class_id ] = scc_gaus_with_cov( X,y,Z,K );
    elseif type == "binary"
        [ class_id ] = scc_binary_with_cov( X,y,Z,K );
    elseif type == "categorical"
        [ class_id ] = scc_category_with_cov( X,y,Z,K );
    elseif type == "count"
        [ class_id ] = scc_count_with_cov( X,y,Z,K );
    elseif type == "survival"
        [ class_id ] = scc_surv_with_cov( X,y,Z,K );
    end
end




end

