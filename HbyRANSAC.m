function [H_best, match_list] = HbyRANSAC(feature_i, feature_j, match_list, optimize_option)
pair_i = feature_i(1:2,match_list(1,:));
pair_j = feature_j(1:2,match_list(2,:));

n_match = size(pair_i,2);

pair_i = [pair_i; ones(1,n_match)];
pair_j = [pair_j; ones(1,n_match)];

RANSAC_t = 1.25;
RANSAC_p = 0.99;
RANSAC_s = 4;

n = 4;
inlier_list_best = [];
N = inf;

prev_n_inlier = -1;
sample_count = 0;
while (N > sample_count)
    rand_list = randperm(n_match);
    
    xi = pair_i(:,rand_list(1:n));
    xj = pair_j(:,rand_list(1:n));
    
    n_inlier = 0;
    
    A = [];
    
    for i = 1:n
        A = [A; 0 0 0 (-xi(:,i)') (xj(2,i) * (xi(:,i)'))];
        A = [A; xi(:,i)' 0 0 0 (-xj(1,i) * (xi(:,i)'))];
    end
    
    [~,~,V] = svd(A,0);
    h = V(:,end);
    H = reshape(h, [3 3]);
    H = H';
    
    inlier_list = [];
    for ii = 1:n_match
        xi = pair_i(:,rand_list(ii));
        xj = pair_j(:,rand_list(ii));
        
        xi_t = H*xi;
        xj_t = H\xj;
        
        d1 = sum((xi - xj_t./xj_t(3)) .^ 2);
        d2 = sum((xj - xi_t./xi_t(3)) .^ 2);
        d = d1 + d2;
        
        if d < RANSAC_t
            n_inlier = n_inlier + 1;
            inlier_list = [inlier_list rand_list(ii)];
        end
    end
    
    if (n_inlier > prev_n_inlier)
        prev_n_inlier = n_inlier;
        inlier_list_best = inlier_list;
        H_best = H;
    end
    
    RANSAC_e = 1 - n_inlier/n_match;
    N = (log(1-RANSAC_p)) / (log(1 - (1 - RANSAC_e) ^ RANSAC_s));
    N = abs(N);
    sample_count = sample_count + 1;
end

match_list = match_list(:,inlier_list_best);

% Optimal Homography estimation
if optimize_option == 1
    fprintf('Optimizing...\n');
    options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt','Display','off');
    
    while 1
        n_match = size(match_list,2);
        
        feature_f_match_1 = [feature_i(1:2,match_list(1,:)); ones(1,n_match)];
        feature_f_match_2 = [feature_j(1:2,match_list(2,:)); ones(1,n_match)];
        
        h = H_best';
        h = h(:);
        
        n_inlier = n_match;
        
        xdata = [feature_f_match_1;feature_f_match_2];
        ydata = zeros(1,n_inlier);
        
        % Minimize cost function using Levenberg-Marquardt Algorithm
        fprintf('--Levenberg-Marquardt Algorithm\n');
        H_LM = lsqcurvefit(@symmetric_transfer_dist,h,xdata,ydata,[],[],options);
        Ht = reshape(H_LM,[3,3]);
        Ht = Ht';
        
        % Find more matches using Guided Matching
        fprintf('--Guided Matching\n');
        
        alone_list_1 = [];
        alone_list_2 = [];
        
        for ii = 1:size(feature_i,2)
            if find(match_list(1,:)==ii)
                continue;
            else
                alone_list_1 = [alone_list_1 ii];
            end
        end
        
        alone_point_1 = feature_i(1:2,alone_list_1);
        alone_point_1 = [alone_point_1;ones(1,size(alone_point_1,2))];
        transformed_1 = Ht * alone_point_1;
        transformed_1 = [transformed_1(1,:) ./ transformed_1(3,:);transformed_1(2,:) ./ transformed_1(3,:);transformed_1(3,:) ./ transformed_1(3,:)];
        
        for ii = 1:size(feature_j,2)
            if find(match_list(2,:)==ii)
                continue;
            else
                alone_list_2 = [alone_list_2 ii];
            end
        end
        
        alone_point_2 = feature_j(1:2,alone_list_2);
        alone_point_2 = [alone_point_2;ones(1,size(alone_point_2,2))];
        transformed_2 = Ht \ alone_point_2;
        transformed_2 = [transformed_2(1,:) ./ transformed_2(3,:);transformed_2(2,:) ./ transformed_2(3,:);transformed_2(3,:) ./ transformed_2(3,:)];
        
        new_pair = [];
        for ii = 1:size(alone_point_1,2)
            d_best = inf;
            tmp_pair = [];
            for jj = 1:size(alone_point_2,2)
                xi = alone_point_1(:,ii);
                xj = alone_point_2(:,jj);
                xi_t = transformed_1(:,ii);
                xj_t = transformed_2(:,jj);
                
                d1 = sum((xi - xj_t) .^ 2);
                d2 = sum((xj - xi_t) .^ 2);
                d = d1 + d2;
                if (d < 1.25) && (d < d_best)
                    d_best = d;
                    tmp_pair = [alone_list_1(ii);alone_list_2(jj);d_best];
                end
            end
            new_pair = [new_pair tmp_pair];
        end
        
        H_best = Ht;
        
        if isempty(new_pair)
            fprintf('----No new matches...\n');
            break; % No new matches, so optimization is done
        end
        fprintf('----%d new matches...\n', size(new_pair,2));
        
        match_list = [match_list new_pair(1:2,:)];
    end
    fprintf('->Done Optimizing...\n');
    
elseif optimize_option == 2
    fprintf('Optimizing...\n');
    
    while 1
        n_match = size(match_list,2);
        
        xi = [feature_i(1:2,match_list(1,:)); ones(1,n_match)];
        xj = [feature_j(1:2,match_list(2,:)); ones(1,n_match)];
        
        A = [];
        
        for i = 1:n_match
            A = [A; 0 0 0 (-xi(:,i)') (xj(2,i) * (xi(:,i)'))];
            A = [A; xi(:,i)' 0 0 0 (-xj(1,i) * (xi(:,i)'))];
        end

        fprintf('--DLT Algorithm\n');
        [~,~,V] = svd(A,0);
        h = V(:,end);
        Ht = reshape(h, [3 3]);
        Ht = Ht';
        
        % Find more matches using Guided Matching
        fprintf('--Guided Matching\n');
        
        alone_list_1 = [];
        alone_list_2 = [];
        
        for ii = 1:size(feature_i,2)
            if find(match_list(1,:)==ii)
                continue;
            else
                alone_list_1 = [alone_list_1 ii];
            end
        end
        
        alone_point_1 = feature_i(1:2,alone_list_1);
        alone_point_1 = [alone_point_1;ones(1,size(alone_point_1,2))];
        transformed_1 = Ht * alone_point_1;
        transformed_1 = [transformed_1(1,:) ./ transformed_1(3,:);transformed_1(2,:) ./ transformed_1(3,:);transformed_1(3,:) ./ transformed_1(3,:)];
        
        for ii = 1:size(feature_j,2)
            if find(match_list(2,:)==ii)
                continue;
            else
                alone_list_2 = [alone_list_2 ii];
            end
        end
        
        alone_point_2 = feature_j(1:2,alone_list_2);
        alone_point_2 = [alone_point_2;ones(1,size(alone_point_2,2))];
        transformed_2 = Ht \ alone_point_2;
        transformed_2 = [transformed_2(1,:) ./ transformed_2(3,:);transformed_2(2,:) ./ transformed_2(3,:);transformed_2(3,:) ./ transformed_2(3,:)];
        
        new_pair = [];
        for ii = 1:size(alone_point_1,2)
            d_best = inf;
            tmp_pair = [];
            for jj = 1:size(alone_point_2,2)
                xi = alone_point_1(:,ii);
                xj = alone_point_2(:,jj);
                xi_t = transformed_1(:,ii);
                xj_t = transformed_2(:,jj);
                
                d1 = sum((xi - xj_t) .^ 2);
                d2 = sum((xj - xi_t) .^ 2);
                d = d1 + d2;
                if (d < 1.25) && (d < d_best)
                    d_best = d;
                    tmp_pair = [alone_list_1(ii);alone_list_2(jj);d_best];
                end
            end
            new_pair = [new_pair tmp_pair];
        end
        
        H_best = Ht;
        
        if isempty(new_pair)
            fprintf('----No new matches...\n');
            break; % No new matches, so optimization is done
        end
        fprintf('----%d new matches...\n', size(new_pair,2));
        
        match_list = [match_list new_pair(1:2,:)];
    end
    fprintf('->Done Optimizing...\n');
end