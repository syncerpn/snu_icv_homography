%%                      HOMOGRAPHY
% |===================================================|
% | Nguyen Tuan Nghia                                 |
% | nghianguyentuan@snu.ac.kr                         |
% | Seoul National University                         |
% | Department of Electrical and Computer Engineering |
% |===================================================|
%
%% SETTING UP:
close all;
clear;
clc;
        
% Optimization options
% Please DO NOT change this
option_none = 0; % No optimization
option_lema = 1; % Levenberg-Marquadt 
option_dlto = 2; % Direct Linear Transform

% Change this one
optimz_option = option_lema;

%% PREPROCESSING AND FEATURE EXTRACTION:
image_dir = '../../images/'; % Specify image set directory

% List of images to be loaded

% ------------------------PLEASE CHANGE THE IMAGE LIST TO FIT THE TEST------------------------ %
% -------------------I do not want to get penalty because of small faults :(------------------ %

% image_list = {'nghia1.jpg';'nghia2.jpg';'nghia3.jpg';'nghia4.jpg';'nghia5.jpg'}; % My images

image_list = {'img1.bmp';'img2.bmp';'img3.bmp';'img4.bmp';'img5.bmp'}; % Your images

% -------------------------------------------------------------------------------------------- %

n_image = size(image_list,1); % Count the number of images

images_ori = cell(n_image,1); % Color images should be kept for later use
images = cell(n_image,1); % Gray images will be stored here

feature_f = cell(n_image,1);
feature_d = cell(n_image,1);

for i = 1:n_image
    images_ori{i} = imread(strcat(image_dir, image_list{i})); % Load image
    images_ori{i} = imresize(images_ori{i},[256 256]); % Resize image
    images{i} = single(rgb2gray(images_ori{i})); % Convert to gray image
    
    % Extract SIFT feature
    [f,d] = vl_sift(images{i},'FirstOctave',-1);
    
    % Remove duplicated points
    [~,sorted_idx] = sort(f(1,:));
    f = f(:,sorted_idx);
    d = d(:,sorted_idx);
    
    filtered_idx = 1;
    for j = 2:size(f,2)
        if ((f(1,j) - f(1,filtered_idx(end)) == 0) && (f(2,j) - f(2,filtered_idx(end)) == 0))
            continue;
        end
        filtered_idx = [filtered_idx, j];
    end
    
    feature_f{i} = f(:,filtered_idx);
    feature_d{i} = d(:,filtered_idx);
    
    % Apply Non-maximum Suppression to get better distribution
    fid = anms(feature_f{i},300); % Keep upto 300 points
    feature_f{i} = feature_f{i}(:,fid);
    feature_d{i} = feature_d{i}(:,fid);
end

[h_ori, w_ori] = size(images{1}); % Get image size

%% FEATURE MATCHING:
match = cell(n_image-1,1); % Matches (index) will be stored here
H = cell(n_image-1,1); % Homography matrix will be stored here
d_sum = cell(n_image-1,1); % Distance measurement will be stored here

for i = 1:n_image-1
    fprintf('Processing Image Pair: %d and %d\n',i,i+1);
    
    fprintf('Matching Key Points\n');
    
    [match{i}, score] = vl_ubcmatch(feature_d{i},feature_d{i+1});
    
    % Remove one-point-multiple-match matches
    [score, sorted_idx] = sort(score);
    match{i} = match{i}(:,sorted_idx);
    for fil_dim = 1:2
        [~,sid] = sort(match{i}(fil_dim,:));
        match{i} = match{i}(:,sid);
        filtered_idx = 1;
        for j = 2:size(match{i},2)
            if match{i}(fil_dim,j) == match{i}(fil_dim,filtered_idx(end))
                continue;
            end
            filtered_idx = [filtered_idx j];
        end
        match{i} = match{i}(:,filtered_idx);
    end
    
    % Homography estimation using RANSAC and DLT
    fprintf('Estimating H using RANSAC\n');
    [H{i}, match{i}] = HbyRANSAC(feature_f{i}, feature_f{i+1},match{i}, optimz_option);
    n_match = size(match{i},2);
    
    % Calculate total distance over all matches
    fprintf('H%d%d = \n',i,i+1);
    disp(H{i});
    feature_f_match_1 = [feature_f{i}(1:2,match{i}(1,:)); ones(1,n_match)];
    feature_f_match_2 = [feature_f{i+1}(1:2,match{i}(2,:)); ones(1,n_match)];
    xdata = [feature_f_match_1;feature_f_match_2];
    ydata = symmetric_transfer_dist(H{i}', xdata);
    d_sum{i} = sum(ydata);
    fprintf('->Total transfer error: %f for %d matches\n\n',d_sum{i},n_match);
end

%% RESULTS:
% Plot matches between two images
for i = 1:n_image-1
    img_merge = [images_ori{i} images_ori{i+1}];
    figure();
    imshow(img_merge);
    hold on;
    for j = 1:size(match{i},2)
        idx_1 = match{i}(1,j);
        idx_2 = match{i}(2,j);
        x = [feature_f{i}(1,idx_1) feature_f{i+1}(1,idx_2)+w_ori];
        y = [feature_f{i}(2,idx_1) feature_f{i+1}(2,idx_2)];
        plot(x,y,'-o','color','y','LineWidth',1)
    end
    
    for j = 1:size(feature_f{i},2)
        if (find(match{i}(1,:) == j))
            continue;
        end
        x = feature_f{i}(1,j);
        y = feature_f{i}(2,j);
        plot(x,y,'-x','color','r','LineWidth',1)
    end
    
    for j = 1:size(feature_f{i+1},2)
        if (find(match{i}(2,:) == j))
            continue;
        end
        x = feature_f{i+1}(1,j)+w_ori;
        y = feature_f{i+1}(2,j);
        plot(x,y,'-x','color','r','LineWidth',1)
    end
    tt = sprintf('Image %d and %d, total transfer error: %f over %d matches',i,i+1,d_sum{i},size(match{i},2));
    title(tt);
    drawnow();
    hold off;
end

%% CONSTRUCTING PANORAMA:
% Get panorama view, including projection plane and size
center_img = floor(n_image/2)+1;

base_ul = [1;1];
base_ur = [w_ori;1];
base_br = [w_ori;h_ori];
base_bl = [1;h_ori];

point_list = [];

dest_ul = [base_ul;1];
dest_ur = [base_ur;1];
dest_br = [base_br;1];
dest_bl = [base_bl;1];

for i = 1 : center_img-1
    
    dest_ul = H{i} * dest_ul;
    dest_ur = H{i} * dest_ur;
    dest_br = H{i} * dest_br;
    dest_bl = H{i} * dest_bl;
    
    point_list = [point_list ...
        dest_ul(1:2)/dest_ul(end) ...
        dest_ur(1:2)/dest_ur(end) ...
        dest_br(1:2)/dest_br(end) ...
        dest_bl(1:2)/dest_bl(end)];
end

point_list = [point_list base_ul base_ur base_br base_bl];

dest_ul = [base_ul;1];
dest_ur = [base_ur;1];
dest_br = [base_br;1];
dest_bl = [base_bl;1];

for i = center_img : n_image-1
    
    dest_ul = H{i} \ dest_ul;
    dest_ur = H{i} \ dest_ur;
    dest_br = H{i} \ dest_br;
    dest_bl = H{i} \ dest_bl;
    
    point_list = [point_list ...
        dest_ul(1:2)/dest_ul(end) ...
        dest_ur(1:2)/dest_ur(end) ...
        dest_br(1:2)/dest_br(end) ...
        dest_bl(1:2)/dest_bl(end)];
end

panorama_ul = [min(point_list(1,:)); min(point_list(2,:))];
panorama_br = [max(point_list(1,:)); max(point_list(2,:))];

x_range = [panorama_ul(1) panorama_br(1)];
y_range = [panorama_ul(2) panorama_br(2)];

sizes = ceil(panorama_br - panorama_ul);
width = sizes(1);
height = sizes(2);

% Create panorama
panorama = zeros(height,width,3,'uint8');
panorama_view = imref2d([height width], x_range, y_range);
% blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

warped = cell(n_image,1); % Warped images
mask = cell(n_image,1); % Masks for blending
for i = 1 : n_image
    % Forward projection
    if i < center_img
        H_c = H{i};
        for j = i+1 : center_img-1
            H_c = H{j} * H_c; % Chain the Homography matrices
        end
        [warped{i}, mask{i}] = warping(images_ori{i}, H_c, panorama_view);
        panorama = uint8(~mask{i}).*panorama + warped{i};
%         panorama = step(blender, panorama, warped{i}, mask{i});

    % Center image projection
    elseif (i == center_img)
        [warped{i}, mask{i}] = warping(images_ori{i}, eye(3), panorama_view);
        panorama = uint8(~mask{i}).*panorama + warped{i};
%         panorama = step(blender, panorama, warped{i}, mask{i});
        
    % Backward projection
    else
        H_c = H{center_img} \ eye(3);
        for j = center_img+1 : i-1
            H_c = H{j} \ H_c; % Chain the Homography matrices
        end
        [warped{i}, mask{i}] = warping(images_ori{i}, H_c, panorama_view);
        panorama = uint8(~mask{i}).*panorama + warped{i};
%         panorama = step(blender, panorama, warped{i}, mask{i});
    end
end

%% Show result
figure();
imshow(panorama);

if (optimz_option == option_lema)
    opti_string = 'Levenberg-Marquardt';
elseif (optimz_option == option_dlto)
    opti_string = 'Direct Linear Transform';
else
    opti_string = 'None';
end

tt = sprintf('Panorama image ~ Optimization: %s',opti_string);
title(tt);
