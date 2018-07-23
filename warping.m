function [warped_img, mask] = warping(img_i, H, view)
tform = projective2d(H');
warped_img = imwarp(img_i, tform, 'OutputView',view);
mask = imwarp(true(size(img_i)), tform, 'OutputView', view);