function fid = anms(points, max_point)
n_points = size(points,2);
l = [];
for i = 1:n_points
    minpoint = inf;
    xi = points(:,i);
    for j = 1:n_points
        xj = points(:,j);
        if ((xi(1) ~= xj(1)) && (xi(2) ~= xj(2))) && (xi(3) < 0.9*xj(3))
            d = sqrt((xj(1) - xi(1))^2 + (xj(2) - xi(2))^2);
            if d < minpoint
                minpoint = d;
            end
        end
    end
    l = [l [i;minpoint]];
end
[~,sort_idx] = sort(l(2,:),'descend');
l = l(:,sort_idx);
fid = l(1,:);

max_point = min(max_point,size(fid,2));
fid = fid(1:max_point);