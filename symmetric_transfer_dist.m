function ydata = symmetric_transfer_dist(x, xdata)
H = reshape(x,[3,3]);
H = H';

n_pair = size(xdata,2);
ydata = [];
for i = 1:n_pair
    xi = xdata(1:3,i);
    xj = xdata(4:6,i);

    xi_t = H*xi;
    xj_t = H\xj;

    d1 = sum((xi - xj_t./xj_t(3)) .^ 2);
    d2 = sum((xj - xi_t./xi_t(3)) .^ 2);
    d = d1 + d2;
    ydata = [ydata d];
end