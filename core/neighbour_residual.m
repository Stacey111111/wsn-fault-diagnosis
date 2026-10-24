function [res, nbMed, netMed] = neighbour_residual(R)
%NEIGHBOUR_RESIDUAL  Reading minus the median of the neighbours' readings.
%
%   [res, nbMed, netMed] = neighbour_residual(R)
%     R       one run (fields reading T x N, neighbors N x N)
%     res     T x N, reading - nbMed (NaN if either is missing)
%     nbMed   T x N, NaN-aware median of the readings of the node's
%             neighbours (nodes within the 30 m radio range)
%     netMed  T x 1, median of all readings received in that period
%
%   Only data the sink has are used (received readings). The median is
%   robust to one or two faulty neighbours.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    [T, N] = size(R.reading);
    nbMed = nan(T, N);
    for i = 1:N
        nb = find(R.neighbors(i, :));
        nb(nb == i) = [];
        if ~isempty(nb), nbMed(:, i) = rowmedian(R.reading(:, nb)); end
    end
    res = R.reading - nbMed;
    netMed = rowmedian(R.reading);
end
