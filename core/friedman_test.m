function [p, chi2, meanRank] = friedman_test(S)
%FRIEDMAN_TEST  Friedman test for b blocks (rows) and k methods (columns).
%
%   [p, chi2, meanRank] = friedman_test(S)
%     S         b x k scores (here: macro-F1 of k methods on b test folds;
%               larger = better)
%     p         p-value of the null hypothesis that all methods perform
%               the same (chi-square approximation, k - 1 degrees of freedom)
%     chi2      Friedman statistic (corrected for ties)
%     meanRank  1 x k mean rank of each method (1 = best)
%
%   Base MATLAB implementation (the toolbox function friedman gives the same
%   statistic); the chi-square tail comes from gammainc.
%
%   Week 10 deliverable, CS 5395 Independent Study (Xiaohui San).

    [b, k] = size(S);
    Rk = zeros(b, k);
    for i = 1:b
        Rk(i, :) = tiedrank_row(-S(i, :));      % rank 1 = largest score
    end
    meanRank = mean(Rk, 1);
    Rj = sum(Rk, 1);
    % tie correction
    T = 0;
    for i = 1:b
        [~, ~, ic] = unique(S(i, :));
        t = accumarray(ic(:), 1);
        T = T + sum(t .^ 3 - t);
    end
    chi2 = (12 / (b * k * (k + 1)) * sum(Rj .^ 2) - 3 * b * (k + 1)) / ...
           max(1 - T / (b * (k ^ 3 - k)), eps);
    p = 1 - gammainc(chi2 / 2, (k - 1) / 2);
end

function r = tiedrank_row(v)
%TIEDRANK_ROW  Ranks 1..n of a vector, ties get their average rank.
    [sv, o] = sort(v);
    n = numel(v);
    r = zeros(1, n);
    i = 1;
    while i <= n
        j = i;
        while j < n && sv(j + 1) == sv(i), j = j + 1; end
        r(o(i:j)) = (i + j) / 2;
        i = j + 1;
    end
end
