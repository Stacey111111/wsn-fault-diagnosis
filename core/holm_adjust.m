function pAdj = holm_adjust(p)
%HOLM_ADJUST  Holm-Bonferroni adjusted p-values for multiple comparisons.
%
%   pAdj = holm_adjust(p)
%     p     vector of raw p-values of m tests
%     pAdj  adjusted p-values (same order); reject a hypothesis when its
%           adjusted p-value is below the chosen level (e.g. 0.05)
%
%   Week 10 deliverable, CS 5395 Independent Study (Xiaohui San).

    m = numel(p);
    [ps, o] = sort(p(:));
    a = min(1, (m - (1:m)' + 1) .* ps);
    a = cummax(a);
    pAdj = zeros(size(p));
    pAdj(o) = a;
end
