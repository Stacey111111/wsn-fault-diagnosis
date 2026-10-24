function T = paired_tests(S, names, pairs)
%PAIRED_TESTS  Wilcoxon signed-rank tests with Holm correction for chosen
%   pairs of methods scored on the same test folds.
%
%   T = paired_tests(S, names, pairs)
%     S      b x k fold scores (rows = test folds, columns = methods)
%     names  1 x k method names
%     pairs  m x 2 column indices of S to compare (default: all pairs)
%     T      struct array with fields a, b, meanDiff (a - b), wins (folds
%            where a > b), p (exact two-sided p), pHolm, significant
%            (pHolm < 0.05)
%
%   Uses signrank_exact and holm_adjust (Week 10). The folds come from
%   repeats on the same networks, so the p-values are read as a ranking of
%   clear and unclear gaps, not as tests on independent data sets.
%
%   Week 11 deliverable, CS 5395 Independent Study (Xiaohui San).

    k = size(S, 2);
    if nargin < 3 || isempty(pairs), pairs = nchoosek(1:k, 2); end
    m = size(pairs, 1);
    p = zeros(m, 1);
    for q = 1:m
        p(q) = signrank_exact(S(:, pairs(q, 1)), S(:, pairs(q, 2)));
    end
    pH = holm_adjust(p);
    T = struct('a', {}, 'b', {}, 'meanDiff', {}, 'wins', {}, 'p', {}, 'pHolm', {}, 'significant', {});
    for q = 1:m
        a = pairs(q, 1);  b = pairs(q, 2);
        T(q) = struct('a', names{a}, 'b', names{b}, 'meanDiff', mean(S(:, a) - S(:, b)), ...
                      'wins', sum(S(:, a) > S(:, b)), 'p', p(q), 'pHolm', pH(q), ...
                      'significant', pH(q) < 0.05);
    end
end
