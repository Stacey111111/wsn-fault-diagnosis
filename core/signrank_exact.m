function p = signrank_exact(a, b)
%SIGNRANK_EXACT  Two-sided Wilcoxon signed-rank test for paired samples.
%
%   p = signrank_exact(a, b)
%     a, b  paired scores (e.g. macro-F1 of two methods on the same folds)
%     p     two-sided p-value
%
%   Zero differences are dropped and tied |differences| get average ranks.
%   For up to 20 non-zero pairs the p-value is exact (all 2^n sign
%   patterns are counted by dynamic programming over the doubled ranks);
%   for more pairs the normal approximation with continuity correction is
%   used. Base MATLAB only (the toolbox function is signrank).
%
%   Week 10 deliverable, CS 5395 Independent Study (Xiaohui San).

    d = a(:) - b(:);
    d = d(abs(d) > 1e-12);
    n = numel(d);
    if n == 0, p = 1; return; end
    [~, o] = sort(abs(d));
    r = zeros(n, 1);
    ad = abs(d(o));
    i = 1;
    while i <= n                                  % average ranks for ties
        j = i;
        while j < n && abs(ad(j + 1) - ad(i)) < 1e-12, j = j + 1; end
        r(o(i:j)) = (i + j) / 2;
        i = j + 1;
    end
    W = sum(r(d > 0));                            % sum of positive ranks
    if n <= 20
        r2 = round(2 * r);                        % integer (doubled) ranks
        tot = sum(r2);
        cnt = zeros(1, tot + 1);  cnt(1) = 1;     % cnt(s+1) = # sign patterns with sum s
        for k = 1:n
            cnt = cnt + [zeros(1, r2(k)), cnt(1:end - r2(k))];
        end
        prob = cnt / 2 ^ n;
        w2 = round(2 * W);
        lo = sum(prob(1:min(w2, tot - w2) + 1));  % P(W <= min(W, Wmax - W))
        p = min(1, 2 * lo);
    else
        mu = n * (n + 1) / 4;
        sg = sqrt(n * (n + 1) * (2 * n + 1) / 24);
        z = (abs(W - mu) - 0.5) / sg;
        p = erfc(z / sqrt(2));
    end
end
