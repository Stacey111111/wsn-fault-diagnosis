function S = select_features(X, y, rMax, names)
%SELECT_FEATURES  Rank features by the one-way ANOVA F statistic and drop
%   features that are strongly correlated with a better-ranked one.
%
%   S = select_features(X, y, rMax, names)
%     X      n x P feature matrix
%     y      n x 1 class labels
%     rMax   largest |Pearson correlation| allowed between two kept
%            features (default 0.9)
%     names  feature names (optional, used in S.dropped)
%
%   S.F         1 x P F statistic (between-class variance / within-class
%               variance of the feature; larger = separates classes better)
%   S.order     feature indices from the highest to the lowest F
%   S.keep      indices of the kept features, best first
%   S.dropped   struct array: feature, correlatedWith, r
%   S.corr      P x P correlation matrix of the features
%
%   Greedy filter: walk down the F ranking and keep a feature only if its
%   |r| with every feature kept so far is at most rMax. A feature with zero
%   variance is dropped. Computed with base MATLAB only.
%
%   Week 8 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 3 || isempty(rMax), rMax = 0.9; end
    P = size(X, 2);
    if nargin < 4 || isempty(names), names = arrayfun(@(j) sprintf('f%d', j), 1:P, 'UniformOutput', false); end
    cls = unique(y(:))';
    n = size(X, 1);  K = numel(cls);
    mu = mean(X, 1);
    ssb = zeros(1, P);  ssw = zeros(1, P);
    for c = cls
        Xc = X(y == c, :);
        ssb = ssb + size(Xc, 1) * (mean(Xc, 1) - mu) .^ 2;
        ssw = ssw + sum((Xc - mean(Xc, 1)) .^ 2, 1);
    end
    F = (ssb / (K - 1)) ./ max(ssw / (n - K), eps);
    sd = std(X, 0, 1);
    F(sd < 1e-12) = 0;
    [~, order] = sort(F, 'descend');

    Xs = (X - mu) ./ max(sd, eps);
    C = (Xs' * Xs) / (n - 1);
    keep = [];
    dropped = struct('feature', {}, 'correlatedWith', {}, 'r', {});
    for j = order
        if sd(j) < 1e-12, continue; end
        if isempty(keep)
            keep = j;  continue;
        end
        [rm, k] = max(abs(C(j, keep)));
        if rm > rMax
            dropped(end + 1) = struct('feature', names{j}, 'correlatedWith', names{keep(k)}, ...
                                      'r', C(j, keep(k))); %#ok<AGROW>
        else
            keep(end + 1) = j; %#ok<AGROW>
        end
    end
    S = struct('F', F, 'order', order, 'keep', keep, 'dropped', dropped, 'corr', C);
end
