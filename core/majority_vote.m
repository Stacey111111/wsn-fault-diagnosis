function [yv, share] = majority_vote(P, classes, tieBreak)
%MAJORITY_VOTE  Plurality vote over the columns of a prediction matrix.
%
%   [yv, share] = majority_vote(P, classes, tieBreak)
%     P         n x M predicted labels of M models
%     classes   1 x C class labels (default: unique values of P)
%     tieBreak  n x 1 labels used when several classes have the most votes
%               (default: the prediction of the first model, P(:, 1))
%     yv        n x 1 voted label
%     share     n x 1 fraction of the models that voted for yv
%
%   Week 11 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 2 || isempty(classes), classes = unique(P(:))'; end
    if nargin < 3 || isempty(tieBreak), tieBreak = P(:, 1); end
    [n, M] = size(P);
    C = numel(classes);
    cnt = zeros(n, C);
    for c = 1:C
        cnt(:, c) = sum(P == classes(c), 2);
    end
    [mx, j] = max(cnt, [], 2);
    yv = classes(j)';
    tie = sum(cnt == mx, 2) > 1;                    % several classes share the top count
    if any(tie)
        tb = tieBreak(:);
        % keep the tie-break label only if it is one of the tied classes
        for i = find(tie)'
            ci = find(classes == tb(i), 1);
            if ~isempty(ci) && cnt(i, ci) == mx(i), yv(i) = tb(i); end
        end
    end
    share = mx / M;
end
