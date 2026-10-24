function S = score_model(mdl, X, classes)
%SCORE_MODEL  Class scores of a model from train_model, one column per class.
%
%   S = score_model(mdl, X, classes)
%     mdl      model from train_model (or a bagged model from bag_train)
%     X        n x p features
%     classes  1 x K class labels; column c of S belongs to classes(c)
%     S        n x K scores; for every model a larger score means "more
%              likely this class", and the predicted class is the column
%              with the largest score:
%                tree, k-NN, random forest, neural network: posterior
%                  probabilities
%                SVM, LogitBoost, GentleBoost (fitcecoc): negative average
%                  binary loss
%                AdaBoost.M2, RUSBoost: ensemble scores
%                bagged models: share of the bagged models voting for the class
%
%   A class that was absent from the training data gets the lowest score
%   of the row. Used by StackingEnsemble (Week 13).
%
%   Week 13 deliverable, CS 5395 Independent Study (Xiaohui San).

    n = size(X, 1);  K = numel(classes);
    if isstruct(mdl) && isfield(mdl, 'type') && strcmp(mdl.type, 'bag')
        S = zeros(n, K);
        for b = 1:numel(mdl.models)
            yp = predict_model(mdl.models{b}, X);
            for c = 1:K, S(:, c) = S(:, c) + (yp == classes(c)); end
        end
        S = S / numel(mdl.models);
        return
    end
    [~, s] = predict(mdl, X);
    cn = mdl.ClassNames;
    if iscell(cn), cn = str2double(cn); end
    cn = double(cn(:))';
    S = nan(n, K);
    for c = 1:K
        j = find(cn == classes(c), 1);
        if ~isempty(j), S(:, c) = s(:, j); end
    end
    if any(isnan(S(:)))
        lo = min(S, [], 2, 'omitnan');
        [r, c] = find(isnan(S));
        S(sub2ind(size(S), r, c)) = lo(r);
    end
end
