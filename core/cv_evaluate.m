function E = cv_evaluate(X, y, groups, name, hp, opts)
%CV_EVALUATE  Repeated grouped cross-validation of one model setting.
%
%   E = cv_evaluate(X, y, groups, name, hp, opts)
%     X, y     features and numeric labels
%     groups   group (network) of each window, see grouped_folds
%     name,hp  model and hyperparameters, see train_model
%     opts     struct with optional fields
%              nFolds   folds per repeat (default 5)
%              seeds    one fold-assignment seed per repeat (default 1:3)
%              classes  class labels (default 0:8)
%              verbose  print one line per fit (default false)
%
%   E.fold      n x R fold number of each window in each repeat
%   E.yPred     n x R test-set prediction of each window in each repeat
%   E.foldF1    (nFolds*R) x 1 macro-F1 of each test fold
%   E.foldAcc   (nFolds*R) x 1 accuracy of each test fold
%   E.metrics   class_metrics of all test predictions pooled over the
%               repeats (the confusion counts are divided by R)
%   E.trainTime, E.predTime   seconds per fit / per prediction call
%   E.msPerWindow             mean prediction time per window [ms]
%   E.modelBytes              size of the last trained model (compact
%                             form, without training data where possible)
%
%   Every window is predicted once per repeat by a model that did not see
%   any window of its network. The random seed of each fit is fixed, so the
%   results can be reproduced.
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 6, opts = struct(); end
    d = struct('nFolds', 5, 'seeds', 1:3, 'classes', 0:8, 'verbose', false);
    f = fieldnames(opts);
    for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end
    n = numel(y);  R = numel(d.seeds);  K = d.nFolds;
    E = struct('name', name, 'hp', hp, 'fold', zeros(n, R), 'yPred', nan(n, R), ...
               'foldF1', zeros(K * R, 1), 'foldAcc', zeros(K * R, 1), ...
               'trainTime', zeros(K * R, 1), 'predTime', zeros(K * R, 1));
    st = rng();
    j = 0;
    for r = 1:R
        fold = grouped_folds(groups, K, d.seeds(r));
        E.fold(:, r) = fold;
        for k = 1:K
            j = j + 1;
            te = fold == k;  tr = ~te;
            rng(1000 * d.seeds(r) + k);                % reproducible fit
            t0 = tic;
            mdl = train_model(name, X(tr, :), y(tr), hp);
            E.trainTime(j) = toc(t0);
            t0 = tic;
            yp = predict_model(mdl, X(te, :));
            E.predTime(j) = toc(t0);
            E.yPred(te, r) = yp;
            M = class_metrics(y(te), yp, d.classes);
            E.foldF1(j) = M.macroF1;  E.foldAcc(j) = M.accuracy;
            if d.verbose
                fprintf('    repeat %d fold %d: macro-F1 %.3f (train %.1f s)\n', r, k, ...
                        M.macroF1, E.trainTime(j));
            end
        end
    end
    rng(st);
    M = class_metrics(repmat(y(:), R, 1), E.yPred(:), d.classes);
    M.confusion = M.confusion / R;
    E.metrics = M;
    E.msPerWindow = 1e3 * sum(E.predTime) / (n * R);
    E.modelBytes = model_bytes(mdl);
end

function b = model_bytes(mdl)
%MODEL_BYTES  Memory size of the compact model.
    try
        c = compact(mdl);
    catch
        c = mdl;
    end
    w = whos('c');
    b = w.bytes;
end
