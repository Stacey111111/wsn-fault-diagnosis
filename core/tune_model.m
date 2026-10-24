function [best, T] = tune_model(X, y, groups, name, opts)
%TUNE_MODEL  Grid search of a model's hyperparameters by grouped CV.
%
%   [best, T] = tune_model(X, y, groups, name, opts)
%     opts.seed    fold-assignment seed used for tuning (default 99, which
%                  is different from the seeds of the final evaluation)
%     opts.nFolds  folds (default 5)
%     best         hyperparameter struct with the highest mean macro-F1
%     T            struct array: hp, text, macroF1 (mean over folds), sd
%
%   Each setting of hp_grid(name) is scored by one round of grouped 5-fold
%   CV with its own fold assignment. The final numbers come from
%   cv_evaluate with other fold assignments, but the networks are the same,
%   so the tuned scores are slightly optimistic (see the report).
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 5, opts = struct(); end
    seed = 99;  nFolds = 5;
    if isfield(opts, 'seed'), seed = opts.seed; end
    if isfield(opts, 'nFolds'), nFolds = opts.nFolds; end
    G = hp_grid(name);
    T = struct('hp', {}, 'text', {}, 'macroF1', {}, 'sd', {});
    for g = 1:numel(G)
        E = cv_evaluate(X, y, groups, name, G(g), struct('seeds', seed, 'nFolds', nFolds));
        T(g) = struct('hp', G(g), 'text', hp_string(G(g)), 'macroF1', mean(E.foldF1), ...
                      'sd', std(E.foldF1));
        fprintf('    %-40s macro-F1 %.3f (sd %.3f)\n', T(g).text, T(g).macroF1, T(g).sd);
    end
    [~, b] = max([T.macroF1]);
    best = G(b);
end
