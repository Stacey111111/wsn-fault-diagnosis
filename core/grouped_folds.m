function fold = grouped_folds(groups, nFolds, seed)
%GROUPED_FOLDS  Cross-validation folds that keep each group together.
%
%   fold = grouped_folds(groups, nFolds, seed)
%     groups  n x 1 group of each sample (here: the network / run number)
%     nFolds  number of folds (default 5)
%     seed    random seed of the assignment (default 1)
%     fold    n x 1 fold number (1..nFolds) of each sample
%
%   All windows of one simulated network go to the same fold, so a model is
%   always tested on networks it has never seen. Windows of one network are
%   strongly related (same layout, same faults, same weather), and splitting
%   them between training and test sets would overstate the accuracy.
%   Groups are shuffled and dealt to the folds in turn (10 networks and
%   5 folds give 2 networks per fold).
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 2 || isempty(nFolds), nFolds = 5; end
    if nargin < 3 || isempty(seed), seed = 1; end
    g = unique(groups(:));
    st = rng();  rng(seed);
    g = g(randperm(numel(g)));
    rng(st);
    fold = zeros(numel(groups), 1);
    for k = 1:numel(g)
        fold(groups == g(k)) = mod(k - 1, nFolds) + 1;
    end
end
