function G = hp_grid(name)
%HP_GRID  Hyperparameter grid searched for each model.
%
%   G = hp_grid(name)  returns a struct array, one element per setting.
%
%   svm   C in {1, 10, 100} x kernel scale in {2, 5, 10}       (9 settings)
%   tree  min leaf size in {1, 5, 10, 20} x max splits in
%         {10, 20, 50, 100, 500}                                (20)
%   knn   k in {1, 3, 5, 9, 15, 25} x weight in {equal, inverse} (12)
%   rf    200 trees, features per split (mtry) in {3, 5, 8, 12} x
%         min leaf size in {1, 5}                               (8)
%   nn    layers in {[20], [50], [50 20]} x lambda in {1e-4, 1e-3, 1e-2} (9)
%   adaboost, rusboost, gentleboost, logitboost (Week 12)
%         200 rounds, tree size (max splits) in {5, 20} x learning rate
%         in {0.1, 1}                                           (4 each)
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    G = [];
    switch lower(name)
        case 'svm'
            for C = [1 10 100]
                for ks = [2 5 10]
                    G = [G, struct('C', C, 'kernelScale', ks)]; %#ok<AGROW>
                end
            end
        case 'tree'
            for ml = [1 5 10 20]
                for ms = [10 20 50 100 500]
                    G = [G, struct('minLeaf', ml, 'maxSplits', ms)]; %#ok<AGROW>
                end
            end
        case 'knn'
            for k = [1 3 5 9 15 25]
                for w = {'equal', 'inverse'}
                    G = [G, struct('k', k, 'weight', w{1})]; %#ok<AGROW>
                end
            end
        case 'rf'
            for m = [3 5 8 12]
                for ml = [1 5]
                    G = [G, struct('nTrees', 200, 'mtry', m, 'minLeaf', ml)]; %#ok<AGROW>
                end
            end
        case 'nn'
            L = {20, 50, [50 20]};
            for i = 1:numel(L)
                for lam = [1e-4 1e-3 1e-2]
                    G = [G, struct('layers', L{i}, 'lambda', lam)]; %#ok<AGROW>
                end
            end
        case {'adaboost', 'rusboost', 'gentleboost', 'logitboost'}
            for ms = [5 20]
                for lr = [0.1 1]
                    G = [G, struct('nLearn', 200, 'learnRate', lr, 'maxSplits', ms)]; %#ok<AGROW>
                end
            end
        otherwise
            error('hp_grid:name', 'Unknown model "%s".', name);
    end
end
