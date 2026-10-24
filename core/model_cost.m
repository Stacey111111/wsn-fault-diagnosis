function C = model_cost(name, mdl, X, hp)
%MODEL_COST  Arithmetic operations and memory needed to classify one window
%   with a trained model, as it would be implemented on a sensor node.
%
%   C = model_cost(name, mdl, X, hp)
%     name  model name of train_model ('tree', 'svm', 'knn', 'rf', 'nn',
%           'adaboost', 'rusboost', 'gentleboost', 'logitboost')
%     mdl   the trained model
%     X     sample of feature vectors; the decision paths of the trees are
%           measured on these windows
%     hp    hyperparameters of the model (only hp.maxSplits of the boosted
%           models is used, as a depth bound if a tree cannot be read)
%     C     struct with fields
%           ops        operations per window (one comparison, addition or
%                      multiplication = 1 operation; exp = 20)
%           bytes      memory of the model in an embedded implementation
%                      (32-bit numbers; tree node = 8 bytes: feature index,
%                      threshold, child index)
%           nTrees     number of trees (tree models)
%           meanDepth  mean number of comparisons per tree on X
%           nNodes     total number of tree nodes
%           nSV        number of stored support vectors / training windows
%           depthBound true if a tree's structure could not be read and its
%                      depth was replaced by the upper bound MaxNumSplits
%
%   Operation counts (p features, K classes):
%     tree        comparisons along the decision path
%     rf          sum of the paths of all trees + one vote per tree
%     adaboost    sum of the paths + one weighted vote per tree
%     logitboost  K one-vs-all ensembles: sum of the paths + one addition
%                 per tree, for all K ensembles
%     nn          standardisation (2p) + 2 x weights + 2 x neurons (bias, ReLU)
%     svm         standardisation + for every support vector of every binary
%                 SVM: squared distance (3p) + exp (20) + weighted sum (2)
%     knn         standardisation + distance to every stored window (3p) +
%                 one comparison per stored window
%   Feature extraction is the same for all models and is not included.
%
%   Week 14 deliverable, CS 5395 Independent Study (Xiaohui San).

    p = size(X, 2);
    if nargin < 4, hp = struct(); end
    ms = NaN;
    if isfield(hp, 'maxSplits'), ms = hp.maxSplits; end
    C = struct('ops', 0, 'bytes', 0, 'nTrees', 0, 'meanDepth', NaN, 'nNodes', 0, 'nSV', 0, ...
               'depthBound', false);
    switch lower(name)
        case 'tree'
            [d, nn] = tree_stats(mdl, X, NaN);
            C.nTrees = 1;  C.meanDepth = d;  C.nNodes = nn;
            C.ops = d;  C.bytes = 8 * nn;
        case 'rf'
            T = mdl.NumTrees;  ds = zeros(T, 1);  ns = zeros(T, 1);
            for t = 1:T, [ds(t), ns(t)] = tree_stats(mdl.Trees{t}, X, NaN); end
            C.nTrees = T;  C.meanDepth = mean(ds);  C.nNodes = sum(ns);
            C.ops = sum(ds) + T;  C.bytes = 8 * sum(ns);
            C.perTreeOps = ds + 1;  C.perTreeBytes = 8 * ns;
        case {'adaboost', 'rusboost'}
            T = mdl.NumTrained;  ds = zeros(T, 1);  ns = zeros(T, 1);
            for t = 1:T, [ds(t), ns(t), b] = tree_stats(mdl.Trained{t}, X, ms); C.depthBound = C.depthBound || b; end
            C.nTrees = T;  C.meanDepth = mean(ds);  C.nNodes = sum(ns);
            C.ops = sum(ds) + T;  C.bytes = 8 * sum(ns) + 4 * T;
            C.perTreeOps = ds + 1;  C.perTreeBytes = 8 * ns + 4;
        case {'gentleboost', 'logitboost'}
            L = mdl.BinaryLearners;  dsum = 0;  nsum = 0;  T = 0;  dl = [];
            for c = 1:numel(L)
                for t = 1:L{c}.NumTrained
                    [d, nn, b] = tree_stats(L{c}.Trained{t}, X, ms);
                    dsum = dsum + d;  nsum = nsum + nn;  T = T + 1;  dl(end + 1) = d; %#ok<AGROW>
                    C.depthBound = C.depthBound || b;
                end
            end
            C.nTrees = T;  C.meanDepth = mean(dl);  C.nNodes = nsum;
            C.ops = dsum + T;  C.bytes = 8 * nsum + 4 * T;
        case 'nn'
            ops = 2 * p;  nw = 0;
            for l = 1:numel(mdl.LayerWeights)
                Wl = mdl.LayerWeights{l};
                ops = ops + 2 * numel(Wl) + 2 * size(Wl, 1);
                nw = nw + numel(Wl) + size(Wl, 1);
            end
            C.ops = ops;  C.bytes = 4 * (nw + 2 * p);
        case 'svm'
            L = mdl.BinaryLearners;  nsv = 0;  ops = 2 * p;
            for c = 1:numel(L)
                s = size(L{c}.SupportVectors, 1);
                nsv = nsv + s;  ops = ops + s * (3 * p + 20 + 2) + 1;
            end
            C.nSV = nsv;  C.ops = ops + numel(L);  C.bytes = 4 * (nsv * (p + 1) + 2 * p);
        case 'knn'
            nTr = size(mdl.X, 1);
            C.nSV = nTr;  C.ops = 2 * p + nTr * (3 * p + 1) + mdl.NumNeighbors;
            C.bytes = 4 * (nTr * (p + 1) + 2 * p);
        otherwise
            error('model_cost:name', 'No cost model for "%s".', name);
    end
end

function [d, nNodes, bound] = tree_stats(t, X, maxSplits)
%TREE_STATS  Mean decision-path length on X and number of nodes of a tree.
%   Works for classification and regression trees; for a wrapped tree whose
%   structure cannot be read, the path length is bounded by maxSplits and
%   the number of nodes by 2 * maxSplits + 1.
    bound = false;
    tr = unwrap(t);
    if isempty(tr)
        bound = true;  d = maxSplits;  nNodes = 2 * maxSplits + 1;
        return
    end
    ch = tr.Children;  nNodes = size(ch, 1);
    depth = zeros(nNodes, 1);
    for i = 1:nNodes
        for c = ch(i, :)
            if c > 0, depth(c) = depth(i) + 1; end
        end
    end
    if contains(class(tr), 'Regression')
        [~, node] = predict(tr, X);
    else
        [~, ~, node] = predict(tr, X);
    end
    d = mean(depth(node));
end

function tr = unwrap(t)
%UNWRAP  The tree object itself, or a tree stored inside a wrapper object.
    tr = [];
    if isprop(t, 'Children') || isfield(t, 'Children')
        tr = t;  return
    end
    for f = {'CompactRegressionLearner', 'RegressionLearner', 'Learner', 'Impl'}
        try
            u = t.(f{1});
            if isobject(u) && isprop(u, 'Children'), tr = u;  return; end
        catch
        end
    end
    try
        w = warning('off', 'all');
        s = struct(t);
        warning(w);
        fn = fieldnames(s);
        for i = 1:numel(fn)
            u = s.(fn{i});
            if isobject(u) && isprop(u, 'Children'), tr = u;  return; end
        end
    catch
        warning(w);
    end
end
