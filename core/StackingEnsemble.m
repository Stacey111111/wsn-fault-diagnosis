classdef StackingEnsemble < handle
%STACKINGENSEMBLE  Stacked generalisation (Wolpert 1992) of the base models
%   of Weeks 9-12 with grouped inner cross-validation.
%
%   se = StackingEnsemble(baseNames, baseHp, opts)
%     baseNames  1 x M cell of model names for train_model
%                (e.g. {'tree','svm','knn','rf','nn','adaboost','logitboost'})
%     baseHp     1 x M cell of their hyperparameter structs
%     opts       optional struct:
%                classes    class labels (default 0:8)
%                nInner     inner folds (default 4)
%                innerSeed  seed of the inner fold assignment (default 1)
%                metaType   'logistic' (default) or 'tree'
%
%   se.fit(X, y, groups)
%     1. Inner grouped CV on the training windows: every base model is
%        trained on nInner-1 inner folds and scores the windows of the
%        remaining fold (score_model). Because the inner folds keep the
%        networks together, the meta-features Zoof of each window come from
%        base models that never saw the window's network, just as at test
%        time.
%     2. Every base model is trained again on all training windows.
%     3. The meta-learner is trained on Zoof (n x M*K class scores).
%   [yPred, Z] = se.predict(X)       base scores Z of new windows, then the
%                                    meta-learner's prediction
%   Z = se.baseScores(X)             n x M*K meta-features of new windows
%   meta = se.fitMeta(type, cols)    train another meta-learner on the
%                                    stored Zoof (optionally on the columns
%                                    cols only, e.g. to leave a base model out)
%   yPred = se.predictMeta(meta, Z)  prediction of such a meta-learner
%   cols = se.blockCols(m)           columns of Z that belong to base model m
%
%   Meta-learners
%     'logistic'  one-vs-all L2-regularised logistic regression on the
%                 standardised scores (fitcecoc + templateLinear, default
%                 lambda = 1/n)
%     'tree'      CART tree on the raw scores (minimum leaf size 20)
%
%   se.baseOofF1 holds the inner-CV macro-F1 of each base model (used for
%   weighted voting).
%
%   Week 13 deliverable, CS 5395 Independent Study (Xiaohui San).

    properties
        baseNames = {}
        baseHp = {}
        classes = 0:8
        nInner = 4
        innerSeed = 1
        metaType = 'logistic'
        baseModels = {}     % base models trained on all training windows
        meta = []           % default meta-learner (struct, see fitMeta)
        Zoof = []           % n x M*K out-of-fold scores of the training windows
        yTrain = []
        baseOofF1 = []      % 1 x M inner-CV macro-F1 of each base model
        fitSeconds = 0
    end

    methods
        function obj = StackingEnsemble(baseNames, baseHp, opts)
            if nargin < 3, opts = struct(); end
            obj.baseNames = baseNames;  obj.baseHp = baseHp;
            f = fieldnames(opts);
            for i = 1:numel(f), obj.(f{i}) = opts.(f{i}); end
        end

        function fit(obj, X, y, groups)
            t0 = tic;
            y = y(:);  n = numel(y);
            M = numel(obj.baseNames);  K = numel(obj.classes);
            fin = grouped_folds(groups, obj.nInner, obj.innerSeed);
            Z = zeros(n, M * K);
            for f = 1:obj.nInner
                te = fin == f;  tr = ~te;
                for m = 1:M
                    mdl = train_model(obj.baseNames{m}, X(tr, :), y(tr), obj.baseHp{m});
                    Z(te, obj.blockCols(m)) = score_model(mdl, X(te, :), obj.classes);
                end
            end
            obj.Zoof = Z;  obj.yTrain = y;
            obj.baseOofF1 = zeros(1, M);
            for m = 1:M
                Mm = class_metrics(y, obj.blockPrediction(Z, m), obj.classes);
                obj.baseOofF1(m) = Mm.macroF1;
            end
            obj.baseModels = cell(1, M);
            for m = 1:M
                mdl = train_model(obj.baseNames{m}, X, y, obj.baseHp{m});
                try
                    mdl = compact(mdl);
                catch
                end
                obj.baseModels{m} = mdl;
            end
            obj.meta = obj.fitMeta(obj.metaType);
            obj.fitSeconds = toc(t0);
        end

        function cols = blockCols(obj, m)
            K = numel(obj.classes);
            cols = (m - 1) * K + (1:K);
        end

        function yp = blockPrediction(obj, Z, m)
            %BLOCKPREDICTION  Class predicted by base model m (largest score).
            [~, j] = max(Z(:, obj.blockCols(m)), [], 2);
            yp = obj.classes(j);  yp = yp(:);
        end

        function Z = baseScores(obj, X)
            M = numel(obj.baseModels);  K = numel(obj.classes);
            Z = zeros(size(X, 1), M * K);
            for m = 1:M
                Z(:, obj.blockCols(m)) = score_model(obj.baseModels{m}, X, obj.classes);
            end
        end

        function meta = fitMeta(obj, type, cols)
            if nargin < 2 || isempty(type), type = obj.metaType; end
            if nargin < 3 || isempty(cols), cols = 1:size(obj.Zoof, 2); end
            Z = obj.Zoof(:, cols);
            mu = mean(Z, 1);  sd = std(Z, 0, 1);  sd(sd < 1e-12) = 1;
            switch lower(type)
                case 'logistic'
                    t = templateLinear('Learner', 'logistic', 'Regularization', 'ridge');
                    mdl = fitcecoc((Z - mu) ./ sd, obj.yTrain, 'Learners', t, 'Coding', 'onevsall', ...
                                   'ClassNames', obj.classes);
                case 'tree'
                    mdl = fitctree(Z, obj.yTrain, 'MinLeafSize', 20, 'ClassNames', obj.classes);
                otherwise
                    error('StackingEnsemble:meta', 'Unknown meta-learner "%s".', type);
            end
            meta = struct('type', lower(type), 'cols', cols, 'mu', mu, 'sd', sd, 'mdl', mdl);
        end

        function yp = predictMeta(obj, meta, Z)
            Zc = Z(:, meta.cols);
            if strcmp(meta.type, 'logistic'), Zc = (Zc - meta.mu) ./ meta.sd; end
            yp = predict(meta.mdl, Zc);
            yp = double(yp(:));
        end

        function [yp, Z] = predict(obj, X)
            Z = obj.baseScores(X);
            yp = obj.predictMeta(obj.meta, Z);
        end

        function w = metaWeights(obj, meta)
            %METAWEIGHTS  Mean |coefficient| of each base model's scores in a
            %   logistic meta-learner (scores are standardised, so the
            %   magnitudes are comparable). 1 x M, NaN for models not used.
            if nargin < 2, meta = obj.meta; end
            M = numel(obj.baseNames);
            w = nan(1, M);
            B = zeros(numel(meta.cols), 0);
            for c = 1:numel(meta.mdl.BinaryLearners)
                B(:, end + 1) = meta.mdl.BinaryLearners{c}.Beta(:); %#ok<AGROW>
            end
            for m = 1:M
                in = ismember(meta.cols, obj.blockCols(m));
                if any(in), w(m) = mean(abs(B(in, :)), 'all'); end
            end
        end
    end
end
