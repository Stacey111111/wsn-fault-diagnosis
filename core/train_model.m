function mdl = train_model(name, X, y, hp)
%TRAIN_MODEL  Train one classifier with given hyperparameters.
%
%   mdl = train_model(name, X, y, hp)
%     name  'svm', 'tree', 'knn', 'rf', 'nn' (Weeks 9-10), 'bag' (Week 11),
%           'adaboost', 'rusboost', 'gentleboost', 'logitboost' (Week 12)
%     X     n x p features,  y  n x 1 numeric class labels
%     hp    struct of hyperparameters (see hp_grid for the fields)
%     mdl   trained model, used with predict_model
%
%   svm   RBF-kernel SVMs combined one-vs-one (fitcecoc + templateSVM),
%         features standardised; hp.C (box constraint), hp.kernelScale
%   tree  CART decision tree with Gini impurity (fitctree);
%         hp.minLeaf, hp.maxSplits
%   knn   k-nearest neighbours, Euclidean distance on standardised
%         features (fitcknn); hp.k, hp.weight ('equal' or 'inverse')
%   rf    random forest (TreeBagger: bootstrap samples + random feature
%         subsets at each split); hp.nTrees, hp.mtry, hp.minLeaf, and
%         optionally hp.importance = true for out-of-bag permutation
%         importance
%   nn    fully connected neural network with ReLU layers (fitcnet),
%         standardised features; hp.layers (e.g. [50 20]), hp.lambda
%         (L2 regularisation), hp.iterLimit (default 500)
%   bag   bagging of any of the models above (bag_train): hp.base,
%         hp.baseHp, hp.nBags, optional hp.sampleFrac            (Week 11)
%   adaboost    AdaBoost.M2 with decision trees (fitcensemble)
%   rusboost    RUSBoost: AdaBoost.M2 on randomly under-sampled data, so
%               every class is equally represented in each round
%   gentleboost GentleBoost, binary, combined one-vs-all (fitcecoc +
%               templateEnsemble)
%   logitboost  LogitBoost, binary, combined one-vs-all
%               boosting fields: hp.nLearn (rounds), hp.learnRate
%               (shrinkage), hp.maxSplits (size of each tree)     (Week 12)
%
%   Needs the Statistics and Machine Learning Toolbox (fitcnet: R2021a+).
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    switch lower(name)
        case 'svm'
            t = templateSVM('KernelFunction', 'gaussian', 'BoxConstraint', hp.C, ...
                            'KernelScale', hp.kernelScale, 'Standardize', true);
            mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsone');
        case 'tree'
            mdl = fitctree(X, y, 'SplitCriterion', 'gdi', 'MinLeafSize', hp.minLeaf, ...
                           'MaxNumSplits', hp.maxSplits);
        case 'knn'
            mdl = fitcknn(X, y, 'NumNeighbors', hp.k, 'Distance', 'euclidean', ...
                          'DistanceWeight', hp.weight, 'Standardize', true);
        case 'rf'
            imp = 'off';
            if isfield(hp, 'importance') && hp.importance, imp = 'on'; end
            mdl = TreeBagger(hp.nTrees, X, y, 'Method', 'classification', ...
                             'NumPredictorsToSample', hp.mtry, 'MinLeafSize', hp.minLeaf, ...
                             'OOBPredictorImportance', imp);
        case 'nn'
            it = 500;
            if isfield(hp, 'iterLimit'), it = hp.iterLimit; end
            mdl = fitcnet(X, y, 'LayerSizes', hp.layers, 'Activations', 'relu', ...
                          'Lambda', hp.lambda, 'Standardize', true, 'IterationLimit', it);
        case 'bag'
            mdl = bag_train(X, y, hp);
        case {'adaboost', 'rusboost'}
            meth = 'AdaBoostM2';
            if strcmpi(name, 'rusboost'), meth = 'RUSBoost'; end
            t = templateTree('MaxNumSplits', hp.maxSplits);
            mdl = fitcensemble(X, y, 'Method', meth, 'NumLearningCycles', hp.nLearn, ...
                               'Learners', t, 'LearnRate', hp.learnRate);
        case {'gentleboost', 'logitboost'}
            meth = 'GentleBoost';
            if strcmpi(name, 'logitboost'), meth = 'LogitBoost'; end
            t = templateEnsemble(meth, hp.nLearn, templateTree('MaxNumSplits', hp.maxSplits), ...
                                 'LearnRate', hp.learnRate);
            mdl = fitcecoc(X, y, 'Learners', t, 'Coding', 'onevsall');
        otherwise
            error('train_model:name', 'Unknown model "%s".', name);
    end
end
