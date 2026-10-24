function mdl = bag_train(X, y, hp)
%BAG_TRAIN  Bagging of any base learner of train_model (bootstrap
%   aggregating, Breiman 1996).
%
%   mdl = bag_train(X, y, hp)
%     hp.base       name of the base learner ('tree', 'svm', 'knn', 'nn', ...)
%     hp.baseHp     hyperparameters of the base learner (see hp_grid)
%     hp.nBags      number of bootstrap models (e.g. 25)
%     hp.sampleFrac size of each bootstrap sample as a fraction of n
%                   (default 1, sampling with replacement)
%     mdl           struct with fields type = 'bag', base, models (cell),
%                   classes; predict with predict_model (majority vote)
%
%   Each base model is trained on its own bootstrap sample of the training
%   windows. The random stream is the one set by the caller (cv_evaluate
%   fixes it before every fit), so the result is reproducible.
%
%   Week 11 deliverable, CS 5395 Independent Study (Xiaohui San).

    n = size(X, 1);
    frac = 1;
    if isfield(hp, 'sampleFrac') && ~isempty(hp.sampleFrac), frac = hp.sampleFrac; end
    m = max(1, round(frac * n));
    models = cell(1, hp.nBags);
    for b = 1:hp.nBags
        idx = randi(n, m, 1);                         % bootstrap sample
        mb = train_model(hp.base, X(idx, :), y(idx), hp.baseHp);
        try
            mb = compact(mb);                          % drop the training data
        catch
            % k-NN keeps its data: the data are the model
        end
        models{b} = mb;
    end
    mdl = struct('type', 'bag', 'base', hp.base, 'models', {models}, ...
                 'classes', unique(y(:))');
end
