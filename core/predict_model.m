function yPred = predict_model(mdl, X)
%PREDICT_MODEL  Class predictions of a model from train_model.
%
%   yPred = predict_model(mdl, X)   returns an n x 1 numeric label vector.
%   TreeBagger returns labels as text; they are converted back to numbers.
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    if isstruct(mdl) && isfield(mdl, 'type') && strcmp(mdl.type, 'bag')
        P = zeros(size(X, 1), numel(mdl.models));
        for b = 1:numel(mdl.models)
            P(:, b) = predict_model(mdl.models{b}, X);
        end
        yPred = majority_vote(P, mdl.classes);           % Week 11
        return
    end
    yPred = predict(mdl, X);
    if iscell(yPred), yPred = str2double(yPred); end
    yPred = double(yPred(:));
end
