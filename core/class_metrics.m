function M = class_metrics(yTrue, yPred, classes)
%CLASS_METRICS  Confusion matrix and per-class precision, recall and F1.
%
%   M = class_metrics(yTrue, yPred, classes)
%     yTrue, yPred  label vectors (numeric)
%     classes       class labels in the order of the output (default 0:8)
%
%   M.confusion(i, j)  number of windows of class classes(i) predicted as
%                      classes(j) (rows = truth, columns = prediction)
%   M.precision, M.recall, M.f1, M.support   1 x C, per class
%   M.macroF1          mean F1 over the classes that occur in yTrue
%   M.accuracy
%   A class that is never predicted has precision 0; F1 is 0 when both
%   precision and recall are 0.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 3 || isempty(classes), classes = 0:8; end
    yTrue = yTrue(:);  yPred = yPred(:);
    C = numel(classes);
    conf = zeros(C);
    for i = 1:C
        for j = 1:C
            conf(i, j) = sum(yTrue == classes(i) & yPred == classes(j));
        end
    end
    tp = diag(conf)';
    predCount = sum(conf, 1);
    support = sum(conf, 2)';
    precision = tp ./ max(predCount, 1);
    recall = tp ./ max(support, 1);
    f1 = 2 * precision .* recall ./ max(precision + recall, eps);
    present = support > 0;
    M = struct('classes', classes(:)', 'confusion', conf, 'precision', precision, ...
               'recall', recall, 'f1', f1, 'support', support, ...
               'macroF1', mean(f1(present)), 'accuracy', sum(tp) / max(numel(yTrue), 1));
end
