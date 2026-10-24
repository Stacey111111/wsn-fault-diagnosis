function q = quantiles(v, p)
%QUANTILES  Sample quantiles by linear interpolation (base MATLAB only;
%   prctile/quantile need the Statistics and Machine Learning Toolbox).
%
%   q = quantiles(v, p)
%     v  data vector (NaN values are ignored)
%     p  probabilities in [0, 1], e.g. [0.25 0.5 0.75]
%     q  quantiles, same size as p (NaN if v has no values)
%
%   Week 8 deliverable, CS 5395 Independent Study (Xiaohui San).

    v = sort(v(~isnan(v(:))));
    n = numel(v);
    if n == 0, q = nan(size(p)); return; end
    if n == 1, q = v * ones(size(p)); return; end
    q = interp1((0:n - 1)' / (n - 1), v, p);
end
