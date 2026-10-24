function score = baseline_kalman(R, s0, nodes, winLen, q, r)
%BASELINE_KALMAN  Kalman filter residual detector (Baseline 2).
%
%   score = baseline_kalman(R, s0, nodes, winLen, q, r)
%     R       one run from simulate_run
%     s0      first sample of each window
%     nodes   node of each window
%     winLen  samples per window (default 12)
%     q, r    filter noise settings passed to kalman_innovations
%     score   largest |normalised innovation| inside each window
%             (0 if the window has no received reading)
%
%   The filter runs once over each node's whole series of received readings,
%   so it is already settled when a window starts. A window is flagged as
%   faulty when score > k; the demo uses k = 4 and sweeps k from 2 to 6.
%   The detector says whether a window is faulty, not which fault it is.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 4 || isempty(winLen), winLen = 12; end
    if nargin < 5, q = []; end
    if nargin < 6, r = []; end
    Z = nan(size(R.reading));
    for i = unique(nodes(:))'
        Z(:, i) = kalman_innovations(R.reading(:, i), q, r);
    end
    score = zeros(numel(s0), 1);
    for w = 1:numel(s0)
        zw = abs(Z(s0(w):s0(w) + winLen - 1, nodes(w)));
        zw = zw(~isnan(zw));
        if ~isempty(zw), score(w) = max(zw); end
    end
end
