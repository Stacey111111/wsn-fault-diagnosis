function z = kalman_innovations(x, q, r)
%KALMAN_INNOVATIONS  Normalised innovations of a local linear trend
%   Kalman filter run over one sensor's readings.
%
%   z = kalman_innovations(x, q, r)
%     x  readings of one node (NaN = missing; the filter only predicts)
%     q  [qLevel qTrend] process noise standard deviations per sample
%        (default [0.05 0.01] deg C)
%     r  measurement noise standard deviation (default 0.2 deg C)
%     z  innovation / sqrt(innovation variance), NaN where x is missing
%
%   State: [level; trend], level(k+1) = level(k) + trend(k). A fault that
%   breaks the smooth behaviour of the signal (spike, noise, a jump at the
%   onset of an offset) gives large |z|. A fault that keeps the signal smooth
%   (slow drift, gain, a sensor stuck at a constant) gives small |z|.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 2 || isempty(q), q = [0.05 0.01]; end
    if nargin < 3 || isempty(r), r = 0.2; end
    x = x(:);  n = numel(x);
    z = nan(n, 1);
    F = [1 1; 0 1];  H = [1 0];
    Q = diag(q .^ 2);  Rm = r ^ 2;
    k0 = find(~isnan(x), 1);
    if isempty(k0), return; end
    s = [x(k0); 0];  P = diag([1 0.1]);
    for k = k0 + 1:n
        s = F * s;  P = F * P * F' + Q;                 % predict
        if isnan(x(k)), continue; end
        S = H * P * H' + Rm;
        v = x(k) - H * s;
        z(k) = v / sqrt(S);
        K = P * H' / S;                                  % update
        s = s + K * v;
        P = (eye(2) - K * H) * P;
    end
end
