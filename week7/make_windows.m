function W = make_windows(R, runId, winLen)
%MAKE_WINDOWS  Cut every node's time series into non-overlapping windows
%   and give each window one label.
%
%   W = make_windows(R, runId, winLen)
%     R       one run from simulate_run
%     runId   index of the run (stored in W.run, used for grouped CV)
%     winLen  samples per window (default 12 = 1 h at 5-minute sampling)
%
%   Window label (Week 6 rules):
%     - any spike sample in the window          -> spike (5)
%     - at least half of the samples faulty     -> most common fault label
%     - no faulty sample                        -> normal (0)
%     - otherwise (a fault starts or ends inside the window) -> -1, a
%       transition window that is kept in the table but not used for
%       training or testing
%   label2 is the most common secondary label (0 if none).
%
%   W fields (one row per window): run, node, s0 (first sample), tStart [h],
%   label, label2, faultFrac, eventFrac, missFrac, relayFrac.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 3 || isempty(winLen), winLen = 12; end
    [T, N] = size(R.reading);
    nW = floor(T / winLen);
    n = nW * N;
    W = struct('run', zeros(n, 1), 'node', zeros(n, 1), 's0', zeros(n, 1), ...
               'tStart', zeros(n, 1), 'label', zeros(n, 1), 'label2', zeros(n, 1), ...
               'faultFrac', zeros(n, 1), 'eventFrac', zeros(n, 1), ...
               'missFrac', zeros(n, 1), 'relayFrac', zeros(n, 1));
    r = 0;
    for i = 1:N
        for w = 1:nW
            r = r + 1;
            s = (w - 1) * winLen + (1:winLen);
            lab = R.label(s, i);
            fault = lab > 0;
            if any(lab == 5)
                y = 5;
            elseif sum(fault) >= winLen / 2
                y = mode(lab(fault));
            elseif ~any(fault)
                y = 0;
            else
                y = -1;
            end
            l2 = R.label2(s, i);
            l2 = l2(l2 > 0 & l2 ~= y);
            if isempty(l2), y2 = 0; else, y2 = mode(l2); end
            W.run(r) = runId;  W.node(r) = i;  W.s0(r) = s(1);
            W.tStart(r) = R.time(s(1));
            W.label(r) = y;  W.label2(r) = y2;
            W.faultFrac(r) = mean(fault);
            W.eventFrac(r) = mean(R.event(s, i));
            W.missFrac(r) = mean(isnan(R.reading(s, i)));
            W.relayFrac(r) = mean(R.relayLost(s, i));
        end
    end
end
