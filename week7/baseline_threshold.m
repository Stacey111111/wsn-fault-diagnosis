function [yPred, info] = baseline_threshold(R, s0, nodes, winLen, th)
%BASELINE_THRESHOLD  Rule-based fault classifier with fixed thresholds
%   (Baseline 1).
%
%   [yPred, info] = baseline_threshold(R, s0, nodes, winLen, th)
%     R       one run from simulate_run
%     s0      first sample of each window (column vector)
%     nodes   node of each window (same length as s0)
%     winLen  samples per window (default 12)
%     th      struct of thresholds (defaults below; any field may be given)
%     yPred   predicted class (0-8) of each window
%     info    struct with the statistics the rules looked at (one row per
%             window), for plots and for checking the rules
%
%   Only data the sink has are used: received readings, the battery voltage
%   carried in those packets, the neighbours' readings and the routing
%   tree. A missing sample is not held against the node when it was lost
%   at a relay further up the tree (R.relayLost) or when the node is alive
%   but has no route to the sink (R.parent < 0 while R.energy > 0; the
%   routing layer knows this when it rebuilds the tree).
%
%   Rules, checked in this order (the first one that fires decides):
%     1. own missing fraction >= th.miss
%            -> battery if the last reported voltage < th.vWarn, else loss
%     2. lowest voltage in the window < th.vLow                   -> battery
%        (2.6 V: a battery fault is labelled from its onset, when the
%        capacity drop already lowers the voltage to about 2.5 V; healthy
%        nodes stay near 3.0 V)
%     3. at least 6 readings and range < th.flat                  -> stuck
%     4. largest jump between readings > th.jump and more than
%        th.spikeRatio times the median jump (an isolated jump)   -> spike
%     5. robust std of the first differences > th.noise           -> noise
%     6. |median neighbour residual| > th.res: fit
%           residual = a + b * (neighbour level) + c * time
%        over the samples of the context (th.ctx samples ending with the
%        window) whose residual is at least half the window's residual
%           |b| > th.gain and the level term explains more      -> gain
%           |c| > th.drift (deg C / h)                           -> drift
%           otherwise                                            -> offset
%     7. |median residual| > th.resLow and c > th.drift with the same
%        sign as the residual (an early, still small drift)       -> drift
%     8. otherwise                                                -> normal
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 4 || isempty(winLen), winLen = 12; end
    d = struct('miss', 0.15, 'vWarn', 2.6, 'vLow', 2.6, 'flat', 0.02, ...
               'jump', 3, 'spikeRatio', 4, 'noise', 0.5, 'res', 1.5, ...
               'resLow', 0.6, 'gain', 0.08, 'drift', 0.1, 'ctx', 288);
    if nargin >= 5 && ~isempty(th)
        f = fieldnames(th);
        for i = 1:numel(f), d.(f{i}) = th.(f{i}); end
    end
    th = d;

    [res, nbMed] = neighbour_residual(R);
    cutOff = R.parent < 0 & R.energy > 0;              % alive, no route
    vRep = R.voltage;  vRep(isnan(R.reading)) = NaN;    % voltage the sink received
    n = numel(s0);
    yPred = zeros(n, 1);
    z = nan(n, 1);
    info = struct('ownMiss', z, 'vLast', z, 'vMin', z, 'range', z, 'maxJump', z, ...
                  'diffStd', z, 'medRes', z, 'gainSlope', z, 'driftSlope', z, ...
                  'rule', zeros(n, 1));
    for w = 1:n
        i = nodes(w);
        s = s0(w):s0(w) + winLen - 1;
        x = R.reading(s, i);
        got = ~isnan(x);
        own = ~got & ~R.relayLost(s, i) & ~cutOff(s, i);
        % battery voltage: lowest in the window, last one ever reported
        vw = vRep(s, i);
        vMin = min([vw(~isnan(vw)); inf]);
        vh = vRep(1:s(end), i);  vh = vh(~isnan(vh));
        if isempty(vh), vLast = NaN; else, vLast = vh(end); end
        % shape of the signal inside the window
        xs = x(got);
        if numel(xs) >= 2, dx = abs(diff(xs)); else, dx = 0; end
        dxs = diff(xs);
        if numel(dxs) >= 2
            rStd = 1.4826 * median(abs(dxs - median(dxs)));   % robust std
        else
            rStd = 0;
        end
        % neighbour residual and its trend over the context
        rw = res(s, i);  rw = rw(~isnan(rw));
        if isempty(rw), medRes = NaN; else, medRes = median(rw); end
        gs = NaN;  ds = NaN;  gainWins = false;
        if ~isnan(medRes)
            c = max(1, s(end) - th.ctx + 1):s(end);
            rc = res(c, i);  lc = nbMed(c, i);  tc = R.time(c);
            ok = ~isnan(rc) & ~isnan(lc) & abs(rc) >= 0.5 * abs(medRes);
            if sum(ok) >= 6
                X = [ones(sum(ok), 1), lc(ok) - mean(lc(ok)), tc(ok) - mean(tc(ok))];
                if rank(X) == 3
                    beta = X \ rc(ok);
                    gs = beta(2);  ds = beta(3);
                    gainWins = abs(gs) * std(lc(ok)) > abs(ds) * std(tc(ok));
                end
            end
        end
        info.ownMiss(w) = mean(own);  info.vLast(w) = vLast;  info.vMin(w) = vMin;
        info.range(w) = max([xs; -inf]) - min([xs; inf]);
        info.maxJump(w) = max(dx);  info.diffStd(w) = rStd;
        info.medRes(w) = medRes;  info.gainSlope(w) = gs;  info.driftSlope(w) = ds;

        if info.ownMiss(w) >= th.miss
            r = 1;
            if ~isnan(vLast) && vLast < th.vWarn, y = 8; else, y = 7; end
        elseif vMin < th.vLow
            y = 8; r = 2;
        elseif numel(xs) >= 6 && info.range(w) < th.flat
            y = 4; r = 3;
        elseif info.maxJump(w) > th.jump && info.maxJump(w) > th.spikeRatio * median(dx)
            y = 5; r = 4;
        elseif numel(xs) >= 3 && rStd > th.noise
            y = 6; r = 5;
        elseif ~isnan(medRes) && abs(medRes) > th.res
            r = 6;
            if ~isnan(gs) && abs(gs) > th.gain && gainWins
                y = 2;
            elseif ~isnan(ds) && abs(ds) > th.drift
                y = 3;
            else
                y = 1;
            end
        elseif ~isnan(medRes) && abs(medRes) > th.resLow && ~isnan(ds) && ...
               abs(ds) > th.drift && sign(ds) == sign(medRes)
            y = 3; r = 7;
        else
            y = 0; r = 8;
        end
        yPred(w) = y;  info.rule(w) = r;
    end
end
