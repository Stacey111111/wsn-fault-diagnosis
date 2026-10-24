function [y, label, info] = inject_fault(x, t, type, opts)
%INJECT_FAULT  Inject one fault type into a clean sensor time series.
%
%   [y, label, info] = inject_fault(x, t, type, opts)
%     x     clean readings (column vector)
%     t     time stamps in hours (same size as x)
%     type  one of the fault classes of the Week 4 taxonomy:
%             'normal'   no fault
%             'offset'   constant bias:          y = x + b
%             'gain'     calibration gain error:  y = g*x
%             'drift'    slowly growing bias:     y = x + r*(t - t0)
%             'stuck'    stuck-at value:          y = x(t0) (constant)
%             'spike'    isolated outliers:       y = x + large random pulses
%             'noise'    precision degradation:   y = x + N(0, s^2), s >> normal
%             'loss'     packet loss / link fault: readings missing (NaN)
%             'battery'  low-battery degradation: bias, noise and loss grow as
%                        the voltage falls, then the node dies (hard failure)
%     opts  struct overriding default parameters (fields below)
%
%   Outputs
%     y      faulty readings (NaN = no packet received)
%     label  0 where the sample is normal, class id where the fault is active
%     info   struct with the parameters actually used and the class id
%
%   Default parameters (opts fields):
%     t0 = 24      fault onset time [h]      t1 = inf  fault end time [h]
%     bias = 3     offset [deg C]            gain = 1.25
%     rate = 0.15  drift rate [deg C / h]    spikeProb = 0.03, spikeAmp = 8
%     noiseStd = 1.5                         lossProb = 0.35 (bursty)
%     vStart = 3.0, vDead = 2.1 [V]          (battery model)
%
%   This is the prototype for FaultInjector.m (Week 6).
%   Week 4 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 4, opts = struct(); end
    d = struct('t0', 24, 't1', inf, 'bias', 3, 'gain', 1.25, 'rate', 0.15, ...
               'spikeProb', 0.03, 'spikeAmp', 8, 'noiseStd', 1.5, ...
               'lossProb', 0.35, 'vStart', 3.0, 'vDead', 2.1);
    f = fieldnames(opts);
    for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end

    classes = {'normal', 'offset', 'gain', 'drift', 'stuck', 'spike', 'noise', 'loss', 'battery'};
    id = find(strcmpi(classes, type)) - 1;
    if isempty(id)
        error('inject_fault:type', 'Unknown fault type "%s".', type);
    end

    x = x(:); t = t(:);
    y = x;
    active = t >= d.t0 & t < d.t1;
    n = numel(x);

    switch lower(type)
        case 'normal'
            active(:) = false;
        case 'offset'
            y(active) = x(active) + d.bias;
        case 'gain'
            y(active) = d.gain * x(active);
        case 'drift'
            y(active) = x(active) + d.rate * (t(active) - d.t0);
        case 'stuck'
            i0 = find(active, 1);
            y(active) = x(i0);
        case 'spike'
            hit = active & (rand(n, 1) < d.spikeProb);
            y(hit) = x(hit) + d.spikeAmp * sign(randn(sum(hit), 1)) .* (0.6 + 0.4 * rand(sum(hit), 1));
            active = hit;                           % only the spikes are faulty samples
        case 'noise'
            y(active) = x(active) + d.noiseStd * randn(sum(active), 1);
        case 'loss'
            lost = active & bursty_loss(n, d.lossProb);
            y(lost) = NaN;
        case 'battery'
            % supply voltage falls linearly from vStart at t0 to vDead at the end
            v = d.vStart * ones(n, 1);
            tEnd = t(end) - 0.15 * (t(end) - d.t0);        % node dies before the end
            v(active) = d.vStart - (d.vStart - d.vDead) * (t(active) - d.t0) / (tEnd - d.t0);
            sev = min(max((d.vStart - v) / (d.vStart - d.vDead), 0), 1);   % 0..1 severity
            y = x - 2.5 * sev.^2 + (0.2 + 1.5 * sev.^2) .* randn(n, 1) .* active;
            lost = active & (rand(n, 1) < 0.6 * sev.^3);
            dead = v <= d.vDead;
            y(lost | dead) = NaN;
            info.voltage = v;
    end

    label = zeros(n, 1);
    label(active) = id;
    info.class = id;
    info.name = lower(type);
    info.params = d;
end

function lost = bursty_loss(n, p)
%BURSTY_LOSS  Two-state Gilbert-Elliott channel: losses come in bursts.
    pGB = 0.05;                      % good -> bad transition probability
    pBG = pGB * (1 - p) / p;         % chosen so the long-run loss rate is ~p
    lost = false(n, 1);
    bad = false;
    for i = 1:n
        if bad
            bad = rand() > pBG;
        else
            bad = rand() < pGB;
        end
        lost(i) = bad;
    end
end
