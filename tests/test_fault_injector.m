function results = test_fault_injector()
%TEST_FAULT_INJECTOR  Unit tests for FaultInjector (Week 6).
%   results = test_fault_injector() runs every test, prints PASS/FAIL and
%   returns a struct array with the name, pass flag and measured value.
%   Each test injects one fault into a single SensorNode that senses a
%   constant or slowly varying field, then checks the statistics of the
%   faulty readings against the fault parameters.

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week3'));
    rng(11);
    dt = 5 / 60;  t = (0:dt:72 - dt)';  n = numel(t);
    truth = 25 * ones(n, 1);
    results = struct('name', {}, 'pass', {}, 'value', {});

    % 1. offset
    [r, lab] = runOne('offset', 24, 48, struct('bias', 3), truth, t, dt);
    in = t >= 24 & t < 48;
    e = mean(r(in) - truth(in));
    results(end + 1) = check('offset: mean error = bias (3)', abs(e - 3) < 0.05 && all(lab(in) == 1) && all(lab(~in) == 0), e);

    % 2. gain
    [r, lab] = runOne('gain', 24, 48, struct('gain', 1.2), truth, t, dt);
    g = mean(r(in)) / mean(truth(in));
    results(end + 1) = check('gain: reading / truth = 1.2', abs(g - 1.2) < 0.01 && all(lab(in) == 2), g);

    % 3. drift
    [r, lab] = runOne('drift', 24, Inf, struct('rate', 0.2), truth, t, dt);
    in2 = t >= 24;
    c = polyfit(t(in2) - 24, r(in2) - truth(in2), 1);
    results(end + 1) = check('drift: slope = 0.2 degC/h', abs(c(1) - 0.2) < 0.01 && all(lab(in2) == 3), c(1));

    % 4. stuck-at
    [r, lab] = runOne('stuck', 24, 48, struct(), truth + sin(t), t, dt);
    results(end + 1) = check('stuck: zero variance in window', std(r(in)) < 1e-12 && all(lab(in) == 4), std(r(in)));

    % 5. spike
    [r, lab] = runOne('spike', 0, Inf, struct('prob', 0.05, 'amp', 8), truth, t, dt);
    fr = mean(lab == 5);
    results(end + 1) = check('spike: labelled fraction = prob (0.05)', abs(fr - 0.05) < 0.02 && min(abs(r(lab == 5) - 25)) > 4, fr);

    % 6. noise
    [r, lab] = runOne('noise', 0, Inf, struct('std', 1.5), truth, t, dt);
    s = std(r - truth);
    results(end + 1) = check('noise: error std = 1.5', abs(s - 1.5) < 0.12 && all(lab == 6), s);

    % 7. packet loss (long run, bursty)
    tl = (0:dt:720 - dt)';
    [r, lab] = runOne('loss', 0, Inf, struct('prob', 0.4), 25 * ones(numel(tl), 1), tl, dt);
    lr = mean(isnan(r));
    runs = diff([0; isnan(r); 0]);  burst = mean(find(runs == -1) - find(runs == 1));
    results(end + 1) = check('loss: loss rate = 0.4, mean burst > 5', abs(lr - 0.4) < 0.08 && burst > 5 && all(lab == 7), lr);

    % 8. battery depletion (node spends energy every step)
    nd = SensorNode(1, 0, 0);
    inj = FaultInjector();  inj.addFault(1, 'battery', 12, Inf, struct('capFrac', 0.1, 'leak', 2));
    r = nan(n, 1);  lab = zeros(n, 1);  v = zeros(n, 1);
    for i = 1:n
        if nd.isAlive()
            [ri, lost, lab(i), ~, drain] = inj.apply(nd, nd.sense(25), 25, t(i), dt);
            if ~lost, r(i) = ri; end
            nd.spend(2e-4 * drain);                    % 2.4 mJ/h baseline use
        else
            lab(i) = inj.labelsFor(nd, t(i));
        end
        v(i) = nd.voltage();
    end
    tDeath = t(find(v <= nd.vMin + 1e-9, 1));
    on = t >= 12;
    degraded = any(v < 2.5 & v > nd.vMin + 1e-9);    % readings degrade before the node dies
    % labelled battery from the onset (the capacity drop shows in the voltage),
    % through the degraded phase, and after the node has died
    ok = ~isempty(tDeath) && tDeath > 12 && degraded && all(lab(~on) == 0) && all(lab(on) == 8);
    results(end + 1) = check('battery: labelled from onset, degrades, then dies', ok, tDeath);

    % 9. simultaneous faults: large offset + small noise
    nd = SensorNode(1, 0, 0);  inj = FaultInjector();
    inj.addFault(1, 'offset', 0, Inf, struct('bias', 4));
    inj.addFault(1, 'noise', 0, Inf, struct('std', 0.8));
    L1 = zeros(200, 1);  L2 = zeros(200, 1);
    for i = 1:200
        [~, ~, L1(i), L2(i)] = inj.apply(nd, nd.sense(25), 25, i * dt, dt);
    end
    results(end + 1) = check('simultaneous: primary offset, secondary noise', all(L1 == 1) && all(L2 == 6), mean(L2 == 6));

    % 10. environmental event
    inj = FaultInjector();  inj.addEvent(50, 50, 20, 10, 14, 8);
    peak = inj.eventDelta(50, 50, 12);  far = inj.eventDelta(100, 100, 12);  after = inj.eventDelta(50, 50, 15);
    results(end + 1) = check('event: +8 degC at centre, 0 far away / after', abs(peak - 8) < 1e-9 && far < 0.01 && after == 0, peak);

    % 11. out-of-bounds saturation
    [r, ~] = runOne('gain', 0, Inf, struct('gain', 5), truth, t, dt);
    results(end + 1) = check('out-of-bounds: saturated at 85 degC', max(r) == 85, max(r));

    fprintf('\n%d of %d tests passed\n', sum([results.pass]), numel(results));
end

function [r, lab] = runOne(type, t0, t1, p, truth, t, dt)
    nd = SensorNode(1, 0, 0);
    inj = FaultInjector();
    inj.addFault(1, type, t0, t1, p);
    n = numel(t);  r = nan(n, 1);  lab = zeros(n, 1);
    for i = 1:n
        [ri, lost, lab(i)] = inj.apply(nd, nd.sense(truth(i)), truth(i), t(i), dt);
        if ~lost, r(i) = ri; end
    end
end

function res = check(name, ok, value)
    if ok, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('%s  %-48s (measured %.4g)\n', tag, name, value);
    res = struct('name', name, 'pass', logical(ok), 'value', value);
end
