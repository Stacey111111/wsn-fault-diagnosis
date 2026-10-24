function R = simulate_run(seed, opts)
%SIMULATE_RUN  Simulate one random network with random faults and events.
%
%   R = simulate_run(seed, opts)
%     seed  random seed of this run (network layout, faults, events, noise)
%     opts  optional struct overriding the defaults below
%
%   Defaults (one run = one independent network):
%     N = 50 nodes, 100 m x 100 m field, multi-hop routing to the sink at
%     (50, 100) m with a 30 m radio range, 864 periods of 5 minutes (72 h),
%     faultFrac = 0.5 of the nodes get one random fault (types balanced),
%     nExtra = 2 extra faults added to nodes that already have one
%     (simultaneous faults), nEvents = 1 environmental heat event.
%
%   R is a struct with the T x N logs that the sink sees plus ground truth:
%     time, reading, truth, label, label2, event, voltage, energy, parent,
%     relayLost, and positions, neighbors, sink, faults, events, seed.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 2, opts = struct(); end
    d = struct('N', 50, 'nSteps', 864, 'commRange', 30, 'faultFrac', 0.5, ...
               'nExtra', 2, 'nEvents', 1, 'tStart', 4, 'tEnd', 72);
    f = fieldnames(opts);
    for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end
    rng(seed);

    sim = WSNSimulator(struct('N', d.N, 'protocol', 'multihop', 'commRange', d.commRange));
    inj = FaultInjector();
    list = inj.addRandomFaults(sim, d.faultFrac, struct('tStart', d.tStart, 'tEnd', d.tEnd));

    % simultaneous faults: a second fault of another transient type on a node
    % that already has one, starting inside the first fault's window
    transient = {'offset', 'gain', 'stuck', 'spike', 'noise', 'loss'};
    pick = randperm(numel(list), min(d.nExtra, numel(list)));
    for j = pick
        F = list(j);
        cand = transient(~strcmp(transient, F.type));
        ty = cand{randi(numel(cand))};
        t0 = F.t0 + 1 + 2 * rand();
        t1 = t0 + 4 + 8 * rand();
        inj.addFault(F.node, ty, t0, t1, FaultInjector.randomSeverity(ty));
    end

    % environmental events (true anomalies, labelled normal)
    for e = 1:d.nEvents
        ex = 15 + 70 * rand();  ey = 15 + 70 * rand();
        radius = 25 + 15 * rand();
        t0 = 12 + 44 * rand();  dur = 4 + 4 * rand();
        amp = (5 + 5 * rand()) * sign(randn());          % heat or cold front
        inj.addEvent(ex, ey, radius, t0, t0 + dur, amp);
    end

    sim.attachInjector(inj);
    sim.run(d.nSteps);

    L = sim.log;
    xy = cell2mat(cellfun(@(nd) [nd.x nd.y], sim.nodes, 'UniformOutput', false));
    R = struct('seed', seed, 'time', L.time(:), 'reading', L.reading, 'truth', L.truth, ...
               'label', L.label, 'label2', L.label2, 'event', L.event, ...
               'voltage', L.voltage, 'energy', L.energy, 'parent', L.parent, ...
               'relayLost', L.relayLost, 'positions', xy, 'neighbors', sim.neighbors, ...
               'sink', sim.sink);
    R.faults = inj.faults;
    R.events = inj.eventList;
end
