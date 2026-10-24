function results = test_multihop()
%TEST_MULTIHOP  Unit tests for multi-hop routing (route_tree.m) and
%   multi-hop forwarding in WSNSimulator.
%   results = test_multihop() runs every test, prints PASS/FAIL and returns
%   a struct array with the name, pass flag and measured value.
%
%   Most tests use a chain of four nodes 25 m apart below the sink, so that
%   with a 30 m radio range node 4 must go through 3, 2 and 1:
%       sink (50,100) <- 1 (50,75) <- 2 (50,50) <- 3 (50,25) <- 4 (50,0)

    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week3'));
    rng(12);
    results = struct('name', {}, 'pass', {}, 'value', {});
    chain = [50 75; 50 50; 50 25; 50 0];
    noLoss = struct('p0', 0, 'd50', 1e4);            % switch off random link loss

    % 1. routing tree on the chain
    D = hypot(chain(:, 1) - chain(:, 1)', chain(:, 2) - chain(:, 2)');
    dS = hypot(chain(:, 1) - 50, chain(:, 2) - 100);
    [par, hops] = route_tree(D, dS, true(4, 1), 30);
    results(end + 1) = check('chain: parents 0-1-2-3, hops 1-4', ...
        isequal(par(:)', [0 1 2 3]) && isequal(hops(:)', 1:4), max(hops));

    % 2. random 50-node network: every hop is within range and every path
    %    reaches the sink in exactly hops(i) steps
    xy = 100 * rand(50, 2);
    D = hypot(xy(:, 1) - xy(:, 1)', xy(:, 2) - xy(:, 2)');
    dS = hypot(xy(:, 1) - 50, xy(:, 2) - 100);
    [par, hops, ~, reach] = route_tree(D, dS, true(50, 1), 30);
    ok = true;
    for i = find(reach)'
        j = i;  steps = 0;
        while j > 0
            if par(j) == 0, d = dS(j); else, d = D(j, par(j)); end
            ok = ok && d <= 30;
            j = par(j);  steps = steps + 1;
            if steps > 50, ok = false; break; end
        end
        ok = ok && steps == hops(i);
    end
    results(end + 1) = check('random: hops within 30 m, paths end at sink', ok, sum(reach));

    % 3. no faults, no link loss: every reading reaches the sink
    sim = newChain(chain, noLoss);
    sim.run(100);
    dr = mean(~isnan(sim.log.reading(:)));
    results(end + 1) = check('chain: delivery = 100% without loss', dr == 1, dr);

    % 4. relays spend more energy than the leaf (energy hole)
    used = sim.opts.E0 - sim.log.energy(end, :);
    results(end + 1) = check('chain: node 1 (3 children) uses most energy', ...
        all(diff(used) < 0), used(1) / used(4));

    % 5. a link fault on relay 2 also removes the readings of nodes 3 and 4
    sim = newChain(chain, noLoss);
    inj = FaultInjector();
    inj.addFault(2, 'loss', 0, Inf, struct('prob', 0.6));
    sim.attachInjector(inj);
    sim.run(400);
    L = sim.log;
    down = L.label(:, 2) == 7 & isnan(L.reading(:, 2));
    ok = any(down) && all(all(isnan(L.reading(down, 3:4)))) && all(all(L.relayLost(down, 3:4))) ...
         && all(all(L.label(:, 3:4) == 0)) && all(~isnan(L.reading(:, 1)));
    results(end + 1) = check('relay fault: children lost, still labelled normal', ok, mean(down));

    % 6. route repair: node 2 dies -> nodes 3 and 4 are cut off
    sim = newChain(chain, noLoss);
    sim.run(5);
    sim.nodes{2}.spend(sim.nodes{2}.energy);
    sim.run(5);
    L = sim.log;
    ok = all(L.parent(end, :) == [0 -1 -1 -1]) && all(all(isnan(L.reading(end-4:end, 2:4)))) ...
         && all(~isnan(L.reading(end-4:end, 1)));
    results(end + 1) = check('route repair: dead relay cuts off nodes 3, 4', ok, sum(L.parent(end, :) < 0));

    % 7. route repair with an alternative path: extra node beside node 2
    pos = [chain; 62 50];                       % node 5 links node 1 and node 3
    sim = newChain(pos, noLoss);
    sim.run(3);
    sim.nodes{2}.spend(sim.nodes{2}.energy);
    sim.run(3);
    p = sim.log.parent(end, :);
    ok = p(3) == 5 && p(5) == 1 && p(4) == 3 && all(~isnan(sim.log.reading(end, [1 3 4 5])));
    results(end + 1) = check('route repair: node 3 switches to node 5', ok, p(3));

    fprintf('\n%d of %d tests passed\n', sum([results.pass]), numel(results));
end

function sim = newChain(pos, extra)
    o = struct('protocol', 'multihop', 'positions', pos, 'commRange', 30);
    f = fieldnames(extra);
    for i = 1:numel(f), o.(f{i}) = extra.(f{i}); end
    sim = WSNSimulator(o);
end

function res = check(name, ok, value)
    if ok, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('%s  %-48s (measured %.4g)\n', tag, name, value);
    res = struct('name', name, 'pass', logical(ok), 'value', value);
end
