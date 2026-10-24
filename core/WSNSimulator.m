classdef WSNSimulator < handle
%WSNSIMULATOR  Time-stepped simulator of a wireless sensor network with
%   multi-hop, LEACH or direct data delivery.
%
%   sim = WSNSimulator(opts)      opts fields override the defaults below
%   sim.run(nSteps)               simulate nSteps sampling periods
%   sim.plotNetwork()             draw nodes, routes or clusters, and sink
%   S = sim.summary()             delivery ratio, energy use, alive nodes
%
%   Every sampling period (default 5 minutes) each alive node
%     1. senses the physical field at its position (SensorNode.sense),
%     2. passes the reading through the FaultInjector, if one is attached,
%     3. sends its reading towards the sink, paying the first-order radio
%        energy (radio_energy.m, Week 3),
%     4. may lose the packet on each wireless hop (distance-dependent loss).
%
%   protocol = 'multihop' (default): a node talks only to neighbours within
%     commRange. Packets travel hop by hop to the sink along the
%     minimum-energy routing tree (route_tree.m, Week 3), which is rebuilt
%     whenever a node dies. Each relay merges the readings it received with
%     its own into one packet (compression below) and forwards it to its
%     parent. If a relay's outgoing link fails, every reading it carries in
%     that period is lost. Alive nodes whose relays have all died are cut off
%     and their readings never reach the sink.
%   protocol = 'leach': cluster heads are re-elected with the LEACH rule every
%     roundSteps periods, receive their members' packets and send them to
%     the sink in one hop (no range limit).
%   protocol = 'direct': every node sends straight to the sink (no range limit).
%
%   Everything the sink sees and the ground truth are logged (T x N):
%     log.truth    true field value at each node
%     log.clean    reading before faults (sensor noise only)
%     log.reading  reading received at the sink (NaN = not received)
%     log.label    primary fault label (0 = normal, see FaultInjector)
%     log.label2   secondary label for simultaneous faults (0 = none)
%     log.event    true if the node is inside an environmental event
%     log.energy, log.voltage, log.isCH, log.alive
%     log.parent   next hop of each node (multi-hop): j > 0 node, 0 sink,
%                  -1 no route (dead or cut off)
%     log.relayLost  true if the node's own hop succeeded but its reading
%                  was lost later on the path, at a relay (multi-hop)
%
%   Default options
%     N = 10, area = [100 100] m, sink = [50 100] m (middle of the top edge),
%     deploy = 'random', E0 = 0.5 J, k = 512 bits, dt = 5/60 h,
%     protocol = 'multihop', commRange = 30 m,
%     connected = true: random positions are redrawn until every node has a
%                 multi-hop route to the sink (multi-hop only),
%     LEACH: p = 0.2, roundSteps = 12 (new cluster heads every hour),
%     field: 22 + 6 sin(2 pi (t - 9)/24) + 0.03 x   [deg C]
%     link loss: pLoss(d) = p0 + (1 - p0) / (1 + exp(-(d - d50)/s)),
%                p0 = 0.01, d50 = 200 m, s = 20 m
%     compress = 0.25: a cluster head or relay forwards n readings in
%                k*(1 + 0.25*(n-1)) bits (delta coding of correlated data)
%
%   Requires radio_energy.m, radio_params.m and route_tree.m (week3 folder)
%   on the path.
%
%   Week 5 deliverable, extended with multi-hop routing; CS 5395
%   Independent Study (Xiaohui San).

    properties
        opts            % options struct (see above)
        nodes = {}      % cell array of SensorNode objects
        N               % number of nodes
        sink            % [x y] of the sink
        P               % radio parameters (radio_params)
        injector = []   % FaultInjector object (optional, Week 6)
        neighbors       % N x N logical, true if within commRange
        D               % N x N distance matrix
        dSink           % N x 1 distance from each node to the sink
        parent          % N x 1 next hop in the routing tree (multi-hop)
        hops            % N x 1 hops to the sink (Inf = no route)
        order           % reachable nodes, farthest first (multi-hop)
        routeAlive = [] % alive set the current routing tree was built for
        t = 0           % current time [h]
        step = 0        % number of completed sampling periods
        round = 0       % LEACH round counter
        log             % struct of T x N logs (see help)
        rxBuffer        % per-node buffer of member packets for this step
    end

    methods
        function obj = WSNSimulator(opts)
            if nargin < 1, opts = struct(); end
            d = struct('N', 10, 'area', [100 100], 'sink', [50 100], ...
                       'deploy', 'random', 'E0', 0.5, 'k', 512, 'dt', 5/60, ...
                       'protocol', 'multihop', 'connected', true, 'p', 0.2, 'roundSteps', 12, ...
                       'commRange', 30, 'base', 22, 'amp', 6, 'gradX', 0.03, ...
                       'p0', 0.01, 'd50', 200, 's', 20, 'compress', 0.25, 'Esense', 5e-6, ...
                       'positions', [], 'nodeOpts', struct());
            f = fieldnames(opts);
            for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end
            obj.opts = d;
            obj.P = radio_params();
            obj.sink = d.sink(:)';
            obj.deploy();
        end

        function deploy(obj)
            %DEPLOY  Place the nodes (random, grid, or given positions).
            o = obj.opts;
            isMH = strcmpi(o.protocol, 'multihop');
            if ~isempty(o.positions)
                xy = o.positions;
            else
                switch lower(o.deploy)
                    case 'random'
                        % multi-hop: redraw until every node can reach the sink
                        for tries = 1:5000
                            xy = [o.area(1) * rand(o.N, 1), o.area(2) * rand(o.N, 1)];
                            if ~(isMH && o.connected), break; end
                            Dt = hypot(xy(:, 1) - xy(:, 1)', xy(:, 2) - xy(:, 2)');
                            dS = hypot(xy(:, 1) - obj.sink(1), xy(:, 2) - obj.sink(2));
                            [~, ~, ~, rc] = route_tree(Dt, dS, true(o.N, 1), o.commRange, o.k, obj.P);
                            if all(rc), break; end
                        end
                    case 'grid'
                        nc = ceil(sqrt(o.N));  nr = ceil(o.N / nc);
                        [gx, gy] = meshgrid(((1:nc) - 0.5) * o.area(1) / nc, ((1:nr) - 0.5) * o.area(2) / nr);
                        xy = [gx(:) gy(:)];  xy = xy(1:o.N, :);
                    otherwise
                        error('WSNSimulator:deploy', 'deploy must be ''random'' or ''grid''.');
                end
            end
            obj.N = size(xy, 1);
            nopts = o.nodeOpts;  nopts.E0 = o.E0;
            obj.nodes = cell(obj.N, 1);
            for i = 1:obj.N
                obj.nodes{i} = SensorNode(i, xy(i, 1), xy(i, 2), nopts);
            end
            obj.D = hypot(xy(:, 1) - xy(:, 1)', xy(:, 2) - xy(:, 2)');
            obj.neighbors = obj.D <= o.commRange & ~eye(obj.N);
            obj.dSink = hypot(xy(:, 1) - obj.sink(1), xy(:, 2) - obj.sink(2));
            obj.routeAlive = [];
            if isMH, obj.updateRoutes(true(obj.N, 1)); end
            obj.resetLog(0);
        end

        function attachInjector(obj, inj)
            obj.injector = inj;
        end

        function v = fieldValue(obj, x, y, t)
            %FIELDVALUE  True temperature at (x, y) and time t [h], including
            %   environmental events registered in the fault injector.
            o = obj.opts;
            v = o.base + o.amp * sin(2 * pi * (t - 9) / 24) + o.gradX * x;
            if ~isempty(obj.injector)
                v = v + obj.injector.eventDelta(x, y, t);
            end
        end

        function p = linkLoss(obj, d)
            %LINKLOSS  Probability that a packet sent over d metres is lost.
            o = obj.opts;
            p = o.p0 + (1 - o.p0) ./ (1 + exp(-(d - o.d50) / o.s));
        end

        function run(obj, nSteps)
            %RUN  Simulate nSteps sampling periods and append to the log.
            T0 = obj.step;
            obj.growLog(T0 + nSteps);
            for s = 1:nSteps
                obj.stepOnce();
            end
        end

        function stepOnce(obj)
            o = obj.opts;
            obj.step = obj.step + 1;
            s = obj.step;
            obj.t = (s - 1) * o.dt;
            if strcmpi(o.protocol, 'multihop')
                obj.stepMultihop(s);
                return
            end
            if mod(s - 1, o.roundSteps) == 0
                obj.round = obj.round + 1;
                obj.electClusters();
            end

            delivered = false(obj.N, 1);
            readings = nan(obj.N, 1);
            nBuffered = zeros(obj.N, 1);     % packets received by each cluster head
            viaCH = zeros(obj.N, 1);         % cluster head that relays node i (0 = direct)

            % ---- sensing, faults and first hop ----
            for i = 1:obj.N
                nd = obj.nodes{i};
                truth = obj.fieldValue(nd.x, nd.y, obj.t);
                obj.log.truth(s, i) = truth;
                if ~isempty(obj.injector)
                    obj.log.event(s, i) = obj.injector.inEvent(nd.x, nd.y, obj.t);
                end
                if ~nd.isAlive()
                    lab = 0;  lab2 = 0;
                    if ~isempty(obj.injector), [lab, lab2] = obj.injector.labelsFor(nd, obj.t); end
                    obj.log.label(s, i) = lab;  obj.log.label2(s, i) = lab2;
                    continue
                end
                clean = nd.sense(truth);
                obj.log.clean(s, i) = clean;
                r = clean;  lostByFault = false;  drain = 1;  lab = 0;  lab2 = 0;
                if ~isempty(obj.injector)
                    [r, lostByFault, lab, lab2, drain] = obj.injector.apply(nd, clean, truth, obj.t, o.dt);
                end
                obj.log.label(s, i) = lab;  obj.log.label2(s, i) = lab2;
                nd.spend(o.Esense * drain);

                % first hop: to the cluster head if it is alive, else to the sink
                ch = nd.clusterHead;
                if nd.isCH || ch == 0 || ~obj.nodes{ch}.isAlive()
                    dist = nd.distanceTo(obj.sink);  viaCH(i) = 0;
                else
                    dist = obj.D(i, ch);  viaCH(i) = ch;
                end
                if nd.isCH
                    readings(i) = r;  delivered(i) = ~lostByFault && ~isnan(r);   % own reading, sent below
                    continue
                end
                nd.spend(drain * radio_energy('tx', o.k, dist, obj.P));
                nd.txCount = nd.txCount + 1;
                ok = ~lostByFault && rand() >= obj.linkLoss(dist) && ~isnan(r);
                if viaCH(i) > 0 && ok
                    chNode = obj.nodes{ch};
                    chNode.spend(radio_energy('rx', o.k, [], obj.P));
                    chNode.rxCount = chNode.rxCount + 1;
                    nBuffered(ch) = nBuffered(ch) + 1;
                end
                readings(i) = r;  delivered(i) = ok;
            end

            % ---- cluster heads forward everything they collected ----
            for c = find(cellfun(@(nd) nd.isCH && nd.isAlive(), obj.nodes))'
                nd = obj.nodes{c};
                nPk = nBuffered(c) + delivered(c);       % members + its own reading
                if nPk == 0, continue; end
                % in-network compression: neighbouring readings are strongly
                % correlated, so each extra reading costs only a fraction of k
                bits = o.k * (1 + (nPk - 1) * o.compress);
                dist = nd.distanceTo(obj.sink);
                drain = 1;
                if ~isempty(obj.injector), drain = obj.injector.drainFactor(nd); end
                nd.spend(drain * (radio_energy('da', bits, [], obj.P) + radio_energy('tx', bits, dist, obj.P)));
                nd.txCount = nd.txCount + 1;
                if rand() < obj.linkLoss(dist)           % whole aggregate packet lost
                    delivered(c) = false;
                    delivered(viaCH == c) = false;
                end
            end
            % members whose cluster head died this step lose their packet
            for i = find(viaCH > 0)'
                if ~obj.nodes{viaCH(i)}.isAlive(), delivered(i) = false; end
            end

            readings(~delivered) = NaN;
            obj.log.reading(s, :) = readings';
            obj.logNodeState(s);
        end

        function stepMultihop(obj, s)
            %STEPMULTIHOP  One sampling period with hop-by-hop forwarding.
            o = obj.opts;
            alive = cellfun(@(nd) nd.isAlive(), obj.nodes);
            if ~isequal(alive, obj.routeAlive), obj.updateRoutes(alive); end
            readings = nan(obj.N, 1);
            hasReading = false(obj.N, 1);
            linkDown = false(obj.N, 1);       % outgoing link failed by a fault
            drain = ones(obj.N, 1);

            % ---- sensing and faults ----
            for i = 1:obj.N
                nd = obj.nodes{i};
                truth = obj.fieldValue(nd.x, nd.y, obj.t);
                obj.log.truth(s, i) = truth;
                if ~isempty(obj.injector)
                    obj.log.event(s, i) = obj.injector.inEvent(nd.x, nd.y, obj.t);
                end
                if ~nd.isAlive()
                    lab = 0;  lab2 = 0;
                    if ~isempty(obj.injector), [lab, lab2] = obj.injector.labelsFor(nd, obj.t); end
                    obj.log.label(s, i) = lab;  obj.log.label2(s, i) = lab2;
                    continue
                end
                clean = nd.sense(truth);
                obj.log.clean(s, i) = clean;
                r = clean;  lostByFault = false;  dr = 1;  lab = 0;  lab2 = 0;
                if ~isempty(obj.injector)
                    [r, lostByFault, lab, lab2, dr] = obj.injector.apply(nd, clean, truth, obj.t, o.dt);
                end
                obj.log.label(s, i) = lab;  obj.log.label2(s, i) = lab2;
                nd.spend(o.Esense * dr);
                readings(i) = r;  hasReading(i) = ~isnan(r);
                linkDown(i) = lostByFault;  drain(i) = dr;
            end

            % ---- forwarding, farthest nodes first ----
            par = obj.parent;
            nCarry = zeros(obj.N, 1);          % readings received from children
            hopOK = false(obj.N, 1);
            for i = obj.order(:)'
                nd = obj.nodes{i};
                if ~nd.isAlive(), continue; end           % died this period
                n = nCarry(i) + hasReading(i);
                if n == 0, continue; end
                bits = o.k * (1 + (n - 1) * o.compress);
                if par(i) == 0, dist = obj.dSink(i); else, dist = obj.D(i, par(i)); end
                E = radio_energy('tx', bits, dist, obj.P);
                if n > 1, E = E + radio_energy('da', bits, [], obj.P); end
                nd.spend(drain(i) * E);
                nd.txCount = nd.txCount + 1;
                hopOK(i) = ~linkDown(i) && rand() >= obj.linkLoss(dist);
                if hopOK(i) && par(i) > 0
                    pn = obj.nodes{par(i)};
                    if pn.isAlive()
                        pn.spend(drain(par(i)) * radio_energy('rx', bits, [], obj.P));
                        pn.rxCount = pn.rxCount + 1;
                        nCarry(par(i)) = nCarry(par(i)) + n;
                    else
                        hopOK(i) = false;
                    end
                end
            end
            % a reading arrives only if every hop on its path succeeded
            pathOK = false(obj.N, 1);
            for i = flipud(obj.order(:))'               % nearest nodes first
                pathOK(i) = hopOK(i) && (par(i) == 0 || pathOK(par(i)));
            end
            delivered = hasReading & pathOK;
            readings(~delivered) = NaN;
            obj.log.reading(s, :) = readings';
            obj.log.relayLost(s, :) = (hasReading & hopOK & ~pathOK)';
            obj.log.parent(s, :) = par';
            obj.logNodeState(s);
        end

        function updateRoutes(obj, alive)
            %UPDATEROUTES  Rebuild the minimum-energy routing tree.
            o = obj.opts;
            [obj.parent, obj.hops, obj.order] = route_tree(obj.D, obj.dSink, alive, ...
                                                           o.commRange, o.k, obj.P);
            obj.routeAlive = alive;
        end

        function electClusters(obj)
            %ELECTCLUSTERS  LEACH cluster-head election and cluster formation.
            o = obj.opts;
            % LEACH: in each epoch of 1/p rounds every node serves as cluster
            % head exactly once in expectation. Only nodes that have not been
            % cluster head in the current epoch are eligible
            % (Heinzelman et al. 2002, Sec. III-A, Eq. (3)).
            epoch = max(round(1 / o.p), 1);
            r0 = obj.round - 1;                       % 0-based round index
            ep = floor(r0 / epoch);                   % current epoch
            T = o.p / (1 - o.p * mod(r0, epoch));
            alive = cellfun(@(nd) nd.isAlive(), obj.nodes);
            isCH = false(obj.N, 1);
            if strcmpi(o.protocol, 'leach')
                for i = find(alive)'
                    nd = obj.nodes{i};
                    if nd.lastCHepoch < ep && rand() < T
                        isCH(i) = true;  nd.lastCHepoch = ep;
                    end
                end
            end
            chIdx = find(isCH);
            for i = 1:obj.N
                nd = obj.nodes{i};
                nd.isCH = isCH(i);  nd.clusterHead = 0;
                if ~isCH(i) && ~isempty(chIdx)
                    [dmin, j] = min(obj.D(i, chIdx));
                    if dmin < nd.distanceTo(obj.sink)
                        nd.clusterHead = chIdx(j);
                    end
                end
            end
        end

        function S = summary(obj)
            %SUMMARY  Key statistics of the run so far.
            L = obj.log;  T = obj.step;
            S.steps = T;
            S.hours = T * obj.opts.dt;
            S.deliveryRatio = mean(~isnan(L.reading(1:T, :)), 1);   % per node
            S.meanDelivery = mean(S.deliveryRatio);
            S.energyUsed = obj.opts.E0 - L.energy(T, :);               % per node [J]
            S.aliveAtEnd = sum(L.alive(T, :));
            S.meanCH = mean(sum(L.isCH(1:T, :), 2));
            if strcmpi(obj.opts.protocol, 'multihop')
                h = obj.hops(isfinite(obj.hops));
                S.meanHops = mean(h);  S.maxHops = max([h; 0]);
                S.cutOffAtEnd = sum(L.alive(T, :) & L.parent(T, :) < 0);
                S.relayLostFraction = mean(mean(L.relayLost(1:T, :)));
            end
            lab = L.label(1:T, :);
            S.faultySampleFraction = mean(lab(:) > 0);
        end

        function h = plotNetwork(obj, stepIdx)
            %PLOTNETWORK  Draw nodes, neighbour links, routes or clusters, and sink.
            %   Multi-hop: h = [nodes, nodes one hop from the sink, sink].
            %   LEACH/direct: h = [members, cluster heads, sink].
            if nargin < 2, stepIdx = max(obj.step, 1); end
            hold on;
            xy = cell2mat(cellfun(@(nd) [nd.x nd.y], obj.nodes, 'UniformOutput', false));
            [ii, jj] = find(triu(obj.neighbors));
            for e = 1:numel(ii)
                plot(xy([ii(e) jj(e)], 1), xy([ii(e) jj(e)], 2), '-', 'Color', [0.85 0.85 0.85]);
            end
            if strcmpi(obj.opts.protocol, 'multihop')
                if obj.step >= stepIdx && stepIdx > 0
                    par = obj.log.parent(stepIdx, :)';
                else
                    par = obj.parent;
                end
                for i = find(par >= 0)'
                    if par(i) == 0, q = obj.sink; else, q = xy(par(i), :); end
                    plot([xy(i, 1) q(1)], [xy(i, 2) q(2)], '-', 'Color', [0.3 0.55 0.85], 'LineWidth', 1.3);
                end
                last = par == 0;
                h(1) = plot([xy(~last, 1); NaN], [xy(~last, 2); NaN], 'o', 'MarkerSize', 8, ...
                            'MarkerFaceColor', [0.3 0.55 0.85], 'MarkerEdgeColor', 'k');
                h(2) = plot([xy(last, 1); NaN], [xy(last, 2); NaN], 's', 'MarkerSize', 10, ...
                            'MarkerFaceColor', [1 0.85 0.1], 'MarkerEdgeColor', 'k');
                h(3) = plot(obj.sink(1), obj.sink(2), 'p', 'MarkerSize', 16, ...
                            'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
                cut = par < 0;
                if any(cut)
                    plot(xy(cut, 1), xy(cut, 2), 'x', 'Color', [0.8 0.1 0.1], 'MarkerSize', 12, 'LineWidth', 1.5);
                end
                for i = 1:obj.N
                    text(xy(i, 1) + 2, xy(i, 2) + 3, sprintf('%d', i), 'FontSize', 8);
                end
                plot([0 obj.opts.area(1) obj.opts.area(1) 0 0], [0 0 obj.opts.area(2) obj.opts.area(2) 0], 'k-');
                axis equal; box on;
                xlabel('x (m)'); ylabel('y (m)');
                return
            end
            isCH = logical(obj.log.isCH(stepIdx, :))';
            chIdx = find(isCH);
            for i = 1:obj.N
                % membership at stepIdx, rebuilt with the same rule as
                % electClusters: join the nearest cluster head if it is
                % closer than the sink, otherwise send to the sink directly
                ch = 0;
                if ~isCH(i) && ~isempty(chIdx)
                    [dmin, j] = min(obj.D(i, chIdx));
                    if dmin < obj.nodes{i}.distanceTo(obj.sink), ch = chIdx(j); end
                end
                if ch > 0 && ~isCH(i)
                    plot(xy([i ch], 1), xy([i ch], 2), '-', 'Color', [0.3 0.55 0.85], 'LineWidth', 1.2);
                elseif ~isCH(i)
                    plot([xy(i, 1) obj.sink(1)], [xy(i, 2) obj.sink(2)], ':', 'Color', [0.6 0.6 0.6]);
                end
            end
            for c = find(isCH)'
                plot([xy(c, 1) obj.sink(1)], [xy(c, 2) obj.sink(2)], 'k--', 'LineWidth', 1.1);
            end
            h(1) = plot([xy(~isCH, 1); NaN], [xy(~isCH, 2); NaN], 'o', 'MarkerSize', 8, ...
                        'MarkerFaceColor', [0.3 0.55 0.85], 'MarkerEdgeColor', 'k');
            h(2) = plot([xy(isCH, 1); NaN], [xy(isCH, 2); NaN], 's', 'MarkerSize', 11, ...
                        'MarkerFaceColor', [1 0.85 0.1], 'MarkerEdgeColor', 'k');
            h(3) = plot(obj.sink(1), obj.sink(2), 'p', 'MarkerSize', 16, ...
                        'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
            for i = 1:obj.N
                text(xy(i, 1) + 2, xy(i, 2) + 3, sprintf('%d', i), 'FontSize', 8);
            end
            plot([0 obj.opts.area(1) obj.opts.area(1) 0 0], [0 0 obj.opts.area(2) obj.opts.area(2) 0], 'k-');
            axis equal; box on;
            xlabel('x (m)'); ylabel('y (m)');
        end
    end

    methods (Access = private)
        function logNodeState(obj, s)
            for i = 1:obj.N
                nd = obj.nodes{i};
                obj.log.energy(s, i)  = nd.energy;
                obj.log.voltage(s, i) = nd.voltage();
                obj.log.isCH(s, i)    = nd.isCH;
                obj.log.alive(s, i)   = nd.isAlive();
            end
            obj.log.time(s) = obj.t;
        end

        function resetLog(obj, T)
            z = zeros(T, obj.N);
            obj.log = struct('time', zeros(T, 1), 'truth', z, 'clean', nan(T, obj.N), ...
                'reading', nan(T, obj.N), 'label', z, 'label2', z, 'event', false(T, obj.N), ...
                'energy', z, 'voltage', z, 'isCH', false(T, obj.N), 'alive', false(T, obj.N), ...
                'parent', -ones(T, obj.N), 'relayLost', false(T, obj.N));
        end

        function growLog(obj, T)
            f = fieldnames(obj.log);
            for i = 1:numel(f)
                A = obj.log.(f{i});
                n = size(A, 1);
                if n >= T, continue; end
                if islogical(A)
                    pad = false(T - n, size(A, 2));
                elseif any(strcmp(f{i}, {'clean', 'reading'}))
                    pad = nan(T - n, size(A, 2));
                elseif strcmp(f{i}, 'parent')
                    pad = -ones(T - n, size(A, 2));
                else
                    pad = zeros(T - n, size(A, 2));
                end
                obj.log.(f{i}) = [A; pad];
            end
        end
    end
end
