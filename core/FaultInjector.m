classdef FaultInjector < handle
%FAULTINJECTOR  Inject labelled sensor, communication and power faults, and
%   environmental events, into a WSNSimulator run.
%
%   inj = FaultInjector();
%   inj.addFault(nodeId, type, t0, t1, params)   one fault on one node
%   inj.addRandomFaults(sim, frac)               random faults on a fraction
%                                                of the nodes (random type,
%                                                onset, duration, severity)
%   inj.addEvent(x, y, radius, t0, t1, amp)      real environmental event
%   sim.attachInjector(inj); sim.run(nSteps);    faults are applied on-line
%
%   Fault classes (label = position in classNames - 1), as in the Week 4
%   taxonomy:
%     0 normal   1 offset   2 gain    3 drift   4 stuck
%     5 spike    6 noise    7 loss    8 battery
%
%   params fields per type (defaults in brackets):
%     offset : bias [3]              gain : gain [1.25]
%     drift  : rate [0.15 deg C/h]   stuck: (value = reading at onset)
%     spike  : prob [0.05], amp [8]  noise: std [1.5]
%     loss   : prob [0.4] long-run loss rate of a bursty Gilbert-Elliott link
%     battery: capFrac [0.12] remaining capacity at onset (fraction of E0),
%              leak [2] drain multiplier, vLow [2.5 V] voltage below which
%              readings degrade. Battery effects follow the node's actual
%              energy (Week 3 radio model), so a busy node degrades sooner.
%              Random battery faults start at 3-6% of E0 (leak x2-3), so the
%              degradation shows within a 72 h run even on a lightly loaded
%              leaf node of a multi-hop network.
%
%   Several faults may be active on one node at the same time. They are
%   applied in the order calibration -> stuck -> noise -> spike -> battery ->
%   loss. The primary label is the fault with the largest effect on the
%   sample; the second largest is returned as the secondary label.
%   Readings outside the sensor range are saturated at the range limit
%   (out-of-bounds is treated as a symptom, not a separate class).
%
%   Week 6 deliverable, CS 5395 Independent Study (Xiaohui San).

    properties
        classNames = {'normal', 'offset', 'gain', 'drift', 'stuck', 'spike', 'noise', 'loss', 'battery'};
        faults = struct('node', {}, 'type', {}, 'label', {}, 't0', {}, 't1', {}, 'params', {}, 'state', {});
        eventList = struct('x', {}, 'y', {}, 'radius', {}, 't0', {}, 't1', {}, 'amp', {});
        byNode = {}          % byNode{i} = indices into faults for node i
    end

    properties (Constant)
        ORDER = {'offset', 'gain', 'drift', 'stuck', 'noise', 'spike', 'battery', 'loss'};
    end

    methods
        function k = addFault(obj, node, type, t0, t1, params)
            %ADDFAULT  Add a fault of the given type to node, active in [t0, t1).
            if nargin < 5 || isempty(t1), t1 = Inf; end
            if nargin < 6, params = struct(); end
            lab = find(strcmpi(obj.classNames, type)) - 1;
            if isempty(lab) || lab == 0
                error('FaultInjector:type', 'Unknown fault type "%s".', type);
            end
            p = FaultInjector.defaults(lower(type));
            f = fieldnames(params);
            for i = 1:numel(f), p.(f{i}) = params.(f{i}); end
            k = numel(obj.faults) + 1;
            obj.faults(k) = struct('node', node, 'type', lower(type), 'label', lab, ...
                                   't0', t0, 't1', t1, 'params', p, 'state', struct('init', false, 'bad', false));
            if numel(obj.byNode) < node, obj.byNode{node} = []; end
            obj.byNode{node} = [obj.byNode{node} k];
        end

        function addEvent(obj, x, y, radius, t0, t1, amp)
            %ADDEVENT  Environmental event (not a fault): the true field rises by
            %   up to amp inside a Gaussian region of the given radius.
            obj.eventList(end + 1) = struct('x', x, 'y', y, 'radius', radius, 't0', t0, 't1', t1, 'amp', amp);
        end

        function list = addRandomFaults(obj, sim, frac, opts)
            %ADDRANDOMFAULTS  Put one random fault on round(frac*N) random nodes.
            %   opts.types   fault types to draw from (default: all 8, balanced)
            %   opts.tStart  earliest onset [h] (default 6)
            %   opts.tEnd    end of the simulation [h] (default 72)
            %   opts.minDur, opts.maxDur  duration of transient faults [h] (4, 24)
            if nargin < 4, opts = struct(); end
            d = struct('types', {obj.classNames(2:end)}, 'tStart', 6, 'tEnd', 72, ...
                       'minDur', 4, 'maxDur', 24);
            f = fieldnames(opts);
            for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end
            nF = round(frac * sim.N);
            nodesSel = randperm(sim.N, nF);
            types = d.types(mod(0:nF - 1, numel(d.types)) + 1);   % balanced
            types = types(randperm(nF));
            list = struct('node', {}, 'type', {}, 't0', {}, 't1', {});
            for j = 1:nF
                ty = types{j};
                t0 = d.tStart + (d.tEnd - d.maxDur - d.tStart) * rand();
                if any(strcmp(ty, {'drift', 'battery'}))
                    t1 = Inf;                                     % permanent
                else
                    t1 = t0 + d.minDur + (d.maxDur - d.minDur) * rand();
                end
                p = FaultInjector.randomSeverity(ty);
                obj.addFault(nodesSel(j), ty, t0, t1, p);
                list(end + 1) = struct('node', nodesSel(j), 'type', ty, 't0', t0, 't1', t1); %#ok<AGROW>
            end
        end

        function [r, lost, lab, lab2, drain] = apply(obj, node, clean, truth, t, dt) %#ok<INUSD>
            %APPLY  Apply every fault active on this node at time t.
            r = clean;  lost = false;  drain = 1;  lab = 0;  lab2 = 0;
            ks = obj.faultsOf(node.id);
            if isempty(ks), return; end
            ks = ks([obj.faults(ks).t0] <= t);          % faults that have started
            if isempty(ks), return; end
            % order of application
            rank = cellfun(@(ty) find(strcmp(FaultInjector.ORDER, ty)), {obj.faults(ks).type});
            [~, o] = sort(rank);  ks = ks(o);
            eff = zeros(size(ks));
            for j = 1:numel(ks)
                k = ks(j);  F = obj.faults(k);  p = F.params;
                if t >= F.t1 && ~strcmp(F.type, 'battery'), continue; end
                before = r;
                switch F.type
                    case 'offset'
                        r = r + p.bias;
                    case 'gain'
                        r = p.gain * r;
                    case 'drift'
                        r = r + p.rate * (t - F.t0);
                    case 'stuck'
                        if ~F.state.init
                            obj.faults(k).state.value = r;  obj.faults(k).state.init = true;
                        end
                        r = obj.faults(k).state.value;
                    case 'noise'
                        r = r + p.std * randn();
                    case 'spike'
                        if rand() < p.prob
                            r = r + p.amp * sign(randn()) * (0.6 + 0.4 * rand());
                        end
                    case 'battery'
                        if ~F.state.init              % capacity fade at onset
                            node.energy = min(node.energy, p.capFrac * node.E0);
                            obj.faults(k).state.init = true;
                        end
                        drain = drain * p.leak;
                        v = node.voltage();
                        if v < p.vLow
                            sev = min(max((p.vLow - v) / (p.vLow - node.vMin), 0), 1);
                            r = r - 3 * sev^2 + (0.2 + 1.5 * sev^2) * randn();
                            if rand() < 0.6 * sev^3, lost = true; end
                            eff(j) = 10;              % degraded power dominates
                        else
                            % fault active but readings not yet degraded: the low
                            % capacity already shows in the reported voltage, so
                            % the sample is labelled battery, not normal
                            eff(j) = 1e-6;
                        end
                    case 'loss'
                        st = obj.faults(k).state;
                        pGB = 0.05;  pBG = pGB * (1 - p.prob) / p.prob;
                        if st.bad, st.bad = rand() > pBG; else, st.bad = rand() < pGB; end
                        obj.faults(k).state = st;
                        if st.bad, lost = true; end
                        eff(j) = 5;                   % link fault labelled over its window
                end
                if ~any(strcmp(F.type, {'battery', 'loss'}))
                    eff(j) = abs(r - before);
                    if strcmp(F.type, 'noise'), eff(j) = p.std; end
                    if strcmp(F.type, 'stuck'), eff(j) = max(eff(j), 0.5); end
                    if ~strcmp(F.type, 'spike'), eff(j) = max(eff(j), 1e-6); end   % active => labelled
                end
            end
            r = min(max(r, node.range(1)), node.range(2));     % out-of-bounds -> saturate
            [e, o] = sort(eff, 'descend');
            labs = [obj.faults(ks(o)).label];
            labs = labs(e > 0);
            if ~isempty(labs), lab = labs(1); end
            if numel(labs) > 1, lab2 = labs(2); end
        end

        function [lab, lab2] = labelsFor(obj, node, t)
            %LABELSFOR  Labels of a dead node (no reading): battery if its
            %   battery fault killed it, otherwise normal depletion (0).
            lab = 0;  lab2 = 0;
            ks = obj.faultsOf(node.id);
            for k = ks
                if strcmp(obj.faults(k).type, 'battery') && t >= obj.faults(k).t0
                    lab = 8;
                end
            end
        end

        function d = drainFactor(obj, node)
            d = 1;
            for k = obj.faultsOf(node.id)
                F = obj.faults(k);
                if strcmp(F.type, 'battery') && F.state.init, d = d * F.params.leak; end
            end
        end

        function v = eventDelta(obj, x, y, t)
            %EVENTDELTA  Temperature added to the true field by active events.
            v = 0;
            for e = obj.eventList
                if t >= e.t0 && t < e.t1
                    w = exp(-((x - e.x)^2 + (y - e.y)^2) / (2 * (e.radius / 2)^2));
                    v = v + e.amp * w * sin(pi * (t - e.t0) / (e.t1 - e.t0));
                end
            end
        end

        function tf = inEvent(obj, x, y, t)
            tf = false;
            for e = obj.eventList
                if t >= e.t0 && t < e.t1 && hypot(x - e.x, y - e.y) <= e.radius
                    tf = true;
                end
            end
        end

        function T = faultTable(obj)
            %FAULTTABLE  Printable list of all injected faults.
            T = obj.faults;
            fprintf('%4s %6s %-8s %7s %7s  %s\n', '#', 'Node', 'Type', 't0 (h)', 't1 (h)', 'Severity');
            for k = 1:numel(T)
                fprintf('%4d %6d %-8s %7.1f %7.1f  %s\n', k, T(k).node, T(k).type, T(k).t0, ...
                        T(k).t1, FaultInjector.describe(T(k).type, T(k).params));
            end
        end
    end

    methods (Access = private)
        function ks = faultsOf(obj, id)
            if id <= numel(obj.byNode), ks = obj.byNode{id}; else, ks = []; end
        end
    end

    methods (Static)
        function p = defaults(type)
            switch type
                case 'offset',  p = struct('bias', 3);
                case 'gain',    p = struct('gain', 1.25);
                case 'drift',   p = struct('rate', 0.15);
                case 'stuck',   p = struct();
                case 'spike',   p = struct('prob', 0.05, 'amp', 8);
                case 'noise',   p = struct('std', 1.5);
                case 'loss',    p = struct('prob', 0.4);
                case 'battery', p = struct('capFrac', 0.12, 'leak', 2, 'vLow', 2.5);
                otherwise,      p = struct();
            end
        end

        function p = randomSeverity(type)
            %RANDOMSEVERITY  Draw a random severity so that subtle and obvious
            %   faults both appear in the data (ranges from the Week 4 document).
            u = @(a, b) a + (b - a) * rand();
            s = sign(randn());
            switch type
                case 'offset',  p = struct('bias', s * u(1, 5));
                case 'gain',    if s > 0, p = struct('gain', u(1.1, 1.4)); else, p = struct('gain', u(0.7, 0.9)); end
                case 'drift',   p = struct('rate', s * u(0.05, 0.3));
                case 'stuck',   p = struct();
                case 'spike',   p = struct('prob', u(0.02, 0.08), 'amp', u(5, 12));
                case 'noise',   p = struct('std', u(0.8, 2.5));
                case 'loss',    p = struct('prob', u(0.2, 0.6));
                case 'battery', p = struct('capFrac', u(0.03, 0.06), 'leak', u(2, 3), 'vLow', 2.5);
            end
        end

        function s = describe(type, p)
            switch type
                case 'offset',  s = sprintf('bias = %+.2f degC', p.bias);
                case 'gain',    s = sprintf('gain = %.2f', p.gain);
                case 'drift',   s = sprintf('rate = %+.3f degC/h', p.rate);
                case 'stuck',   s = 'value frozen at onset';
                case 'spike',   s = sprintf('prob = %.3f, amp = %.1f degC', p.prob, p.amp);
                case 'noise',   s = sprintf('std = %.2f degC', p.std);
                case 'loss',    s = sprintf('loss rate = %.2f (bursty)', p.prob);
                case 'battery', s = sprintf('capacity = %.0f%% of E0, leak x%.1f', 100 * p.capFrac, p.leak);
                otherwise,      s = '';
            end
        end
    end
end
