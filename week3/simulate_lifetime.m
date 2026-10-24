function res = simulate_lifetime(net, protocol, k, maxRounds, p, P, R)
%SIMULATE_LIFETIME  Round-based network lifetime simulation.
%
%   res = simulate_lifetime(net, protocol, k, maxRounds, p, P, R)
%     net       struct from deploy_nodes()
%     protocol  'direct'       - every node sends its packet straight to the sink
%               'leach'        - LEACH clustering: randomly rotating cluster
%                                heads collect, aggregate and forward data
%               'multihop'     - every node talks only to neighbours within R;
%                                packets are relayed hop by hop to the sink
%                                along minimum-energy paths (route_tree.m),
%                                and every relay forwards each packet as is
%               'multihop-agg' - same routes, but every relay fuses what it
%                                received with its own reading into one
%                                k-bit packet (in-network aggregation)
%     k         packet size in bits (default 4000)
%     maxRounds stop after this many rounds (default 3000)
%     p         desired cluster-head fraction for LEACH (default 0.05)
%     P         radio parameters (default radio_params())
%     R         radio range for the multi-hop protocols [m] (default 30)
%
%   Returns struct res with per-round traces
%     alive(r)     number of nodes with energy > 0 after round r
%     reach(r)     number of alive nodes with a route to the sink in round r
%                  (equal to alive(r) for 'direct' and 'leach')
%     Etotal(r)    total residual energy after round r [J]
%     nCH(r)       number of cluster heads in round r (LEACH)
%     FND, HND, LND  rounds at which the first node, half of the nodes and
%                    the last node were lost (NaN if not reached). A node is
%                    lost when its battery is empty or, in the multi-hop
%                    protocols, when it is alive but cut off from the sink
%                    because the relays around the sink have died. For
%                    'direct' and 'leach' every alive node reaches the sink,
%                    so these are the usual first/half/last node death.
%     FDN          round of the first battery death (equals FND except when
%                  a multi-hop node is cut off before any node dies)
%     lastCH       logical vector of cluster heads in the final round
%     parent       next hop of every node in the last round (multi-hop)
%
%   CS 5395 Independent Study (Xiaohui San).

    if nargin < 3 || isempty(k),         k = 4000;           end
    if nargin < 4 || isempty(maxRounds), maxRounds = 3000;   end
    if nargin < 5 || isempty(p),         p = 0.05;           end
    if nargin < 6 || isempty(P),         P = radio_params(); end
    if nargin < 7 || isempty(R),         R = 30;             end

    N = numel(net.x);
    E = net.E;
    lastCHepoch = -inf(N, 1);        % epoch in which each node was last CH
    epoch = round(1 / p);

    res.alive  = nan(maxRounds, 1);
    res.reach  = nan(maxRounds, 1);
    res.Etotal = nan(maxRounds, 1);
    res.nCH    = zeros(maxRounds, 1);
    res.FND = NaN; res.HND = NaN; res.LND = NaN; res.FDN = NaN;
    res.lastCH = false(N, 1);
    res.parent = -ones(N, 1);
    isMH = any(strcmpi(protocol, {'multihop', 'multihop-agg'}));
    if isMH                          % initial routing tree
        [parent, ~, order, reach] = route_tree(net.D, net.dSink, E > 0, R, k, P);
        res.parent = parent;
    end

    for r = 1:maxRounds
        alive = E > 0;
        if ~any(alive)
            res.alive(r:end) = 0; res.reach(r:end) = 0; res.Etotal(r:end) = 0;
            break
        end
        spent = zeros(N, 1);

        switch lower(protocol)
            case 'direct'
                spent(alive) = radio_energy('tx', k, net.dSink(alive), P);

            case 'leach'
                % --- cluster-head election (LEACH threshold) ---
                % eligible: nodes that have not been CH in the current epoch
                % (Heinzelman et al. 2002, Sec. III-A, Eq. (3))
                r0 = r - 1;  ep = floor(r0 / epoch);
                eligible = alive & (lastCHepoch < ep);
                T = p / (1 - p * mod(r0, epoch));
                isCH = eligible & (rand(N, 1) < T);
                lastCHepoch(isCH) = ep;
                res.nCH(r) = sum(isCH);
                res.lastCH = isCH;

                chIdx = find(isCH);
                members = find(alive & ~isCH);
                if isempty(chIdx)
                    % no cluster head this round: everyone sends directly
                    spent(alive) = radio_energy('tx', k, net.dSink(alive), P);
                else
                    % --- members join the nearest cluster head ---
                    [dCH, j] = min(net.D(members, chIdx), [], 2);
                    toSinkIsCloser = net.dSink(members) < dCH;
                    dSend = dCH;
                    dSend(toSinkIsCloser) = net.dSink(members(toSinkIsCloser));
                    spent(members) = radio_energy('tx', k, dSend, P);
                    % --- cluster heads receive, aggregate and forward ---
                    joined = j(~toSinkIsCloser);
                    nMem = accumarray(joined(:), 1, [numel(chIdx) 1]);
                    spent(chIdx) = nMem .* radio_energy('rx', k, [], P) ...
                                 + (nMem + 1) .* radio_energy('da', k, [], P) ...
                                 + radio_energy('tx', k, net.dSink(chIdx), P);
                end

            case {'multihop', 'multihop-agg'}
                % routes were built for the current alive set (see below)
                % packets each node sends to its next hop (leaves first);
                % nodes cut off from the sink do not transmit
                nIn = zeros(N, 1);                    % packets received
                nOut = zeros(N, 1);                   % packets sent
                for i = order(:)'
                    if strcmpi(protocol, 'multihop')
                        nOut(i) = nIn(i) + 1;         % relay everything as is
                    else
                        nOut(i) = 1;                  % fuse into one packet
                    end
                    if parent(i) > 0, nIn(parent(i)) = nIn(parent(i)) + nOut(i); end
                end
                dNext = net.dSink;
                hasNode = reach & parent > 0;
                dNext(hasNode) = net.D(sub2ind([N N], find(hasNode), parent(hasNode)));
                spent(reach) = nOut(reach) .* radio_energy('tx', k, dNext(reach), P) ...
                             + nIn(reach) .* radio_energy('rx', k, [], P);
                if strcmpi(protocol, 'multihop-agg')
                    % only relays fuse packets: a leaf has nothing to aggregate
                    nAgg = (nIn + 1) .* (nIn > 0);
                    spent(reach) = spent(reach) + nAgg(reach) .* radio_energy('da', k, [], P);
                end

            otherwise
                error('simulate_lifetime:protocol', ...
                      'protocol must be ''direct'', ''leach'', ''multihop'' or ''multihop-agg''.');
        end

        E = max(E - spent, 0);
        nAlive = sum(E > 0);
        if isMH
            % rebuild the routing tree whenever a node has died; nodes whose
            % relays are all dead are cut off from the sink
            if nAlive < sum(alive)
                [parent, ~, order, reach] = route_tree(net.D, net.dSink, E > 0, R, k, P);
                res.parent = parent;
            end
            nReach = sum(reach);
        else
            nReach = nAlive;
        end
        res.alive(r)  = nAlive;
        res.reach(r)  = nReach;
        res.Etotal(r) = sum(E);
        % a node is lost when it is dead or can no longer reach the sink;
        % for 'direct' and 'leach' nReach equals nAlive
        if isnan(res.FND) && nReach < N,     res.FND = r; end
        if isnan(res.HND) && nReach <= N/2,  res.HND = r; end
        if isnan(res.LND) && nReach == 0,    res.LND = r; end
        if isnan(res.FDN) && nAlive < N,     res.FDN = r; end
    end
    res.rounds = (1:maxRounds)';
    res.protocol = protocol;
end
