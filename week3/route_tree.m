function [parent, hops, order, reach] = route_tree(D, dSink, alive, R, k, P)
%ROUTE_TREE  Minimum-energy multi-hop routing tree towards the sink.
%
%   [parent, hops, order, reach] = route_tree(D, dSink, alive, R, k, P)
%     D      N x N distance matrix between nodes [m]
%     dSink  N x 1 distance from each node to the sink [m]
%     alive  N x 1 logical, nodes that can send and relay
%     R      radio range [m]: a node can only talk to nodes (or the sink)
%            within R, so packets from distant nodes are relayed hop by hop
%     k      packet size used for the link cost [bits] (default 4000)
%     P      radio parameters (default radio_params())
%
%   The cost of a hop from node i to node j is the energy the two nodes
%   spend on one k-bit packet, E_tx(k, d_ij) + E_rx(k); the last hop to the
%   sink costs only E_tx(k, d_i,sink), because the sink is mains powered.
%   Dijkstra's algorithm from the sink gives every node the path with the
%   least total energy. With the default radio parameters every hop is
%   shorter than d0 = 87.7 m, and the fixed electronics cost (2 k E_elec)
%   dominates, so the paths are also close to minimum-hop paths.
%
%   Returns
%     parent  N x 1 next hop of each node: j > 0 another node, 0 the sink,
%             -1 no route (dead node, or alive but cut off from the sink)
%     hops    N x 1 number of hops to the sink (Inf if no route)
%     order   alive reachable nodes sorted from the farthest (most hops) to
%             the nearest, so packets can be forwarded leaves-first
%     reach   N x 1 logical, alive nodes that have a route to the sink
%
%   CS 5395 Independent Study (Xiaohui San).

    if nargin < 5 || isempty(k), k = 4000; end
    if nargin < 6 || isempty(P), P = radio_params(); end
    N = numel(dSink);
    alive = logical(alive(:));
    dSink = dSink(:);

    cost   = inf(N, 1);
    parent = -ones(N, 1);
    first  = alive & dSink <= R;                 % can reach the sink in one hop
    cost(first)   = radio_energy('tx', k, dSink(first), P);
    parent(first) = 0;

    Erx  = radio_energy('rx', k, [], P);
    done = ~alive;                               % dead nodes never relay
    for it = 1:N
        cand = cost;  cand(done) = Inf;
        [cu, u] = min(cand);
        if ~isfinite(cu), break; end             % remaining nodes are cut off
        done(u) = true;
        nb = find(~done & D(:, u) <= R);
        if isempty(nb), continue; end
        c = cu + radio_energy('tx', k, D(nb, u), P) + Erx;
        better = c < cost(nb);
        cost(nb(better))   = c(better);
        parent(nb(better)) = u;
    end
    parent(~alive) = -1;
    reach = alive & parent >= 0;

    % hop counts by walking the tree (at most N hops)
    hops = inf(N, 1);
    hops(reach & parent == 0) = 1;
    for it = 2:N
        nxt = reach & ~isfinite(hops) & parent > 0;
        if ~any(nxt), break; end
        idx = find(nxt);
        ph = hops(parent(idx));
        ok = isfinite(ph);
        hops(idx(ok)) = ph(ok) + 1;
    end

    idx = find(reach);
    [~, s] = sort(hops(idx), 'descend');
    order = idx(s);
end
