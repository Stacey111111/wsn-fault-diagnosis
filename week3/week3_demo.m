%% WEEK3_DEMO  Energy consumption model + node deployment visualization
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San 
%
%  Produces (in ../figures):
%    fig3_1_radio_energy.png     transmit/receive energy vs. distance
%    fig3_2_deployment.png       random and grid deployment of 100 nodes
%    fig3_3_routing.png          multi-hop routing tree and one LEACH round
%    fig3_4_lifetime.png         working nodes and residual energy per round
%    fig3_5_large_field.png      large-field comparison: routes and working nodes
%  and prints a lifetime table (FND / HND / LND) for 50, 100 and 200 nodes
%  and four protocols: direct transmission, LEACH, multi-hop relaying and
%  multi-hop with in-network aggregation.
%
%  Section 6 repeats the comparison in a 200 m x 200 m field (sink at the
%  middle of the top edge, up to 224 m away) with 400 nodes, the same node
%  density as 100 nodes in 100 m x 100 m. Most nodes are then beyond
%  d0 = 87.7 m, where direct transmission pays the d^4 amplifier cost.
%
%  Multi-hop: a node talks only to neighbours within R = 30 m, and packets
%  are relayed hop by hop to the sink (route_tree.m). Random deployments
%  are redrawn until every node has a route to the sink, and all four
%  protocols run on the same node positions.

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
figDir = fullfile(here, '..', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
dataDir = fullfile(here, '..', 'data');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
rng(42);                                   % reproducible results

P  = radio_params();
k  = 4000;                                 % bits per data packet
W  = 100; H = 100;                         % field size [m]
sink = [50 100];                           % sink at the middle of the top edge
E0 = 0.5;                                  % initial energy per node [J]
R  = 30;                                   % radio range for multi-hop [m]
protos = {'direct', 'leach', 'multihop', 'multihop-agg'};
pNames = {'Direct', 'LEACH', 'Multi-hop', 'Multi-hop + agg.'};

fprintf('Crossover distance d0 = %.1f m\n', P.d0);

%% 1. Energy vs distance -------------------------------------------------
d = linspace(1, 200, 400);
Etx = radio_energy('tx', k, d, P);
Erx = radio_energy('rx', k, [], P) * ones(size(d));
f1 = figure('Visible', 'off', 'Position', [100 100 720 420]);
semilogy(d, Etx * 1e3, 'LineWidth', 1.8); hold on;
semilogy(d, Erx * 1e3, '--', 'LineWidth', 1.8, 'Color', [0.85 0.33 0.1]);
ylim([0.1 20]); yl = ylim; semilogy([P.d0 P.d0], yl, 'k:', 'LineWidth', 1.2);
text(P.d0 + 3, 8, sprintf('d_0 = %.1f m', P.d0));
xlabel('Distance d (m)'); ylabel('Energy per 4000-bit packet (mJ, log scale)');
title('First-order radio model: transmit vs. receive energy');
legend({'E_{tx}(k,d)', 'E_{rx}(k)', 'crossover d_0'}, 'Location', 'northwest');
grid on;
print(f1, fullfile(figDir, 'fig3_1_radio_energy.png'), '-dpng', '-r150');

%% 2. Node deployment visualization --------------------------------------
for tries = 1:1000                         % redraw until every node reaches the sink
    netR = deploy_nodes(100, W, H, sink, 'random', E0);
    [~, ~, ~, rc] = route_tree(netR.D, netR.dSink, true(100, 1), R, k, P);
    if all(rc), break; end
end
netG = deploy_nodes(100, W, H, sink, 'grid',   E0);
commRange = R;                             % radio range used for the link graph [m]

f2 = figure('Visible', 'off', 'Position', [100 100 1100 480]);
nets = {netR, netG}; names = {'Random deployment', 'Grid deployment'};
for s = 1:2
    subplot(1, 2, s); hold on;
    net = nets{s};
    [ii, jj] = find(triu(net.D <= commRange & net.D > 0));
    for e = 1:numel(ii)
        plot(net.x([ii(e) jj(e)]), net.y([ii(e) jj(e)]), '-', 'Color', [0.82 0.82 0.82]);
    end
    scatter(net.x, net.y, 36, net.dSink, 'filled');
    plot(sink(1), sink(2), 'p', 'MarkerSize', 16, 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
    text(sink(1) + 4, sink(2), 'Sink');
    plot([0 W W 0 0], [0 0 H H 0], 'k-');
    cb = colorbar; ylabel(cb, 'Distance to sink (m)');
    deg = sum(net.D <= commRange & net.D > 0, 2);
    title(sprintf('%s: N = %d, mean degree = %.1f', names{s}, numel(net.x), mean(deg)));
    xlabel('x (m)'); ylabel('y (m)');
    axis equal; xlim([-5 W + 15]); ylim([-5 sink(2) + 12]); box on;
end
print(f2, fullfile(figDir, 'fig3_2_deployment.png'), '-dpng', '-r150');

%% 3. Multi-hop routing tree and one LEACH round on the random network ---
p = 0.05;
[par, hops] = route_tree(netR.D, netR.dSink, true(100, 1), R, k, P);
f3 = figure('Visible', 'off', 'Position', [100 100 1200 560]);
subplot(1, 2, 1); hold on;
for i = 1:100
    if par(i) == 0, q = sink; else, q = [netR.x(par(i)) netR.y(par(i))]; end
    plot([netR.x(i) q(1)], [netR.y(i) q(2)], '-', 'Color', [0.3 0.55 0.85], 'LineWidth', 1.1);
end
scatter(netR.x, netR.y, 48, hops, 'filled', 'MarkerEdgeColor', 'k');
plot(sink(1), sink(2), 'p', 'MarkerSize', 16, 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
cb = colorbar; ylabel(cb, 'Hops to the sink');
title(sprintf('Multi-hop routing tree (R = %d m, up to %d hops)', R, max(hops)));
xlabel('x (m)'); ylabel('y (m)'); axis equal; xlim([-5 W + 5]); ylim([-5 sink(2) + 12]); box on;

subplot(1, 2, 2); hold on;
isCH = false(100, 1);
while sum(isCH) < 3                         % draw a representative round
    isCH = rand(100, 1) < p;
end
chIdx = find(isCH);
[~, j] = min(netR.D(:, chIdx), [], 2);
cmap = lines(numel(chIdx));
for c = 1:numel(chIdx)
    m = find(j == c & ~isCH);
    for q = m'
        plot([netR.x(q) netR.x(chIdx(c))], [netR.y(q) netR.y(chIdx(c))], '-', 'Color', 0.45*cmap(c,:) + 0.55);
    end
    scatter(netR.x(m), netR.y(m), 30, cmap(c,:), 'filled');
    plot([netR.x(chIdx(c)) sink(1)], [netR.y(chIdx(c)) sink(2)], 'k--');
end
hCH = plot(netR.x(isCH), netR.y(isCH), 's', 'MarkerSize', 11, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
hS = plot(sink(1), sink(2), 'p', 'MarkerSize', 16, 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
legend([hCH hS], {'Cluster head', 'Sink'}, 'Location', 'northwest');
title(sprintf('One LEACH round: %d cluster heads (p = %.2f)', numel(chIdx), p));
xlabel('x (m)'); ylabel('y (m)'); axis equal; xlim([-5 W + 5]); ylim([-5 sink(2) + 12]); box on;
print(f3, fullfile(figDir, 'fig3_3_routing.png'), '-dpng', '-r150');

%% 4. Network lifetime of the four protocols (N = 100) --------------------
maxR = 3000;
res = cell(1, 4);
for q = 1:4
    res{q} = simulate_lifetime(netR, protos{q}, k, maxR, p, P, R);
end
f4 = figure('Visible', 'off', 'Position', [100 100 1100 420]);
subplot(1, 2, 1); hold on;
for q = 1:4, plot(res{q}.rounds, res{q}.reach, 'LineWidth', 1.6); end
xlabel('Round'); ylabel('Working nodes (alive and connected)');
title('Working nodes per round (N = 100)');
legend(pNames, 'Location', 'northeast'); grid on; box on;
subplot(1, 2, 2); hold on;
for q = 1:4, plot(res{q}.rounds, res{q}.Etotal, 'LineWidth', 1.6); end
xlabel('Round'); ylabel('Total residual energy (J)'); title('Residual network energy');
legend(pNames, 'Location', 'northeast'); grid on; box on;
print(f4, fullfile(figDir, 'fig3_4_lifetime.png'), '-dpng', '-r150');

% energy hole: every packet passes through the few nodes one hop from the sink
r0 = res{3}.FND;
fprintf('Multi-hop relaying, N = 100: first node lost in round %d; %d nodes are one hop from the sink\n', ...
        r0, sum(par == 0));

%% 5. Lifetime table for 50, 100 and 200 nodes ---------------------------
Ns = [50 100 200];
fprintf('\nNetwork lifetime (rounds), field %dx%d m, sink at (%d,%d), E0 = %.1f J, k = %d bits, R = %d m\n', ...
        W, H, sink(1), sink(2), E0, k, R);
fprintf('A node is lost when its battery is empty or it can no longer reach the sink.\n');
fprintf('%6s  %-17s %6s %6s %6s\n', 'N', 'Protocol', 'FND', 'HND', 'LND');
T = zeros(numel(Ns) * 4, 5);
row = 0;
for n = Ns
    if n == 100
        % reuse the network and results of Section 4, so that the N = 100
        % rows of the table and fig3_4_lifetime.png come from the same run
        net = netR;
    else
        for tries = 1:1000                 % same rule as above: every node connected
            net = deploy_nodes(n, W, H, sink, 'random', E0);
            [~, ~, ~, rc] = route_tree(net.D, net.dSink, true(n, 1), R, k, P);
            if all(rc), break; end
        end
        if ~all(rc), warning('week3_demo:deploy', 'N = %d: no connected deployment found.', n); end
    end
    for q = 1:4
        if n == 100
            rq = res{q};                   % same run as the figure
        else
            rq = simulate_lifetime(net, protos{q}, k, maxR, p, P, R);
        end
        fprintf('%6d  %-17s %6s %6s %6s\n', n, pNames{q}, num2str(rq.FND), num2str(rq.HND), num2str(rq.LND));
        row = row + 1; T(row, :) = [n q rq.FND rq.HND rq.LND];
    end
end
fid = fopen(fullfile(dataDir, 'week3_lifetime_table.csv'), 'w');
fprintf(fid, 'N,protocol,FND,HND,LND\n');
for r = 1:size(T, 1)
    fprintf(fid, '%d,%s,%d,%d,%d\n', T(r,1), pNames{T(r,2)}, T(r,3), T(r,4), T(r,5));
end
fclose(fid);

%% 6. Large field: 200 m x 200 m, 400 nodes (same density) -------------------
WL = 200; HL = 200; sinkL = [WL / 2, HL]; NL = 400;
for tries = 1:1000                         % every node must reach the sink
    netL = deploy_nodes(NL, WL, HL, sinkL, 'random', E0);
    [parL, hopsL, ~, rc] = route_tree(netL.D, netL.dSink, true(NL, 1), R, k, P);
    if all(rc), break; end
end
if ~all(rc), warning('week3_demo:deploy', 'Large field: no connected deployment found.'); end
fprintf('\nLarge field %dx%d m, sink at (%d,%d), N = %d, R = %d m\n', WL, HL, sinkL(1), sinkL(2), NL, R);
fprintf('Distance to sink: median %.0f m, max %.0f m; %.0f%% of nodes beyond d0 = %.1f m\n', ...
        median(netL.dSink), max(netL.dSink), 100 * mean(netL.dSink >= P.d0), P.d0);
fprintf('Hops to the sink: mean %.1f, max %d\n', mean(hopsL), max(hopsL));
fprintf('%6s  %-17s %6s %6s %6s\n', 'N', 'Protocol', 'FND', 'HND', 'LND');
resL = cell(1, 4);  TL = zeros(4, 5);
for q = 1:4
    resL{q} = simulate_lifetime(netL, protos{q}, k, maxR, p, P, R);
    fprintf('%6d  %-17s %6s %6s %6s\n', NL, pNames{q}, num2str(resL{q}.FND), ...
            num2str(resL{q}.HND), num2str(resL{q}.LND));
    TL(q, :) = [NL q resL{q}.FND resL{q}.HND resL{q}.LND];
end
fprintf('Half-network lifetime, multi-hop + agg. vs. direct: %.1f times\n', resL{4}.HND / resL{1}.HND);
fid = fopen(fullfile(dataDir, 'week3_lifetime_large_field.csv'), 'w');
fprintf(fid, 'N,protocol,FND,HND,LND\n');
for r = 1:4
    fprintf(fid, '%d,%s,%d,%d,%d\n', TL(r,1), pNames{TL(r,2)}, TL(r,3), TL(r,4), TL(r,5));
end
fclose(fid);

f5 = figure('Visible', 'off', 'Position', [100 100 1200 520]);
subplot(1, 2, 1); hold on;
for i = 1:NL
    if parL(i) == 0, q = sinkL; else, q = [netL.x(parL(i)) netL.y(parL(i))]; end
    plot([netL.x(i) q(1)], [netL.y(i) q(2)], '-', 'Color', [0.3 0.55 0.85], 'LineWidth', 0.8);
end
scatter(netL.x, netL.y, 20, hopsL, 'filled');
plot(sinkL(1), sinkL(2), 'p', 'MarkerSize', 16, 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
th = linspace(pi, 2 * pi, 100);
plot(sinkL(1) + P.d0 * cos(th), sinkL(2) + P.d0 * sin(th), 'k--', 'LineWidth', 1);
cb = colorbar; ylabel(cb, 'Hops to the sink');
title(sprintf('200 m x 200 m, N = %d (dashed: d_0 = %.1f m)', NL, P.d0));
xlabel('x (m)'); ylabel('y (m)'); axis equal; xlim([-5 WL + 5]); ylim([-5 HL + 12]); box on;
subplot(1, 2, 2); hold on;
for q = 1:4, plot(resL{q}.rounds, resL{q}.reach, 'LineWidth', 1.6); end
xlabel('Round'); ylabel('Working nodes (alive and connected)');
title(sprintf('Working nodes per round, large field (N = %d)', NL));
legend(pNames, 'Location', 'northeast'); grid on; box on;
print(f5, fullfile(figDir, 'fig3_5_large_field.png'), '-dpng', '-r150');

fprintf('\nFigures saved to %s\n', figDir);
