%% WEEK5_DEMO  10-node network demo of SensorNode + WSNSimulator
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  Simulates 10 nodes in a 100 m x 100 m field for 72 hours (one reading
%  every 5 minutes = 864 sampling periods). Each node talks only to
%  neighbours within 30 m, and readings are relayed hop by hop to the sink
%  at (50, 100) (multi-hop routing). The positions are redrawn until every
%  node has a route to the sink. The run is then repeated on the same
%  positions with LEACH clustering and with direct transmission.
%
%  Outputs:
%    ../figures/fig5_1_network.png     topology, neighbour links and routes
%    ../figures/fig5_2_readings.png    readings received at the sink
%    ../figures/fig5_3_energy.png      energy used per node, three protocols
%    ../data/week5_demo_log.mat        full simulation log
%    printed summary table

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week3'));
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
rng(5);

nSteps = 864;                                   % 72 h at 5-minute sampling
opts = struct('N', 10, 'protocol', 'multihop', 'commRange', 30);
sim = WSNSimulator(opts);
tic; sim.run(nSteps); tRun = toc;
S = sim.summary();

% same positions, LEACH and direct transmission, for comparison
xy = cell2mat(cellfun(@(nd) [nd.x nd.y], sim.nodes, 'UniformOutput', false));
simL = WSNSimulator(struct('N', 10, 'protocol', 'leach', 'p', 0.2, 'positions', xy));
simL.run(nSteps);
SL = simL.summary();
simD = WSNSimulator(struct('N', 10, 'protocol', 'direct', 'positions', xy));
simD.run(nSteps);
SD = simD.summary();

%% Summary table -------------------------------------------------------------
fprintf('Simulated %d sampling periods (%.0f h) in %.1f s\n', S.steps, S.hours, tRun);
fprintf('%4s %6s %6s %6s %4s %6s %8s %8s %8s %8s\n', 'Node', 'x (m)', 'y (m)', 'dSink', ...
        'Hops', 'Next', 'Relayed', 'E MH mJ', 'E LEACH', 'E direct');
for i = 1:sim.N
    nd = sim.nodes{i};
    if sim.parent(i) == 0, nxt = 'sink'; else, nxt = sprintf('%d', sim.parent(i)); end
    fprintf('%4d %6.1f %6.1f %6.1f %4d %6s %8d %8.1f %8.1f %8.1f\n', i, nd.x, nd.y, ...
            sim.dSink(i), sim.hops(i), nxt, nd.rxCount, 1e3 * S.energyUsed(i), ...
            1e3 * SL.energyUsed(i), 1e3 * SD.energyUsed(i));
end
fprintf('(Relayed = packets received from children and forwarded, multi-hop)\n');
fprintf('Mean delivery ratio : multi-hop %.1f%%, LEACH %.1f%%, direct %.1f%%\n', ...
        100 * S.meanDelivery, 100 * SL.meanDelivery, 100 * SD.meanDelivery);
fprintf('Total energy used   : multi-hop %.1f mJ, LEACH %.1f mJ, direct %.1f mJ\n', ...
        1e3 * sum(S.energyUsed), 1e3 * sum(SL.energyUsed), 1e3 * sum(SD.energyUsed));
fprintf('Max energy by 1 node: multi-hop %.1f mJ, LEACH %.1f mJ, direct %.1f mJ\n', ...
        1e3 * max(S.energyUsed), 1e3 * max(SL.energyUsed), 1e3 * max(SD.energyUsed));
fprintf('Mean / max hops to the sink: %.2f / %d\n', S.meanHops, S.maxHops);
fprintf('Readings lost at a relay   : %.2f%% of node-samples\n', 100 * S.relayLostFraction);
err = sim.log.reading - sim.log.truth;
fprintf('Reading error at sink: mean %.3f, std %.3f deg C\n', mean(err(~isnan(err))), std(err(~isnan(err))));

%% Figure 5.1: network ------------------------------------------------------
f1 = figure('Visible', 'off', 'Position', [100 100 640 640]);
h = sim.plotNetwork(nSteps);
legend(h, {'Node', 'Node one hop from the sink', 'Sink'}, 'Location', 'southwest');
title(sprintf('10-node network, multi-hop routes (R = %d m), t = %.0f h', ...
      sim.opts.commRange, sim.log.time(nSteps)));
xlim([-5 110]); ylim([-5 110]);
print(f1, fullfile(figDir, 'fig5_1_network.png'), '-dpng', '-r150');

%% Figure 5.2: readings at the sink ------------------------------------------
t = sim.log.time;
f2 = figure('Visible', 'off', 'Position', [50 50 1300 900]);
for i = 1:sim.N
    subplot(5, 2, i); hold on;
    plot(t, sim.log.truth(:, i), '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
    plot(t, sim.log.reading(:, i), '-', 'Color', [0 0.35 0.7], 'LineWidth', 0.9);
    lost = isnan(sim.log.reading(:, i));
    if any(lost), plot(t(lost), 13 * ones(sum(lost), 1), 'x', 'Color', [0.85 0.2 0.1], 'MarkerSize', 3); end
    title(sprintf('Node %d  (delivery %.1f%%)', i, 100 * S.deliveryRatio(i)));
    xlim([0 72]); ylim([12 34]); set(gca, 'XTick', 0:12:72); box on;
    if i > 8, xlabel('Time (h)'); end
    if mod(i, 2) == 1, ylabel('Temp. (°C)'); end
end
print(f2, fullfile(figDir, 'fig5_2_readings.png'), '-dpng', '-r150');

%% Figure 5.3: energy ---------------------------------------------------------
f3 = figure('Visible', 'off', 'Position', [100 100 1100 420]);
subplot(1, 2, 1);
plot(t, 1e3 * (sim.opts.E0 - sim.log.energy), 'LineWidth', 1.2);
xlabel('Time (h)'); ylabel('Energy used (mJ)'); title('Energy used per node, multi-hop');
xlim([0 72]); set(gca, 'XTick', 0:12:72); grid on;
lastHop = find(sim.parent == 0)';
legend(arrayfun(@(i) sprintf('Node %d%s', i, repmat('*', 1, double(any(lastHop == i)))), 1:sim.N, ...
       'UniformOutput', false), 'Location', 'northwest', 'FontSize', 7);
subplot(1, 2, 2);
bar([1e3 * S.energyUsed(:), 1e3 * SL.energyUsed(:), 1e3 * SD.energyUsed(:)]);
xlabel('Node'); ylabel('Energy used in 72 h (mJ)'); title('Multi-hop vs. LEACH vs. direct');
legend({'Multi-hop', 'LEACH', 'Direct'}, 'Location', 'northwest'); grid on;
print(f3, fullfile(figDir, 'fig5_3_energy.png'), '-dpng', '-r150');

%% Save log ---------------------------------------------------------------------
simLog = sim.log;  positions = xy;  sinkPos = sim.sink;  simOpts = sim.opts; %#ok<NASGU>
save(fullfile(dataDir, 'week5_demo_log.mat'), 'simLog', 'positions', 'sinkPos', 'simOpts', '-v7');
fprintf('Figures saved to %s\n', figDir);
