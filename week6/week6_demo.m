%% WEEK6_DEMO  Fault injection on a 50-node network
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  50 nodes, multi-hop routing to the sink at (50, 100) (each node talks
%  only to neighbours within 30 m), 72 h at 5-minute sampling. FaultInjector puts
%  one random fault on 48% of the nodes (24 nodes, 3 per class,
%  random onset, duration and severity), adds a second, simultaneous fault
%  to two of those nodes, and adds one environmental heat event.
%  Because readings are relayed, a fault on a relay's outgoing link (loss,
%  battery) also removes the readings of the nodes behind it; the dataset
%  keeps these as missing readings with their own (normal) labels and marks
%  them in dataset.relayLost.
%
%  Outputs:
%    ../figures/fig6_1_fault_map.png      where the faults and the event are
%    ../figures/fig6_2_fault_traces.png   one simulated example per class
%    ../figures/fig6_3_label_map.png      label of every node over time
%    ../figures/fig6_4_event.png          event vs. fault, neighbour residuals
%    ../data/week6_sim_dataset.mat        readings, labels and ground truth
%    printed fault list and class distribution

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week3'));
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
rng(6);

nSteps = 864;
sim = WSNSimulator(struct('N', 50, 'protocol', 'multihop', 'commRange', 30));
inj = FaultInjector();
list = inj.addRandomFaults(sim, 0.48, struct('tStart', 6, 'tEnd', 72));   % 24 nodes = 3 per class
% two simultaneous faults on nodes that already have one
inj.addFault(list(1).node, 'noise', list(1).t0 + 2, list(1).t0 + 14, struct('std', 1.0));
inj.addFault(list(2).node, 'loss',  list(2).t0 + 1, list(2).t0 + 13, struct('prob', 0.3));
% environmental event: heat source near the centre of the field
inj.addEvent(50, 50, 40, 40, 46, 8);
sim.attachInjector(inj);

tic; sim.run(nSteps); tRun = toc;
fprintf('Simulated 50 nodes x %d periods in %.1f s\n\n', nSteps, tRun);
inj.faultTable();

L = sim.log;  t = L.time;
names = inj.classNames;
cols = [0.75 0.75 0.75; 0.12 0.47 0.71; 1.00 0.50 0.05; 0.17 0.63 0.17; 0.84 0.15 0.16; ...
        0.58 0.40 0.74; 0.55 0.34 0.29; 0.89 0.47 0.76; 0.09 0.75 0.81];

%% Class distribution --------------------------------------------------------
fprintf('\nClass distribution of all %d node-samples (primary label):\n', numel(L.label));
cnt = zeros(1, 9);
for c = 0:8
    cnt(c + 1) = sum(L.label(:) == c);
    fprintf('  %d %-8s %7d  (%5.2f%%)\n', c, names{c + 1}, cnt(c + 1), 100 * cnt(c + 1) / numel(L.label));
end
fprintf('Samples with a secondary label : %d\n', sum(L.label2(:) > 0));
fprintf('Node-samples inside the event  : %d (all labelled normal)\n', sum(L.event(:)));
fprintf('Readings missing at the sink   : %.1f%%\n', 100 * mean(isnan(L.reading(:))));
fprintf('  of which lost at a relay     : %.1f%% of node-samples\n', 100 * mean(L.relayLost(:)));
S = sim.summary();
fprintf('Alive at the end               : %d of %d nodes (%d cut off from the sink)\n', ...
        S.aliveAtEnd, sim.N, S.cutOffAtEnd);
fprintf('Hops to the sink (end)         : mean %.2f, max %d\n', S.meanHops, S.maxHops);

%% Figure 6.1: fault map -----------------------------------------------------
xy = cell2mat(cellfun(@(nd) [nd.x nd.y], sim.nodes, 'UniformOutput', false));
nodeClass = zeros(sim.N, 1);
for k = 1:numel(inj.faults)
    if nodeClass(inj.faults(k).node) == 0, nodeClass(inj.faults(k).node) = inj.faults(k).label; end
end
multi = accumarray([inj.faults.node]', 1, [sim.N 1]) > 1;
f1 = figure('Visible', 'off', 'Position', [100 100 760 700]); hold on;
th = linspace(0, 2 * pi, 100);
hEv = fill(50 + 40 * cos(th), 50 + 40 * sin(th), [1 0.93 0.85], 'EdgeColor', [0.9 0.6 0.3]);
par0 = L.parent(1, :)';                           % routing tree at the start
for i = find(par0 >= 0)'
    if par0(i) == 0, q = sim.sink; else, q = xy(par0(i), :); end
    plot([xy(i, 1) q(1)], [xy(i, 2) q(2)], '-', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.8);
end
hh = zeros(1, 9);
for c = 0:8
    m = nodeClass == c;
    hh(c + 1) = plot([xy(m, 1); NaN], [xy(m, 2); NaN], 'o', 'MarkerSize', 9, ...
                     'MarkerFaceColor', cols(c + 1, :), 'MarkerEdgeColor', 'k');
end
plot(xy(multi, 1), xy(multi, 2), 'ko', 'MarkerSize', 15, 'LineWidth', 1.5);
plot(sim.sink(1), sim.sink(2), 'p', 'MarkerSize', 16, 'MarkerFaceColor', [0.85 0.2 0.2], 'MarkerEdgeColor', 'k');
text(sim.sink(1) + 4, sim.sink(2), 'Sink');
plot([0 100 100 0 0], [0 0 100 100 0], 'k-');
legend([hh hEv], [names, {'heat event (40-46 h)'}], 'Location', 'eastoutside');
title('Injected faults and multi-hop routes (ring = two simultaneous faults)');
axis equal; xlim([-5 105]); ylim([-5 110]); box on; xlabel('x (m)'); ylabel('y (m)');
print(f1, fullfile(figDir, 'fig6_1_fault_map.png'), '-dpng', '-r150');

%% Figure 6.2: one example trace per fault class ----------------------------
f2 = figure('Visible', 'off', 'Position', [50 50 1400 1000]);
for c = 1:8
    subplot(4, 2, c); hold on;
    k = find([inj.faults.label] == c, 1);
    i = inj.faults(k).node;
    act = L.label(:, i) > 0 | L.label2(:, i) > 0;
    if any(act)
        ta = t(act);
        patch([ta(1) ta(end) ta(end) ta(1)], [0 0 50 50], [1 0.92 0.85], 'EdgeColor', 'none');
    end
    plot(t, L.truth(:, i), '-', 'Color', [0.6 0.6 0.6]);
    plot(t, L.reading(:, i), '-', 'Color', cols(c + 1, :) * 0.85);
    miss = isnan(L.reading(:, i));
    if any(miss), plot(t(miss), 7 * ones(sum(miss), 1), 'x', 'Color', [0.85 0.2 0.1], 'MarkerSize', 3); end
    title(sprintf('%s - node %d (%s)', names{c + 1}, i, FaultInjector.describe(inj.faults(k).type, inj.faults(k).params)));
    xlim([0 72]); ylim([5 45]); set(gca, 'XTick', 0:12:72); box on;
    if c > 6, xlabel('Time (h)'); end
    if mod(c, 2) == 1, ylabel('Temp. (°C)'); end
end
print(f2, fullfile(figDir, 'fig6_2_fault_traces.png'), '-dpng', '-r150');

%% Figure 6.3: label map -------------------------------------------------------
[~, order] = sortrows([nodeClass, (1:sim.N)']);
f3 = figure('Visible', 'off', 'Position', [100 100 1100 650]);
img = L.label(:, order)';
img(L.event(:, order)' & img == 0) = 9;          % show event samples
image(t, 1:sim.N, img + 1);
colormap([cols; 1 0.95 0.55]);   % light yellow = event (distinct from orange = gain)
set(gca, 'YTick', 1:sim.N, 'YTickLabel', arrayfun(@num2str, order, 'UniformOutput', false), 'FontSize', 6);
xlabel('Time (h)', 'FontSize', 10); ylabel('Node', 'FontSize', 10);
title('Primary label of every node-sample (light yellow = inside the heat event, labelled normal)', 'FontSize', 10);
cb = colorbar; set(cb, 'YTick', (1:10) + 0.5, 'YTickLabel', [names, {'event'}], 'FontSize', 8);
caxis([1 11]);
print(f3, fullfile(figDir, 'fig6_3_label_map.png'), '-dpng', '-r150');

%% Figure 6.4: event vs fault residuals ----------------------------------------
res = nan(size(L.reading));
for i = 1:sim.N
    nb = find(sim.neighbors(i, :));
    if isempty(nb), continue; end
    med = rowmedian(L.reading(:, nb));
    res(:, i) = L.reading(:, i) - med;
end
inside = find(any(L.event, 1) & nodeClass' == 0);       % normal nodes hit by the event
offN = find(nodeClass == 1)';                              % offset-fault nodes
f4 = figure('Visible', 'off', 'Position', [100 100 1100 420]);
subplot(1, 2, 1); hold on;
patch([40 46 46 40], [-6 -6 6 6], [1 0.93 0.85], 'EdgeColor', 'none');
for i = inside, plot(t, res(:, i), 'Color', [0.95 0.55 0.2]); end
xlim([0 72]); ylim([-6 6]); set(gca, 'XTick', 0:12:72); box on;
xlabel('Time (h)'); ylabel('Reading - neighbour median (°C)');
title(sprintf('%d normal nodes inside the heat event (shaded)', numel(inside)), 'FontSize', 9);
subplot(1, 2, 2); hold on;
cmapO = [0.12 0.47 0.71; 0.35 0.65 0.9; 0.05 0.25 0.5];
for j = 1:numel(offN)
    i = offN(j);  a = find(L.label(:, i) == 1);
    patch([t(a(1)) t(a(end)) t(a(end)) t(a(1))], [-6 -6 6 6], [0.88 0.93 1], 'EdgeColor', 'none');
end
for j = 1:numel(offN)
    plot(t, res(:, offN(j)), 'Color', cmapO(mod(j - 1, 3) + 1, :));
end
xlim([0 72]); ylim([-6 6]); set(gca, 'XTick', 0:12:72); box on;
xlabel('Time (h)');
title(sprintf('Offset-fault nodes %s (fault windows shaded)', mat2str(offN)), 'FontSize', 9);
print(f4, fullfile(figDir, 'fig6_4_event.png'), '-dpng', '-r150');
ev = L.event(:, inside);  rr = res(:, inside);  ok = ev & ~isnan(rr);
fprintf('Event nodes during the event: median |residual| %.2f, 95th pct %.2f, max %.2f degC\n', ...
        median(abs(rr(ok))), prctile(abs(rr(ok)), 95), max(abs(rr(ok))));
for i = offN
    a = L.label(:, i) == 1 & ~isnan(res(:, i));
    fprintf('Offset node %d (bias %+.2f): median |residual| during fault %.2f degC\n', i, ...
            inj.faults(find([inj.faults.node] == i & [inj.faults.label] == 1, 1)).params.bias, median(abs(res(a, i))));
end

%% Save dataset -----------------------------------------------------------------
dataset = struct('time', t, 'reading', L.reading, 'truth', L.truth, 'label', L.label, ...
    'label2', L.label2, 'event', L.event, 'voltage', L.voltage, 'energy', L.energy, ...
    'isCH', L.isCH, 'positions', xy, 'neighbors', sim.neighbors, 'sink', sim.sink, ...
    'parent', L.parent, 'relayLost', L.relayLost, 'protocol', 'multihop'); %#ok<NASGU>
dataset.classNames = names;
faults = inj.faults; eventList = inj.eventList; %#ok<NASGU>
save(fullfile(dataDir, 'week6_sim_dataset.mat'), 'dataset', 'faults', 'eventList', '-v7');
fprintf('Dataset saved to %s\n', fullfile(dataDir, 'week6_sim_dataset.mat'));
