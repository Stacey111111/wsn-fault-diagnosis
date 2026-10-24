%% WEEK7_DEMO  Fault dataset and two baseline detectors
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  1. Builds the window-level dataset from 10 independent simulated networks
%     (seeds 701-710; 50 nodes each, multi-hop routing, 72 h at 5-minute
%     sampling, random faults on half of the nodes, simultaneous faults and
%     one environmental event per network). Windows are 1 h (12 samples).
%  2. Runs two baselines on the labelled windows:
%       - fixed-threshold rules (baseline_threshold), which detect AND
%         classify a fault;
%       - a Kalman filter residual detector (baseline_kalman), which only
%         detects (a window is faulty when its largest |innovation| > 4).
%  3. Reports per-class precision, recall and F1, detection rates and false
%     alarms, and a threshold sweep for the Kalman detector.
%
%  Outputs:
%    ../data/fault_dataset.mat               runs + window table (reused if present)
%    ../data/week7_baselines.mat             baseline outputs per labelled window
%    ../data/week7_results.json              all numbers used in the report
%    ../figures/fig7_1_dataset.png           class distribution of the dataset
%    ../figures/fig7_2_threshold_confusion.png
%    ../figures/fig7_3_detection_by_class.png
%    ../figures/fig7_4_kalman.png            Kalman innovations and k sweep

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week3'), here);
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
dsFile = fullfile(dataDir, 'fault_dataset.mat');

%% 1. Dataset
if exist(dsFile, 'file')
    S = load(dsFile);  D = S.D;
    fprintf('Loaded %s (%d runs)\n', dsFile, numel(D.runs));
else
    fprintf('Simulating 10 networks ...\n');
    tSim = tic;
    D = build_dataset(701:710);
    fprintf('Simulation time %.1f s\n', toc(tSim));
    save(dsFile, 'D', '-v7');
end
W = D.win;  names = D.classNames;  K = numel(names);
inSet = W.inSet;
nRuns = numel(D.runs);

fprintf('\nWindows: %d in total (%d runs x %d nodes x %d windows)\n', numel(W.label), ...
        nRuns, size(D.runs(1).reading, 2), floor(size(D.runs(1).reading, 1) / D.winLen));
fprintf('Transition windows (label -1, excluded): %d\n', sum(W.label == -1));
fprintf('Labelled dataset: %d windows\n\n', sum(inSet));
fprintf('%-9s %9s %9s %10s %9s\n', 'Class', 'All', 'Dataset', 'In event', '2nd label');
cls = struct('name', {}, 'all', {}, 'dataset', {}, 'inEvent', {}, 'second', {});
for c = 0:K - 1
    m = W.label == c;
    cls(c + 1) = struct('name', names{c + 1}, 'all', sum(m), 'dataset', sum(m & inSet), ...
                        'inEvent', sum(m & inSet & W.eventFrac > 0), ...
                        'second', sum(m & inSet & W.label2 > 0));
    fprintf('%-9s %9d %9d %10d %9d\n', names{c + 1}, cls(c + 1).all, cls(c + 1).dataset, ...
            cls(c + 1).inEvent, cls(c + 1).second);
end
nFaults = arrayfun(@(r) numel(r.faults), D.runs);
fprintf('Faults injected: %d (%.1f per network)\n', sum(nFaults), mean(nFaults));

%% 2. Baselines on the labelled windows
idx = find(inSet);
y = W.label(idx);
yThr = zeros(size(idx));  zK = zeros(size(idx));
rule = zeros(size(idx));  ownMiss = zeros(size(idx));
tThr = 0;  tKal = 0;
for k = 1:nRuns
    m = W.run(idx) == k;
    t0 = tic;
    [yThr(m), info] = baseline_threshold(D.runs(k), W.s0(idx(m)), W.node(idx(m)), D.winLen);
    tThr = tThr + toc(t0);
    rule(m) = info.rule;  ownMiss(m) = info.ownMiss;
    t0 = tic;
    zK(m) = baseline_kalman(D.runs(k), W.s0(idx(m)), W.node(idx(m)), D.winLen);
    tKal = tKal + toc(t0);
end
kK = 4;                                  % Kalman decision threshold
yKal = double(zK > kK);                  % 1 = faulty

%% 3. Metrics
Mthr = class_metrics(y, yThr, 0:K - 1);
fprintf('\nBaseline 1 - fixed thresholds (multi-class)\n');
fprintf('%-9s %9s %7s %7s %7s\n', 'Class', 'Precision', 'Recall', 'F1', 'Support');
for c = 1:K
    fprintf('%-9s %9.3f %7.3f %7.3f %7d\n', names{c}, Mthr.precision(c), Mthr.recall(c), ...
            Mthr.f1(c), Mthr.support(c));
end
fprintf('Macro-F1 %.3f   accuracy %.3f\n', Mthr.macroF1, Mthr.accuracy);
fprintf('\nConfusion matrix (rows = true class, columns = predicted class)\n');
fprintf('%-9s', '');  fprintf('%6d', 0:K - 1);  fprintf('\n');
for c = 1:K
    fprintf('%-9s', names{c});  fprintf('%6d', Mthr.confusion(c, :));  fprintf('\n');
end

isF = y > 0;
isEv = y == 0 & W.eventFrac(idx) > 0;
isQuiet = y == 0 & W.eventFrac(idx) == 0;
bin = @(pred) struct('precision', sum(pred & isF) / max(sum(pred), 1), ...
                     'recall', sum(pred & isF) / max(sum(isF), 1), ...
                     'f1', 2 * sum(pred & isF) / max(sum(pred) + sum(isF), 1), ...
                     'farAll', sum(pred & ~isF) / max(sum(~isF), 1), ...
                     'farQuiet', sum(pred & isQuiet) / max(sum(isQuiet), 1), ...
                     'farEvent', sum(pred & isEv) / max(sum(isEv), 1));
Bthr = bin(yThr > 0);
Bkal = bin(yKal > 0);
fprintf('\nFault / no-fault detection\n');
fprintf('%-18s %9s %7s %7s %9s %9s %9s\n', 'Method', 'Precision', 'Recall', 'F1', ...
        'FA all', 'FA quiet', 'FA event');
fprintf('%-18s %9.3f %7.3f %7.3f %9.3f %9.3f %9.3f\n', 'Thresholds', Bthr.precision, ...
        Bthr.recall, Bthr.f1, Bthr.farAll, Bthr.farQuiet, Bthr.farEvent);
fprintf('%-18s %9.3f %7.3f %7.3f %9.3f %9.3f %9.3f\n', sprintf('Kalman |z|>%g', kK), ...
        Bkal.precision, Bkal.recall, Bkal.f1, Bkal.farAll, Bkal.farQuiet, Bkal.farEvent);

detThr = zeros(1, K - 1);  detKal = zeros(1, K - 1);
fprintf('\nDetection rate per fault class (window flagged as faulty)\n');
fprintf('%-9s %11s %8s\n', 'Class', 'Thresholds', 'Kalman');
for c = 1:K - 1
    m = y == c;
    detThr(c) = mean(yThr(m) > 0);  detKal(c) = mean(yKal(m) > 0);
    fprintf('%-9s %11.3f %8.3f\n', names{c + 1}, detThr(c), detKal(c));
end

% Kalman threshold sweep
kGrid = 2:0.5:6;
sweep = struct('k', num2cell(kGrid), 'precision', 0, 'recall', 0, 'f1', 0, ...
               'farAll', 0, 'farEvent', 0);
for j = 1:numel(kGrid)
    B = bin(zK > kGrid(j));
    sweep(j).precision = B.precision;  sweep(j).recall = B.recall;  sweep(j).f1 = B.f1;
    sweep(j).farAll = B.farAll;  sweep(j).farEvent = B.farEvent;
end
fprintf('\nKalman sweep:  k   precision  recall   F1   FA all\n');
for j = 1:numel(kGrid)
    fprintf('             %4.1f  %7.3f  %7.3f  %5.3f  %6.3f\n', kGrid(j), sweep(j).precision, ...
            sweep(j).recall, sweep(j).f1, sweep(j).farAll);
end

% spread over networks (one value per run)
runF1 = zeros(1, nRuns);
for k = 1:nRuns
    m = W.run(idx) == k;
    Mk = class_metrics(y(m), yThr(m), 0:K - 1);
    runF1(k) = Mk.macroF1;
end
fprintf('\nThreshold macro-F1 per network: mean %.3f, std %.3f, min %.3f, max %.3f\n', ...
        mean(runF1), std(runF1), min(runF1), max(runF1));
fprintf('Time: thresholds %.2f ms/window, Kalman %.2f ms/window\n', ...
        1e3 * tThr / numel(idx), 1e3 * tKal / numel(idx));

lossNoMiss = mean(ownMiss(y == 7) == 0);
fprintf('Loss windows without any own missing reading: %.1f%%\n', 100 * lossNoMiss);
save(fullfile(dataDir, 'week7_baselines.mat'), 'idx', 'y', 'yThr', 'zK', 'rule', '-v7');

%% 4. Figures
% Fig 7.1  class distribution
f1 = figure('Name', 'Fig 7.1 Dataset', 'Position', [100 100 1100 420]);
subplot(1, 2, 1);
cntSet = [cls.dataset];  cntEv = [cls.inEvent];
b = barh(0:K - 1, [cntSet - cntEv; cntEv]', 'stacked');
set(b(1), 'FaceColor', [0.30 0.50 0.75]);  set(b(2), 'FaceColor', [0.95 0.60 0.20]);
set(gca, 'YTick', 0:K - 1, 'YTickLabel', names, 'YDir', 'reverse');
xlabel('Labelled windows');  legend({'outside events', 'inside an event'}, 'Location', 'southeast');
title(sprintf('Labelled dataset: %d one-hour windows', sum(inSet)));  grid on;
subplot(1, 2, 2);
perRun = zeros(nRuns, K);
for k = 1:nRuns
    for c = 0:K - 1, perRun(k, c + 1) = sum(inSet & W.run == k & W.label == c); end
end
pal = [0.55 0.55 0.55; 0.00 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.47 0.67 0.19; ...
       0.30 0.75 0.93; 0.64 0.08 0.18; 0.85 0.33 0.10; 0.10 0.10 0.10];   % one colour per class
bs = bar(1:nRuns, perRun, 'stacked');
for c = 1:K, set(bs(c), 'FaceColor', pal(c, :)); end
set(gca, 'XTick', 1:nRuns, 'XTickLabel', arrayfun(@num2str, D.seeds, 'UniformOutput', false));
xlabel('Network (random seed)');  ylabel('Labelled windows');
title('Windows per network and class');  legend(names, 'Location', 'eastoutside');  grid on;
print(f1, fullfile(figDir, 'fig7_1_dataset.png'), '-dpng', '-r150');

% Fig 7.2  confusion matrix of the threshold rules (row-normalised)
f2 = figure('Name', 'Fig 7.2 Threshold confusion', 'Position', [100 100 640 560]);
Cn = Mthr.confusion ./ max(sum(Mthr.confusion, 2), 1);
imagesc(Cn, [0 1]);  colorbar;
colormap(gca, [linspace(1, 0.13, 256)', linspace(1, 0.30, 256)', linspace(1, 0.60, 256)']);
for i = 1:K
    for j = 1:K
        if Mthr.confusion(i, j) > 0
            if Cn(i, j) > 0.5, tc = 'w'; else, tc = 'k'; end
            text(j, i, sprintf('%d', Mthr.confusion(i, j)), 'HorizontalAlignment', 'center', ...
                 'Color', tc, 'FontSize', 8);
        end
    end
end
set(gca, 'XTick', 1:K, 'XTickLabel', names, 'YTick', 1:K, 'YTickLabel', names);  xtickangle(30);
xlabel('Predicted class');  ylabel('True class');
title(sprintf('Fixed thresholds: macro-F1 = %.3f', Mthr.macroF1));
print(f2, fullfile(figDir, 'fig7_2_threshold_confusion.png'), '-dpng', '-r150');

% Fig 7.3  detection rate per class and false alarms
f3 = figure('Name', 'Fig 7.3 Detection by class', 'Position', [100 100 900 420]);
vals = [detThr, Bthr.farQuiet, Bthr.farEvent; detKal, Bkal.farQuiet, Bkal.farEvent]';
bh = bar(vals);
set(bh(1), 'FaceColor', [0.30 0.50 0.75]);  set(bh(2), 'FaceColor', [0.85 0.33 0.10]);
labs = [names(2:end), {'FA quiet', 'FA event'}];
set(gca, 'XTick', 1:numel(labs), 'XTickLabel', labs, 'FontSize', 9);
xlim([0.4 numel(labs) + 0.6]);  ylim([0 1.25]);  ylabel('Fraction of windows flagged');  grid on;
hold on;  plot([K - 0.5 K - 0.5], [0 1.25], 'k--');
legend({'Fixed thresholds', sprintf('Kalman |z| > %g', kK)}, 'Location', 'north', ...
       'Orientation', 'horizontal');
title('Detection rate per fault class (left) and false-alarm rate (right)');
print(f3, fullfile(figDir, 'fig7_3_detection_by_class.png'), '-dpng', '-r150');

% Fig 7.4  Kalman innovations on one node, and the k sweep
f4 = figure('Name', 'Fig 7.4 Kalman', 'Position', [100 100 1100 420]);
ex = struct('run', 0, 'node', 0);  best = 0;
for k = 1:nRuns                          % the drift fault with the largest rate
    F = D.runs(k).faults;
    for j = 1:numel(F)
        if strcmp(F(j).type, 'drift') && abs(F(j).params.rate) > best
            best = abs(F(j).params.rate);  ex = struct('run', k, 'node', F(j).node);
        end
    end
end
Rr = D.runs(ex.run);  xi = Rr.reading(:, ex.node);
zi = kalman_innovations(xi);
fl = Rr.label(:, ex.node) > 0;
tf = Rr.time(fl);
exInfo = struct('seed', D.seeds(ex.run), 'node', ex.node, 'rate', best, ...
                'maxDev', max(abs(xi(fl) - Rr.truth(fl, ex.node))), 'maxZ', max(abs(zi(fl))));
subplot(2, 2, 1);
plot(Rr.time, xi, 'b-', Rr.time, Rr.truth(:, ex.node), 'k:');
ylabel('Temperature (deg C)');  set(gca, 'XTick', 0:12:72);  grid on;
legend({'reading', 'true value'}, 'Location', 'northwest');
if ~isempty(tf)
    title(sprintf('Seed %d, node %d: drift fault from %.0f h', D.seeds(ex.run), ex.node, tf(1)));
end
subplot(2, 2, 3);
plot(Rr.time, abs(zi), 'r-', Rr.time([1 end]), [kK kK], 'k--');
xlabel('Time (h)');  ylabel('|z| (innovation / \sigma)');  set(gca, 'XTick', 0:12:72);  grid on;
legend({'|z|', sprintf('k = %g', kK)}, 'Location', 'northwest');
subplot(1, 2, 2);
plot(kGrid, [sweep.precision], 'o-', kGrid, [sweep.recall], 's-', kGrid, [sweep.f1], 'd-', ...
     kGrid, [sweep.farAll], 'x--', 'LineWidth', 1.2);
xlabel('Decision threshold k (|z| > k)');  ylim([0 1]);  grid on;
legend({'precision', 'recall', 'F1', 'false-alarm rate'}, 'Location', 'east');
title('Kalman detector: threshold sweep');
print(f4, fullfile(figDir, 'fig7_4_kalman.png'), '-dpng', '-r150');

%% 5. Results file (numbers used in the report)
out = struct();
out.seeds = D.seeds;  out.winLen = D.winLen;
out.nWindows = numel(W.label);  out.nTransition = sum(W.label == -1);
out.nLabelled = sum(inSet);  out.nFaults = sum(nFaults);
out.classes = cls;
out.perRun = sum(perRun, 2)';
out.normalShareAll = mean(W.label == 0);
out.lossNoMiss = lossNoMiss;
out.kalmanExample = exInfo;
out.threshold = struct('macroF1', Mthr.macroF1, 'accuracy', Mthr.accuracy, ...
    'precision', Mthr.precision, 'recall', Mthr.recall, 'f1', Mthr.f1, ...
    'support', Mthr.support, 'confusion', Mthr.confusion, 'binary', Bthr, ...
    'detection', detThr, 'runMacroF1', runF1, 'msPerWindow', 1e3 * tThr / numel(idx));
out.kalman = struct('k', kK, 'binary', Bkal, 'detection', detKal, 'sweep', sweep, ...
    'msPerWindow', 1e3 * tKal / numel(idx));
fid = fopen(fullfile(dataDir, 'week7_results.json'), 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('\nSaved data/week7_results.json and figures fig7_1 ... fig7_4\n');
