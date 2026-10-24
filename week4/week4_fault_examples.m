%% WEEK4_FAULT_EXAMPLES  Initial fault examples for the WSN fault taxonomy
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  Generates a clean 72-hour temperature trace (one reading every 5 minutes),
%  injects every fault class of the Week 4 taxonomy with inject_fault(), and
%  adds an environmental event seen by a node and its neighbours (a true
%  anomaly that must NOT be reported as a sensor fault).
%
%  Outputs:
%    ../figures/fig4_1_fault_examples.png   one panel per fault class
%    ../figures/fig4_2_event_vs_fault.png   spatial correlation: event vs. fault
%    ../data/initial_fault_examples.csv     time, class, reading, label
%    ../data/initial_fault_examples.mat     same data as MATLAB variables
%    printed table of per-class statistics

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
figDir  = fullfile(here, '..', 'figures');
dataDir = fullfile(here, '..', 'data');
if ~exist(figDir, 'dir'),  mkdir(figDir);  end
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
rng(7);

%% Clean signal: diurnal temperature + small AR(1) noise -------------------
dt = 5 / 60;                               % 5-minute sampling [h]
t  = (0:dt:72 - dt)';                      % 3 days, 864 samples
n  = numel(t);
diurnal = 22 + 6 * sin(2 * pi * (t - 9) / 24);
ar = filter(1, [1 -0.8], 0.12 * randn(n, 1));   % correlated sensor noise
x  = diurnal + ar;

%% Inject each fault class --------------------------------------------------
types  = {'normal', 'offset', 'gain', 'drift', 'stuck', 'spike', 'noise', 'loss', 'battery'};
titles = {'Normal', 'Offset (bias +3 °C)', 'Gain (x1.25)', 'Drift (+0.15 °C/h)', ...
          'Stuck-at', 'Spike / outlier', 'High noise (\sigma = 1.5 °C)', ...
          'Packet loss (bursty)', 'Battery depletion -> node death'};
opts = struct('t0', 24, 't1', 60);         % fault active from hour 24 to 60
Y = zeros(n, numel(types)); L = zeros(n, numel(types));
for c = 1:numel(types)
    o = opts;
    if any(strcmp(types{c}, {'battery', 'drift'})), o.t1 = inf; end
    [Y(:, c), L(:, c)] = inject_fault(x, t, types{c}, o);
end

%% Figure 4.1: one panel per class -----------------------------------------
f1 = figure('Visible', 'off', 'Position', [50 50 1400 1000]);
for c = 1:numel(types)
    subplot(3, 3, c); hold on;
    act = L(:, c) > 0;
    if any(act) && ~strcmp(types{c}, 'spike')
        ta = t(act);
        patch([ta(1) ta(end) ta(end) ta(1)], [5 5 45 45], [1 0.92 0.85], 'EdgeColor', 'none');
    end
    plot(t, x, '-', 'Color', [0.65 0.65 0.65], 'LineWidth', 1.0);
    plot(t, Y(:, c), '-', 'Color', [0.0 0.35 0.7], 'LineWidth', 1.1);
    if strcmp(types{c}, 'spike')
        plot(t(act), Y(act, c), 'o', 'Color', [0.85 0.2 0.1], 'MarkerSize', 4);
    end
    if any(isnan(Y(:, c)))
        miss = isnan(Y(:, c));
        plot(t(miss), 8 * ones(sum(miss), 1), 'x', 'Color', [0.85 0.2 0.1], 'MarkerSize', 3);
    end
    title(titles{c}); xlim([0 72]); ylim([5 40]); box on;
    set(gca, 'XTick', 0:12:72);
    if c > 6, xlabel('Time (h)'); end
    if mod(c, 3) == 1, ylabel('Temperature (°C)'); end
end
print(f1, fullfile(figDir, 'fig4_1_fault_examples.png'), '-dpng', '-r150');

%% Figure 4.2: event vs. fault (spatial correlation) -----------------------
% Three neighbouring nodes observe the same field. A heat event between
% hours 30 and 34 raises all three readings (true anomaly, label 0), while an
% offset fault on node 3 between hours 48 and 60 affects only that node.
evt = zeros(n, 1);
inE = t >= 30 & t < 34;
evt(inE) = 7 * sin(pi * (t(inE) - 30) / 4);
nb = zeros(n, 3);
for k = 1:3
    nb(:, k) = diurnal + evt * (1 - 0.05 * (k - 1)) + filter(1, [1 -0.8], 0.12 * randn(n, 1));
end
[nb(:, 3), lab3] = inject_fault(nb(:, 3), t, 'offset', struct('t0', 48, 't1', 60, 'bias', 4));

f2 = figure('Visible', 'off', 'Position', [50 50 1100 420]);
evtCol = [0.87 0.93 1.00];   % light blue: event window (30-34 h)
fltCol = [1.00 0.90 0.80];   % light orange: fault window (48-60 h)

subplot(1, 2, 1); hold on;
yl = [10 36];
patch([30 34 34 30], yl([1 1 2 2]), evtCol, 'EdgeColor', 'none');
patch([48 60 60 48], yl([1 1 2 2]), fltCol, 'EdgeColor', 'none');
h = zeros(1, 3);
for k = 1:3, h(k) = plot(t, nb(:, k), 'LineWidth', 1.1); end
% labels sit just above the part of the curve they describe
[pk, ipk] = max(nb(inE, 1));  tE = t(inE);
text(tE(ipk), pk + 1.5, 'event: all nodes agree', 'FontSize', 9, 'HorizontalAlignment', 'center');
inF = t >= 48 & t < 60;
text(54, max(nb(inF, 3)) + 1.5, 'fault: node 3 only', 'FontSize', 9, 'HorizontalAlignment', 'center');
xlim([0 72]); ylim(yl); box on; set(gca, 'XTick', 0:12:72, 'Layer', 'top');
xlabel('Time (h)'); ylabel('Temperature (°C)');
legend(h, {'Node 1', 'Node 2', 'Node 3'}, 'Location', 'southwest');
title('Readings of three neighbouring nodes');

subplot(1, 2, 2); hold on;
med = median(nb, 2);
yr = [-1.5 5.5];
patch([30 34 34 30], yr([1 1 2 2]), evtCol, 'EdgeColor', 'none');
patch([48 60 60 48], yr([1 1 2 2]), fltCol, 'EdgeColor', 'none');
for k = 1:3, plot(t, nb(:, k) - med, 'LineWidth', 1.1); end
text(32, 1.9, 'event: residuals stay small', 'FontSize', 9, 'HorizontalAlignment', 'center');
text(54, 5.0, 'fault: node 3 about +4 °C', 'FontSize', 9, 'HorizontalAlignment', 'center');
xlim([0 72]); ylim(yr); box on; set(gca, 'XTick', 0:12:72, 'Layer', 'top');
xlabel('Time (h)'); ylabel('Deviation from neighbourhood median (°C)');
title('Residuals: only the faulty node deviates');
print(f2, fullfile(figDir, 'fig4_2_event_vs_fault.png'), '-dpng', '-r150');

%% Save labelled examples ---------------------------------------------------
fid = fopen(fullfile(dataDir, 'initial_fault_examples.csv'), 'w');
fprintf(fid, 'time_h,true_value,fault_type,reading,label\n');
for c = 1:numel(types)
    for i = 1:n
        if isnan(Y(i, c)), r = 'NaN'; else, r = sprintf('%.3f', Y(i, c)); end
        fprintf(fid, '%.4f,%.3f,%s,%s,%d\n', t(i), x(i), types{c}, r, L(i, c));
    end
end
fclose(fid);
faultTypes = types; readings = Y; labels = L; trueValue = x; time_h = t; %#ok<NASGU>
save(fullfile(dataDir, 'initial_fault_examples.mat'), 'faultTypes', 'readings', ...
     'labels', 'trueValue', 'time_h', '-v7');

%% Per-class statistics inside the fault window -----------------------------
fprintf('%-9s %6s %9s %9s %9s %8s\n', 'Class', 'Label', 'MeanErr', 'StdErr', 'MaxAbsErr', 'Missing');
for c = 1:numel(types)
    act = L(:, c) > 0;
    if ~any(act), act = true(n, 1); end
    e = Y(act, c) - x(act);
    ok = ~isnan(e);
    fprintf('%-9s %6d %9.2f %9.2f %9.2f %7.1f%%\n', types{c}, c - 1, mean(e(ok)), std(e(ok)), ...
            max(abs(e(ok))), 100 * mean(~ok));
end
fprintf('\nFigures saved to %s\nData saved to %s\n', figDir, dataDir);
