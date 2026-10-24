%% WEEK8_DEMO  Feature extraction and feature selection
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  Turns every labelled one-hour window of the Week 7 dataset into a vector
%  of 29 features (time domain, frequency domain, Haar wavelet, spatial
%  neighbour residual, Kalman innovation, node state), ranks the features
%  by their ANOVA F statistic and removes features that are strongly
%  correlated (|r| > 0.9) with a better-ranked one.
%
%  Needs ../data/fault_dataset.mat (run week7_demo first).
%
%  Outputs:
%    ../data/features.mat                 X, y, names, groups, run, node, ...
%    ../data/features.csv                 the same table as text
%    ../data/week8_results.json           all numbers used in the report
%    ../figures/fig8_1_feature_profiles.png
%    ../figures/fig8_2_feature_ranking.png
%    ../figures/fig8_3_feature_correlation.png
%    ../figures/fig8_4_pca.png

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), fullfile(here, '..', 'week7'), here);
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end

S = load(fullfile(dataDir, 'fault_dataset.mat'));  D = S.D;
W = D.win;  cn = D.classNames;  K = numel(cn);
idx = find(W.inSet);
y = W.label(idx);
fprintf('Labelled windows: %d from %d networks\n', numel(idx), numel(D.runs));

%% 1. Feature extraction
X = [];  tEx = 0;
for k = 1:numel(D.runs)
    m = W.run(idx) == k;
    t0 = tic;
    [Xk, names, groups] = extract_features(D.runs(k), W.s0(idx(m)), W.node(idx(m)), D.winLen);
    tEx = tEx + toc(t0);
    if isempty(X), X = zeros(numel(idx), size(Xk, 2)); end
    X(m, :) = Xk;
end
P = numel(names);
msPerWin = 1e3 * tEx / numel(idx);
fprintf('Extracted %d features per window in %.1f s (%.2f ms per window)\n', P, tEx, msPerWin);

%% 2. Ranking and correlation filter
rMax = 0.9;
Sel = select_features(X, y, rMax, names);
kept = false(1, P);  kept(Sel.keep) = true;
fprintf('\n%4s %-13s %-8s %10s  %s\n', 'Rank', 'Feature', 'Group', 'F', 'Kept');
for r = 1:P
    j = Sel.order(r);
    if kept(j), kk = 'yes'; else, kk = '-'; end
    fprintf('%4d %-13s %-8s %10.1f  %s\n', r, names{j}, groups{j}, Sel.F(j), kk);
end
fprintf('\nDropped by the correlation filter (|r| > %.1f):\n', rMax);
for d = 1:numel(Sel.dropped)
    fprintf('  %-13s r = %+.3f with %s\n', Sel.dropped(d).feature, Sel.dropped(d).r, ...
            Sel.dropped(d).correlatedWith);
end
fprintf('Kept %d of %d features\n', numel(Sel.keep), P);

gNames = {'time', 'freq', 'wavelet', 'spatial', 'kalman', 'state'};
grp = struct('group', gNames, 'nFeatures', 0, 'nKept', 0, 'bestFeature', '', 'bestF', 0);
fprintf('\n%-8s %5s %5s  %-13s %9s\n', 'Group', 'Feat.', 'Kept', 'Best feature', 'F');
for g = 1:numel(gNames)
    m = strcmp(groups, gNames{g});
    [bf, b] = max(Sel.F .* m);
    grp(g).nFeatures = sum(m);  grp(g).nKept = sum(m & kept);
    grp(g).bestFeature = names{b};  grp(g).bestF = bf;
    fprintf('%-8s %5d %5d  %-13s %9.1f\n', gNames{g}, sum(m), sum(m & kept), names{b}, bf);
end

% median of a few features per class (shown in the report)
show = {'maxJump', 'diffStd', 'flatFrac', 'resAbs', 'resGrowth', 'levelAbs', 'ownMiss', 'vMin'};
med = zeros(K, numel(show));
fprintf('\nClass medians\n%-9s', '');  fprintf('%10s', show{:});  fprintf('\n');
for c = 1:K
    for j = 1:numel(show)
        med(c, j) = median(X(y == c - 1, strcmp(names, show{j})));
    end
    fprintf('%-9s', cn{c});  fprintf('%10.3f', med(c, :));  fprintf('\n');
end

%% 3. Save the feature table
run = W.run(idx);  node = W.node(idx);  s0 = W.s0(idx);
label2 = W.label2(idx);  eventFrac = W.eventFrac(idx);
keep = Sel.keep;  F = Sel.F;  classNames = cn;
save(fullfile(dataDir, 'features.mat'), 'X', 'y', 'names', 'groups', 'run', 'node', 's0', ...
     'label2', 'eventFrac', 'keep', 'F', 'classNames', '-v7');
fid = fopen(fullfile(dataDir, 'features.csv'), 'w');
fprintf(fid, 'run,node,s0,label,label2,eventFrac');  fprintf(fid, ',%s', names{:});  fprintf(fid, '\n');
fmt = ['%d,%d,%d,%d,%d,%.4f', repmat(',%.6g', 1, P), '\n'];
fprintf(fid, fmt, [run, node, s0, y, label2, eventFrac, X]');
fclose(fid);

%% 4. Figures
gCol = [0.00 0.45 0.74; 0.93 0.69 0.13; 0.47 0.67 0.19; 0.85 0.33 0.10; 0.49 0.18 0.56; 0.30 0.30 0.30];
gIdx = cellfun(@(g) find(strcmp(gNames, g)), groups);

% Fig 8.1  class profiles (median and interquartile range) of 8 features
f1 = figure('Name', 'Fig 8.1 Feature profiles', 'Position', [100 100 1200 560]);
for j = 1:numel(show)
    subplot(2, 4, j);
    col = strcmp(names, show{j});
    q = zeros(K, 3);
    for c = 1:K, q(c, :) = quantiles(X(y == c - 1, col), [0.25 0.5 0.75]); end
    he = errorbar(0:K - 1, q(:, 2), q(:, 2) - q(:, 1), q(:, 3) - q(:, 2), 'o');
    set(he, 'LineWidth', 1.2, 'Color', gCol(gIdx(col), :), 'MarkerFaceColor', gCol(gIdx(col), :));
    set(gca, 'XTick', 0:K - 1, 'XTickLabel', cellfun(@(s) s(1:3), cn, 'UniformOutput', false), ...
        'FontSize', 8);
    xlim([-0.5 K - 0.5]);  grid on;
    title(sprintf('%s (%s)', show{j}, groups{col}));
end
print(f1, fullfile(figDir, 'fig8_1_feature_profiles.png'), '-dpng', '-r150');

% Fig 8.2  F ranking, coloured by group; dropped features in light colour
f2 = figure('Name', 'Fig 8.2 Feature ranking', 'Position', [100 100 760 720]);
hold on;
for r = 1:P                              % one horizontal bar per feature, best at the top
    j = Sel.order(r);
    cc = gCol(gIdx(j), :);
    if ~kept(j), cc = 0.35 * cc + 0.65; end
    patch(log10([1 Sel.F(j) Sel.F(j) 1]), [r - 0.4 r - 0.4 r + 0.4 r + 0.4], cc, 'EdgeColor', 'none');
end
hg = zeros(1, numel(gNames));
for g = 1:numel(gNames), hg(g) = plot(nan, nan, 's', 'MarkerFaceColor', gCol(g, :), ...
                                      'MarkerEdgeColor', gCol(g, :), 'MarkerSize', 9); end
set(gca, 'YDir', 'reverse', 'YTick', 1:P, 'YTickLabel', names(Sel.order), 'FontSize', 8, ...
    'XTick', 0:5, 'XTickLabel', {'1', '10', '10^2', '10^3', '10^4', '10^5'});
ylim([0.4 P + 0.6]);  xlim([0 5]);  xlabel('ANOVA F statistic (log scale)');  grid on;  box on;
legend(hg, gNames, 'Location', 'southeast');
title(sprintf('%d of %d features kept (pale = dropped, |r| > %.1f)', numel(Sel.keep), P, rMax));
print(f2, fullfile(figDir, 'fig8_2_feature_ranking.png'), '-dpng', '-r150');

% Fig 8.3  correlation matrix (features in group order)
f3 = figure('Name', 'Fig 8.3 Correlation', 'Position', [100 100 760 680]);
imagesc(Sel.corr, [-1 1]);  axis square;  colorbar;
n2 = 128;  cmap = [linspace(0.13, 1, n2)', linspace(0.40, 1, n2)', linspace(0.75, 1, n2)'; ...
                   linspace(1, 0.80, n2)', linspace(1, 0.20, n2)', linspace(1, 0.15, n2)'];
colormap(gca, cmap);
set(gca, 'XTick', 1:P, 'YTick', 1:P, 'FontSize', 7, ...
    'YTickLabel', arrayfun(@(j) sprintf('%s %2d', names{j}, j), 1:P, 'UniformOutput', false));
xlabel('Feature number');
hold on;
edges = find(diff(gIdx)) + 0.5;
for e = edges, plot([e e], [0.5 P + 0.5], 'k-');  plot([0.5 P + 0.5], [e e], 'k-'); end
title('Pearson correlation between the 29 features');
print(f3, fullfile(figDir, 'fig8_3_feature_correlation.png'), '-dpng', '-r150');

% Fig 8.4  first two principal components of the kept (standardised) features
f4 = figure('Name', 'Fig 8.4 PCA', 'Position', [100 100 760 600]);
Xk = X(:, Sel.keep);
Xk = (Xk - mean(Xk, 1)) ./ max(std(Xk, 0, 1), eps);
[~, Sv, V] = svd(Xk, 'econ');
pc = Xk * V(:, 1:2);
ev = diag(Sv) .^ 2;  ev = ev / sum(ev);
pal = [0.55 0.55 0.55; 0.00 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.47 0.67 0.19; ...
       0.30 0.75 0.93; 0.64 0.08 0.18; 0.85 0.33 0.10; 0.10 0.10 0.10];
hold on;
for c = 1:K
    m = y == c - 1;
    plot(pc(m, 1), pc(m, 2), '.', 'Color', pal(c, :), 'MarkerSize', 6);
end
xlabel(sprintf('PC 1 (%.0f%% of variance)', 100 * ev(1)));
ylabel(sprintf('PC 2 (%.0f%% of variance)', 100 * ev(2)));
axis([quantiles(pc(:, 1), [0.002 0.998]), quantiles(pc(:, 2), [0.002 0.998])]);  grid on;  box on;
legend(cn, 'Location', 'eastoutside');
title('Labelled windows in the first two principal components');
print(f4, fullfile(figDir, 'fig8_4_pca.png'), '-dpng', '-r150');

%% 5. Results file
out = struct();
out.nWindows = numel(idx);  out.nFeatures = P;  out.msPerWindow = msPerWin;
out.names = names;  out.groups = groups;  out.F = Sel.F;
out.order = Sel.order;  out.keep = Sel.keep;  out.rMax = rMax;
out.dropped = Sel.dropped;  out.groupSummary = grp;
out.medians = struct('features', {show}, 'values', med);
out.pcaVariance = ev(1:2)';
fid = fopen(fullfile(dataDir, 'week8_results.json'), 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('\nSaved data/features.mat, data/features.csv, data/week8_results.json, fig8_1 ... fig8_4\n');

