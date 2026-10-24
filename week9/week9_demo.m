%% WEEK9_DEMO  Single classifiers I: SVM, decision tree, k-NN
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  Trains and evaluates three single classifiers on the Week 8 features:
%    - SVM with an RBF kernel (one-vs-one, fitcecoc + templateSVM)
%    - CART decision tree (fitctree)
%    - k-nearest neighbours (fitcknn)
%  Evaluation pipeline (shared with Weeks 10-13):
%    - grouped cross-validation: the 10 simulated networks are split into
%      5 folds of 2 networks, so every test network is unseen;
%    - hyperparameters tuned by grid search with one fold assignment
%      (seed 99), then 3 repeats of grouped 5-fold CV with other fold
%      assignments (seeds 1-3) give 15 test-fold scores per model;
%    - the Week 7 threshold rules are scored on the same test folds.
%
%  Needs ../data/features.mat (week8_demo) and ../data/week7_baselines.mat
%  (week7_demo). Needs the Statistics and Machine Learning Toolbox.
%
%  Outputs:
%    ../data/week9_cv.mat                 fold assignments and predictions
%    ../data/week9_results.json           all numbers used in the report
%    ../figures/fig9_1_tuning.png         grid-search results
%    ../figures/fig9_2_confusion.png      confusion matrices of the 3 models
%    ../figures/fig9_3_class_f1.png       F1 per class, models vs. thresholds
%    ../figures/fig9_4_fold_scores.png    macro-F1 of every test fold

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), here);
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');

F = load(fullfile(dataDir, 'features.mat'));
B = load(fullfile(dataDir, 'week7_baselines.mat'));
assert(isequal(B.y, F.y), 'week7_baselines.mat and features.mat do not match; rerun week7 and week8.');
y = F.y;  groups = F.run;  cn = F.classNames;  K = numel(cn);
Xsel = F.X(:, F.keep);  Xall = F.X;
fprintf('%d windows, %d networks, %d selected features (of %d)\n\n', numel(y), ...
        numel(unique(groups)), size(Xsel, 2), size(Xall, 2));
opts = struct('nFolds', 5, 'seeds', 1:3, 'classes', 0:K - 1);

models = {'svm', 'tree', 'knn'};
labels = {'SVM (RBF)', 'Decision tree', 'k-NN'};
tune = struct();  E = struct();  Eall = struct();
for m = 1:numel(models)
    nm = models{m};
    fprintf('== %s: grid search (grouped 5-fold CV, seed 99)\n', labels{m});
    t0 = tic;
    [best, T] = tune_model(Xsel, y, groups, nm);
    tune.(nm) = struct('best', best, 'table', T, 'seconds', toc(t0));
    fprintf('   best: %s\n', hp_string(best));
    fprintf('== %s: 3 x grouped 5-fold CV\n', labels{m});
    E.(nm) = cv_evaluate(Xsel, y, groups, nm, best, opts);
    Eall.(nm) = cv_evaluate(Xall, y, groups, nm, best, opts);
    fprintf('   macro-F1 %.3f +/- %.3f (selected features), %.3f +/- %.3f (all features)\n\n', ...
            mean(E.(nm).foldF1), std(E.(nm).foldF1), mean(Eall.(nm).foldF1), std(Eall.(nm).foldF1));
end

%% Threshold baseline on the same test folds
fold = E.svm.fold;                                     % identical for every model
thrF1 = zeros(size(E.svm.foldF1));  j = 0;
for r = 1:size(fold, 2)
    for k = 1:opts.nFolds
        j = j + 1;  te = fold(:, r) == k;
        M = class_metrics(y(te), B.yThr(te), 0:K - 1);
        thrF1(j) = M.macroF1;
    end
end
Mthr = class_metrics(y, B.yThr, 0:K - 1);

%% Summary tables
fprintf('%-14s %-34s %13s %9s %9s %9s %10s\n', 'Model', 'Hyperparameters', 'Macro-F1', ...
        'Accuracy', 'Train s', 'ms/win', 'Size kB');
fprintf('%-14s %-34s %6.3f+/-%.3f %9.3f %9s %9s %10s\n', 'Thresholds', '(fixed rules, Week 7)', ...
        mean(thrF1), std(thrF1), Mthr.accuracy, '-', '-', '-');
for m = 1:numel(models)
    e = E.(models{m});
    fprintf('%-14s %-34s %6.3f+/-%.3f %9.3f %9.2f %9.3f %10.1f\n', labels{m}, hp_string(e.hp), ...
            mean(e.foldF1), std(e.foldF1), e.metrics.accuracy, mean(e.trainTime), ...
            e.msPerWindow, e.modelBytes / 1024);
end

fprintf('\nF1 per class\n%-14s', '');  fprintf('%9s', cn{:});  fprintf('\n');
fprintf('%-14s', 'Thresholds');  fprintf('%9.3f', Mthr.f1);  fprintf('\n');
for m = 1:numel(models)
    fprintf('%-14s', labels{m});  fprintf('%9.3f', E.(models{m}).metrics.f1);  fprintf('\n');
end

% fault detection and false alarms (multi-class prediction > 0 = fault)
isF = y > 0;  isEv = y == 0 & F.eventFrac > 0;  isQ = y == 0 & F.eventFrac == 0;
det = @(yp) struct('recall', mean(yp(isF) > 0), 'farQuiet', mean(yp(isQ) > 0), ...
                   'farEvent', mean(yp(isEv) > 0));
fprintf('\n%-14s %10s %10s %10s\n', 'Detection', 'Recall', 'FA quiet', 'FA event');
D0 = det(B.yThr);
fprintf('%-14s %10.3f %10.3f %10.3f\n', 'Thresholds', D0.recall, D0.farQuiet, D0.farEvent);
detect = struct();
for m = 1:numel(models)
    yp = E.(models{m}).yPred;
    Dm = struct('recall', mean(mean(yp(isF, :) > 0)), 'farQuiet', mean(mean(yp(isQ, :) > 0)), ...
                'farEvent', mean(mean(yp(isEv, :) > 0)));
    detect.(models{m}) = Dm;
    fprintf('%-14s %10.3f %10.3f %10.3f\n', labels{m}, Dm.recall, Dm.farQuiet, Dm.farEvent);
end

% decision tree trained on all windows: size and most important features
rng(9);
tr = train_model('tree', Xsel, y, E.tree.hp);
imp = predictorImportance(tr);
[~, oi] = sort(imp, 'descend');
selNames = F.names(F.keep);
rootVar = tr.CutPredictor{1};
tok = regexp(rootVar, '^x(\d+)$', 'tokens', 'once');   % unnamed predictors are called x1, x2, ...
if ~isempty(tok), rootVar = selNames{str2double(tok{1})}; end
fprintf('\nTree on all windows: %d leaves, root split on %s\n', sum(~tr.IsBranchNode), rootVar);
fprintf('Most important features (tree): %s\n', strjoin(selNames(oi(1:6)), ', '));
treeInfo = struct('leaves', sum(~tr.IsBranchNode), 'root', rootVar, ...
                  'topFeatures', {selNames(oi(1:6))}, 'importance', imp(oi(1:6)));

save(fullfile(dataDir, 'week9_cv.mat'), 'E', 'Eall', 'tune', 'thrF1', '-v7');

%% Figures
% Fig 9.1  grid search
f1 = figure('Name', 'Fig 9.1 Tuning', 'Position', [100 100 1300 400]);
subplot(1, 3, 1);
T = tune.svm.table;  hp = [T.hp];
Cs = unique([hp.C]);  ks = unique([hp.kernelScale]);
Z = zeros(numel(Cs), numel(ks));
for g = 1:numel(T), Z(Cs == hp(g).C, ks == hp(g).kernelScale) = T(g).macroF1; end
imagesc(Z);  colorbar;  title('SVM: macro-F1');
set(gca, 'XTick', 1:numel(ks), 'XTickLabel', arrayfun(@num2str, ks, 'UniformOutput', false), ...
    'YTick', 1:numel(Cs), 'YTickLabel', arrayfun(@num2str, Cs, 'UniformOutput', false));
xlabel('Kernel scale');  ylabel('Box constraint C');
for a = 1:numel(Cs), for b = 1:numel(ks)
    text(b, a, sprintf('%.3f', Z(a, b)), 'HorizontalAlignment', 'center', 'FontSize', 8);
end, end
subplot(1, 3, 2);
T = tune.tree.table;  hp = [T.hp];
ls = unique([hp.minLeaf]);  ms = unique([hp.maxSplits]);
Z = zeros(numel(ls), numel(ms));
for g = 1:numel(T), Z(ls == hp(g).minLeaf, ms == hp(g).maxSplits) = T(g).macroF1; end
imagesc(Z);  colorbar;  title('Decision tree: macro-F1');
set(gca, 'XTick', 1:numel(ms), 'XTickLabel', arrayfun(@num2str, ms, 'UniformOutput', false), ...
    'YTick', 1:numel(ls), 'YTickLabel', arrayfun(@num2str, ls, 'UniformOutput', false));
xlabel('Max. number of splits');  ylabel('Min. leaf size');
for a = 1:numel(ls), for b = 1:numel(ms)
    text(b, a, sprintf('%.3f', Z(a, b)), 'HorizontalAlignment', 'center', 'FontSize', 8);
end, end
subplot(1, 3, 3);
T = tune.knn.table;  hp = [T.hp];
kk = unique([hp.k]);
fe = arrayfun(@(g) strcmp(hp(g).weight, 'equal'), 1:numel(hp));
vE = zeros(size(kk));  vI = zeros(size(kk));
for g = 1:numel(T)
    if fe(g), vE(kk == hp(g).k) = T(g).macroF1; else, vI(kk == hp(g).k) = T(g).macroF1; end
end
plot(kk, vE, 'o-', kk, vI, 's-', 'LineWidth', 1.2);  grid on;
xlabel('Number of neighbours k');  ylabel('Macro-F1');  title('k-NN');
legend({'equal weights', 'inverse-distance weights'}, 'Location', 'south');
print(f1, fullfile(figDir, 'fig9_1_tuning.png'), '-dpng', '-r150');

% Fig 9.2  confusion matrices (row-normalised)
f2 = figure('Name', 'Fig 9.2 Confusion', 'Position', [100 100 1500 470]);
cmap = [linspace(1, 0.13, 256)', linspace(1, 0.30, 256)', linspace(1, 0.60, 256)'];
for m = 1:numel(models)
    subplot(1, 3, m);
    C = E.(models{m}).metrics.confusion;
    Cn = C ./ max(sum(C, 2), 1);
    imagesc(Cn, [0 1]);  colormap(gca, cmap);  axis square;
    for a = 1:K, for b = 1:K
        if Cn(a, b) >= 0.005
            if Cn(a, b) > 0.5, tc = 'w'; else, tc = 'k'; end
            text(b, a, sprintf('%.0f', 100 * Cn(a, b)), 'HorizontalAlignment', 'center', ...
                 'Color', tc, 'FontSize', 7);
        end
    end, end
    set(gca, 'XTick', 1:K, 'XTickLabel', cellfun(@(s) s(1:3), cn, 'UniformOutput', false), ...
        'YTick', 1:K, 'YTickLabel', cn, 'FontSize', 8);
    xlabel('Predicted');  if m == 1, ylabel('True class'); end
    title(sprintf('%s (macro-F1 %.3f)', labels{m}, E.(models{m}).metrics.macroF1));
end
print(f2, fullfile(figDir, 'fig9_2_confusion.png'), '-dpng', '-r150');

% Fig 9.3  F1 per class
f3 = figure('Name', 'Fig 9.3 F1 per class', 'Position', [100 100 1100 420]);
V = [Mthr.f1; E.svm.metrics.f1; E.tree.metrics.f1; E.knn.metrics.f1]';
bh = bar(V);
col = [0.60 0.60 0.60; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.85 0.33 0.10];
for i = 1:4, set(bh(i), 'FaceColor', col(i, :)); end
set(gca, 'XTick', 1:K, 'XTickLabel', cn);  ylim([0 1.15]);  grid on;  ylabel('F1');
legend([{'Thresholds'}, labels], 'Location', 'north', 'Orientation', 'horizontal');
title('F1 per class (pooled over 3 x 5 test folds)');
print(f3, fullfile(figDir, 'fig9_3_class_f1.png'), '-dpng', '-r150');

% Fig 9.4  macro-F1 of each test fold
f4 = figure('Name', 'Fig 9.4 Fold scores', 'Position', [100 100 900 420]);
hold on;
S = [thrF1, E.svm.foldF1, E.tree.foldF1, E.knn.foldF1];
Sa = [nan(size(thrF1)), Eall.svm.foldF1, Eall.tree.foldF1, Eall.knn.foldF1];
for i = 1:4
    xj = i - 0.12 + 0.08 * ((1:size(S, 1))' / size(S, 1) - 0.5);
    plot(xj, S(:, i), 'o', 'Color', col(i, :), 'MarkerFaceColor', col(i, :), 'MarkerSize', 4);
    plot([i - 0.25 i + 0.01], mean(S(:, i)) * [1 1], 'k-', 'LineWidth', 2);
    if i > 1
        plot(xj + 0.24, Sa(:, i), 'o', 'Color', col(i, :), 'MarkerSize', 4);
        plot([i + 0.01 i + 0.27], mean(Sa(:, i)) * [1 1], 'k:', 'LineWidth', 2);
    end
end
set(gca, 'XTick', 1:4, 'XTickLabel', [{'Thresholds'}, labels]);  xlim([0.5 4.5]);
ylabel('Macro-F1 of a test fold');  grid on;  box on;
title(sprintf('15 test folds per method: filled = %d selected features, open = all %d', size(Xsel, 2), size(Xall, 2)));
print(f4, fullfile(figDir, 'fig9_4_fold_scores.png'), '-dpng', '-r150');

%% Results file
out = struct();
out.nWindows = numel(y);  out.nSelected = size(Xsel, 2);  out.nAll = size(Xall, 2);
out.threshold = struct('foldF1', thrF1, 'macroF1', mean(thrF1), 'sd', std(thrF1), ...
                       'f1', Mthr.f1, 'accuracy', Mthr.accuracy, 'detect', D0);
for m = 1:numel(models)
    nm = models{m};  e = E.(nm);
    out.(nm) = struct('label', labels{m}, 'best', hp_string(e.hp), ...
        'grid', struct('text', {tune.(nm).table.text}, 'macroF1', {tune.(nm).table.macroF1}), ...
        'tuneSeconds', tune.(nm).seconds, ...
        'foldF1', e.foldF1, 'macroF1', mean(e.foldF1), 'sd', std(e.foldF1), ...
        'accuracy', e.metrics.accuracy, 'precision', e.metrics.precision, ...
        'recall', e.metrics.recall, 'f1', e.metrics.f1, 'confusion', e.metrics.confusion, ...
        'trainSeconds', mean(e.trainTime), 'msPerWindow', e.msPerWindow, ...
        'modelKB', e.modelBytes / 1024, 'detect', detect.(nm), ...
        'allFeatures', struct('macroF1', mean(Eall.(nm).foldF1), 'sd', std(Eall.(nm).foldF1)));
end
out.tree.final = treeInfo;
fid = fopen(fullfile(dataDir, 'week9_results.json'), 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('\nSaved data/week9_cv.mat, data/week9_results.json and fig9_1 ... fig9_4\n');
