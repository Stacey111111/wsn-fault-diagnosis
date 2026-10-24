%% WEEK10_DEMO  Single classifiers II: random forest, neural network, and
%  the comparison of all single models
%  CS 5395 Independent Study - Ensemble Learning-Based Fault Diagnosis for WSNs
%  Student: Xiaohui San            Advisor: Dr. Hongchi Shi
%
%  1. Tunes and evaluates a random forest (TreeBagger) and a neural network
%     (fitcnet) with the Week 9 pipeline (same features, same grouped
%     folds, same seeds).
%  2. Compares all five single models and the Week 7 threshold rules on the
%     same 15 test folds: macro-F1, per-class F1, fault detection and false
%     alarms, training and prediction cost, model size.
%  3. Tests whether the differences are significant: Friedman test over
%     the 15 folds, then pairwise Wilcoxon signed-rank tests with Holm
%     correction.
%  4. Random-forest out-of-bag permutation importance vs. the Week 8
%     ANOVA F ranking.
%  These results are the reference for the ensembles of Weeks 11-13.
%
%  Needs ../data/features.mat, ../data/week7_baselines.mat and
%  ../data/week9_cv.mat (run week7_demo, week8_demo and week9_demo first).
%
%  Outputs:
%    ../data/week10_cv.mat                fold predictions of RF and NN
%    ../data/week10_results.json          all numbers used in the report
%    ../figures/fig10_1_fold_scores.png   macro-F1 of every test fold, 6 methods
%    ../figures/fig10_2_class_f1.png      F1 per class and method (heat map)
%    ../figures/fig10_3_importance.png    RF importance vs. ANOVA F
%    ../figures/fig10_4_cost.png          macro-F1 vs. prediction cost
%    ../figures/fig10_5_confusion_best.png confusion matrix of the best model

clear; close all; clc;
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(fullfile(here, '..', 'core'), here);
figDir = fullfile(here, '..', 'figures');  dataDir = fullfile(here, '..', 'data');

F = load(fullfile(dataDir, 'features.mat'));
B = load(fullfile(dataDir, 'week7_baselines.mat'));
W9 = load(fullfile(dataDir, 'week9_cv.mat'));
y = F.y;  groups = F.run;  cn = F.classNames;  K = numel(cn);
Xsel = F.X(:, F.keep);  selNames = F.names(F.keep);
opts = struct('nFolds', 5, 'seeds', 1:3, 'classes', 0:K - 1);

%% 1. Random forest and neural network
E = W9.E;  tune = W9.tune;
newM = {'rf', 'nn'};  newL = {'Random forest', 'Neural network'};
for m = 1:2
    nm = newM{m};
    fprintf('== %s: grid search (grouped 5-fold CV, seed 99)\n', newL{m});
    t0 = tic;
    [best, T] = tune_model(Xsel, y, groups, nm);
    tune.(nm) = struct('best', best, 'table', T, 'seconds', toc(t0));
    fprintf('   best: %s\n', hp_string(best));
    fprintf('== %s: 3 x grouped 5-fold CV\n', newL{m});
    E.(nm) = cv_evaluate(Xsel, y, groups, nm, best, opts);
    fprintf('   macro-F1 %.3f +/- %.3f\n\n', mean(E.(nm).foldF1), std(E.(nm).foldF1));
end
assert(isequal(E.rf.fold, E.svm.fold), 'Fold assignments differ from Week 9.');
save(fullfile(dataDir, 'week10_cv.mat'), 'E', 'tune', '-v7');

%% 2. Comparison table
models = {'svm', 'tree', 'knn', 'rf', 'nn'};
labels = {'SVM (RBF)', 'Decision tree', 'k-NN', 'Random forest', 'Neural network'};
meth = [{'Thresholds'}, labels];
nM = numel(meth);
S = [W9.thrF1, zeros(numel(W9.thrF1), numel(models))];   % fold x method macro-F1
F1c = zeros(nM, K);
Mthr = class_metrics(y, B.yThr, 0:K - 1);
F1c(1, :) = Mthr.f1;
isF = y > 0;  isEv = y == 0 & F.eventFrac > 0;  isQ = y == 0 & F.eventFrac == 0;
detect = zeros(nM, 3);                                       % recall, FA quiet, FA event
detect(1, :) = [mean(B.yThr(isF) > 0), mean(B.yThr(isQ) > 0), mean(B.yThr(isEv) > 0)];
cost = nan(nM, 3);                                           % train s, ms/window, size kB
acc = zeros(nM, 1);  acc(1) = Mthr.accuracy;
for m = 1:numel(models)
    e = E.(models{m});
    S(:, m + 1) = e.foldF1;
    F1c(m + 1, :) = e.metrics.f1;
    yp = e.yPred;
    detect(m + 1, :) = [mean(mean(yp(isF, :) > 0)), mean(mean(yp(isQ, :) > 0)), ...
                        mean(mean(yp(isEv, :) > 0))];
    cost(m + 1, :) = [mean(e.trainTime), e.msPerWindow, e.modelBytes / 1024];
    acc(m + 1) = e.metrics.accuracy;
end
fprintf('%-15s %14s %9s %8s %9s %9s %9s %9s %9s\n', 'Method', 'Macro-F1', 'Accuracy', ...
        'Recall', 'FA quiet', 'FA event', 'Train s', 'ms/win', 'Size kB');
for i = 1:nM
    fprintf('%-15s %7.3f+/-%.3f %9.3f %8.3f %9.3f %9.3f %9.2f %9.3f %9.1f\n', meth{i}, ...
            mean(S(:, i)), std(S(:, i)), acc(i), detect(i, :), cost(i, :));
end
fprintf('\nF1 per class\n%-15s', '');  fprintf('%9s', cn{:});  fprintf('\n');
for i = 1:nM
    fprintf('%-15s', meth{i});  fprintf('%9.3f', F1c(i, :));  fprintf('\n');
end

%% 3. Significance tests
[pF, chi2, meanRank] = friedman_test(S);
fprintf('\nFriedman test over %d folds and %d methods: chi2 = %.2f, p = %.2g\n', ...
        size(S, 1), nM, chi2, pF);
fprintf('Mean rank (1 = best): ');
for i = 1:nM, fprintf('%s %.2f; ', meth{i}, meanRank(i)); end
fprintf('\n');
pairs = nchoosek(1:nM, 2);
pRaw = zeros(size(pairs, 1), 1);
for q = 1:size(pairs, 1)
    pRaw(q) = signrank_exact(S(:, pairs(q, 1)), S(:, pairs(q, 2)));
end
pHolm = holm_adjust(pRaw);
fprintf('\nPairwise Wilcoxon signed-rank tests (Holm-adjusted p)\n');
for q = 1:size(pairs, 1)
    a = pairs(q, 1);  b = pairs(q, 2);
    fprintf('  %-15s vs %-15s  mean diff %+.3f  p = %.4f  p(Holm) = %.4f%s\n', meth{a}, meth{b}, ...
            mean(S(:, a) - S(:, b)), pRaw(q), pHolm(q), repmat(' *', 1, double(pHolm(q) < 0.05)));
end
[~, bestM] = max(mean(S(:, 2:end), 1));
bestM = bestM + 1;
fprintf('Best single model: %s\n', meth{bestM});

%% 4. Random-forest importance (OOB permutation) vs. ANOVA F
rng(10);
hpRF = E.rf.hp;  hpRF.importance = true;
rf = train_model('rf', Xsel, y, hpRF);
imp = rf.OOBPermutedPredictorDeltaError;
Fsel = F.F(F.keep);
[~, oI] = sort(imp, 'descend');
rI = zeros(size(imp));  rI(oI) = 1:numel(imp);
[~, oF] = sort(Fsel, 'descend');
rF = zeros(size(Fsel));  rF(oF) = 1:numel(Fsel);
cc = corrcoef(rI, rF);  rhoRank = cc(1, 2);              % Spearman rank correlation
fprintf('\nRF permutation importance (top 8): %s\n', strjoin(selNames(oI(1:8)), ', '));
fprintf('Spearman correlation of RF importance and ANOVA F ranks: %.2f\n', rhoRank);

%% 5. Figures
col = [0.60 0.60 0.60; 0.00 0.45 0.74; 0.47 0.67 0.19; 0.85 0.33 0.10; 0.49 0.18 0.56; 0.93 0.69 0.13];

% Fig 10.1  fold scores
f1 = figure('Name', 'Fig 10.1 Fold scores', 'Position', [100 100 1000 430]);
hold on;
for i = 1:nM
    xj = i + 0.25 * ((1:size(S, 1))' / size(S, 1) - 0.5);
    plot(xj, S(:, i), 'o', 'Color', col(i, :), 'MarkerFaceColor', col(i, :), 'MarkerSize', 4);
    he = errorbar(i + 0.25, mean(S(:, i)), std(S(:, i)), 'ks');
    set(he, 'MarkerFaceColor', 'k', 'Color', 'k');
end
set(gca, 'XTick', 1:nM, 'XTickLabel', meth);  xlim([0.5 nM + 0.6]);  grid on;  box on;
ylabel('Macro-F1 of a test fold');
title(sprintf('15 test folds per method (square = mean +/- sd); Friedman p = %.1g', pF));
print(f1, fullfile(figDir, 'fig10_1_fold_scores.png'), '-dpng', '-r150');

% Fig 10.2  F1 per class and method
f2 = figure('Name', 'Fig 10.2 F1 per class', 'Position', [100 100 1000 420]);
imagesc(F1c, [0 1]);  colorbar;
colormap(gca, [linspace(1, 0.13, 256)', linspace(1, 0.30, 256)', linspace(1, 0.60, 256)']);
for a = 1:nM, for b = 1:K
    if F1c(a, b) > 0.6, tc = 'w'; else, tc = 'k'; end
    text(b, a, sprintf('%.2f', F1c(a, b)), 'HorizontalAlignment', 'center', 'Color', tc, 'FontSize', 9);
end, end
set(gca, 'XTick', 1:K, 'XTickLabel', cn, 'YTick', 1:nM, 'YTickLabel', meth);
title('F1 per class (pooled over 15 test folds)');
print(f2, fullfile(figDir, 'fig10_2_class_f1.png'), '-dpng', '-r150');

% Fig 10.3  RF importance vs. ANOVA F
f3 = figure('Name', 'Fig 10.3 Importance', 'Position', [100 100 1100 460]);
subplot(1, 2, 1);
nTop = min(15, numel(imp));
hb = barh(imp(oI(1:nTop)));
set(hb, 'FaceColor', col(5, :));
set(gca, 'YDir', 'reverse', 'YTick', 1:nTop, 'YTickLabel', selNames(oI(1:nTop)), 'FontSize', 8);
xlabel('Increase in OOB error when the feature is permuted');  grid on;
title('Random forest: permutation importance');
subplot(1, 2, 2);
plot(rF, rI, 'o', 'MarkerFaceColor', col(2, :), 'Color', col(2, :));  hold on;
for j = 1:numel(imp)
    text(rF(j) + 0.3, rI(j), selNames{j}, 'FontSize', 7);
end
plot([0 numel(imp) + 1], [0 numel(imp) + 1], 'k:');
axis([0 numel(imp) + 4 0 numel(imp) + 1]);  set(gca, 'YDir', 'reverse');  grid on;  box on;
xlabel('Rank by ANOVA F (Week 8)');  ylabel('Rank by RF importance');
title(sprintf('Rank agreement (Spearman \\rho = %.2f)', rhoRank));
print(f3, fullfile(figDir, 'fig10_3_importance.png'), '-dpng', '-r150');

% Fig 10.4  accuracy vs. cost
f4 = figure('Name', 'Fig 10.4 Cost', 'Position', [100 100 1000 420]);
subplot(1, 2, 1);
hold on;
for i = 2:nM
    plot(cost(i, 2), mean(S(:, i)), 'o', 'MarkerSize', 9, 'MarkerFaceColor', col(i, :), 'Color', col(i, :));
    text(cost(i, 2) * 1.15, mean(S(:, i)), meth{i}, 'FontSize', 8);
end
set(gca, 'XScale', 'log');  grid on;  box on;
xlabel('Prediction time per window (ms, log scale)');  ylabel('Mean macro-F1');
title('Accuracy vs. prediction cost');
subplot(1, 2, 2);
hold on;
for i = 2:nM
    plot(cost(i, 3), mean(S(:, i)), 's', 'MarkerSize', 9, 'MarkerFaceColor', col(i, :), 'Color', col(i, :));
    text(cost(i, 3) * 1.15, mean(S(:, i)), meth{i}, 'FontSize', 8);
end
set(gca, 'XScale', 'log');  grid on;  box on;
xlabel('Model size (kB, log scale)');  ylabel('Mean macro-F1');
title('Accuracy vs. model size');
print(f4, fullfile(figDir, 'fig10_4_cost.png'), '-dpng', '-r150');

% Fig 10.5  confusion matrix of the best model
f5 = figure('Name', 'Fig 10.5 Best model', 'Position', [100 100 640 560]);
C = E.(models{bestM - 1}).metrics.confusion;
Cn = C ./ max(sum(C, 2), 1);
imagesc(Cn, [0 1]);  colorbar;
colormap(gca, [linspace(1, 0.13, 256)', linspace(1, 0.30, 256)', linspace(1, 0.60, 256)']);
for a = 1:K, for b = 1:K
    if C(a, b) >= 0.5
        if Cn(a, b) > 0.5, tc = 'w'; else, tc = 'k'; end
        text(b, a, sprintf('%.0f', C(a, b)), 'HorizontalAlignment', 'center', 'Color', tc, 'FontSize', 8);
    end
end, end
set(gca, 'XTick', 1:K, 'XTickLabel', cellfun(@(s) s(1:3), cn, 'UniformOutput', false), ...
    'YTick', 1:K, 'YTickLabel', cn);
xlabel('Predicted class');  ylabel('True class');
title(sprintf('%s: windows per repeat (macro-F1 %.3f)', meth{bestM}, ...
              E.(models{bestM - 1}).metrics.macroF1));
print(f5, fullfile(figDir, 'fig10_5_confusion_best.png'), '-dpng', '-r150');

%% 6. Results file
out = struct();
out.methods = meth;  out.classNames = cn;
out.foldF1 = S;  out.macroF1 = mean(S, 1);  out.sd = std(S, 0, 1);  out.accuracy = acc';
out.f1 = F1c;  out.detect = detect;  out.cost = cost;
out.friedman = struct('chi2', chi2, 'p', pF, 'meanRank', meanRank);
out.wilcoxon = struct('a', meth(pairs(:, 1)), 'b', meth(pairs(:, 2)), ...
                      'meanDiff', num2cell(mean(S(:, pairs(:, 1)) - S(:, pairs(:, 2)), 1)), ...
                      'p', num2cell(pRaw'), 'pHolm', num2cell(pHolm'));
out.best = meth{bestM};
out.bestConfusion = C;
for m = 1:2
    nm = newM{m};
    out.(nm) = struct('best', hp_string(E.(nm).hp), ...
        'grid', struct('text', {tune.(nm).table.text}, 'macroF1', {tune.(nm).table.macroF1}), ...
        'tuneSeconds', tune.(nm).seconds, 'precision', E.(nm).metrics.precision, ...
        'recall', E.(nm).metrics.recall);
end
out.importance = struct('names', {selNames}, 'deltaError', imp, 'rankRF', rI, 'rankF', rF, ...
                        'spearman', rhoRank);
fid = fopen(fullfile(dataDir, 'week10_results.json'), 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('\nSaved data/week10_cv.mat, data/week10_results.json and fig10_1 ... fig10_5\n');
