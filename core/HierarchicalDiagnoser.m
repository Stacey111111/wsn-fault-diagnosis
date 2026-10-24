classdef HierarchicalDiagnoser < handle
%HIERARCHICALDIAGNOSER  Two-stage fault diagnosis: a cheap fault detector
%   on every window, and the full ensemble only for suspicious windows.
%
%   hd = HierarchicalDiagnoser(opts)
%     opts.stage1Hp      decision tree of stage 1 (default minLeaf 20,
%                        maxSplits 20, as the tuned tree of Week 9)
%     opts.targetRecall  share of the fault windows that stage 1 must pass
%                        on (default 0.95)
%     opts.stage2Name    'rf', 'logitboost', ... (train_model) or 'stack'
%     opts.stage2Hp      hyperparameters of stage 2 (for 'stack': struct
%                        with fields names and hps, see StackingEnsemble)
%     opts.nInner        inner folds used to choose the threshold (default 4)
%
%   hd.fit(X, y, groups)
%     Stage 1 is a binary decision tree (fault vs. normal). Its threshold
%     tau on the predicted fault probability is the largest value that
%     still passes targetRecall of the fault windows, measured on
%     out-of-fold probabilities of an inner grouped CV, so that tau is not
%     fitted to windows the tree has seen. Stage 2 is the 9-class model,
%     trained on all windows.
%   [yPred, pFault, escalated] = hd.predict(X)
%     windows with pFault >= tau are escalated to stage 2 and get its
%     label; all other windows are labelled normal (0).
%   tau = HierarchicalDiagnoser.thresholdFor(pFault, isFault, target)
%     (never below 1e-6, so windows with fault probability 0 stay at stage 1)
%
%   Week 15 deliverable, CS 5395 Independent Study (Xiaohui San).

    properties
        stage1Hp = struct('minLeaf', 20, 'maxSplits', 20)
        targetRecall = 0.95
        stage2Name = 'rf'
        stage2Hp = struct()
        nInner = 4
        innerSeed = 1
        stage1 = []
        stage2 = []
        tau = 0.5
        oofRecall = NaN       % stage-1 recall of the faults at tau (inner CV)
    end

    methods
        function obj = HierarchicalDiagnoser(opts)
            if nargin < 1, opts = struct(); end
            f = fieldnames(opts);
            for i = 1:numel(f), obj.(f{i}) = opts.(f{i}); end
        end

        function fit(obj, X, y, groups)
            y = y(:);  isF = y > 0;
            fin = grouped_folds(groups, obj.nInner, obj.innerSeed);
            p = zeros(numel(y), 1);
            for f = 1:obj.nInner
                te = fin == f;
                t = HierarchicalDiagnoser.trainStage1(X(~te, :), isF(~te), obj.stage1Hp);
                p(te) = HierarchicalDiagnoser.faultProb(t, X(te, :));
            end
            obj.tau = HierarchicalDiagnoser.thresholdFor(p, isF, obj.targetRecall);
            obj.oofRecall = mean(p(isF) >= obj.tau);
            obj.stage1 = HierarchicalDiagnoser.trainStage1(X, isF, obj.stage1Hp);
            if strcmpi(obj.stage2Name, 'stack')
                se = StackingEnsemble(obj.stage2Hp.names, obj.stage2Hp.hps, struct('classes', 0:8));
                se.fit(X, y, groups);
                obj.stage2 = se;
            else
                obj.stage2 = train_model(obj.stage2Name, X, y, obj.stage2Hp);
            end
        end

        function [yp, pF, esc] = predict(obj, X)
            pF = HierarchicalDiagnoser.faultProb(obj.stage1, X);
            esc = pF >= obj.tau;
            yp = zeros(size(X, 1), 1);
            if any(esc)
                if isa(obj.stage2, 'StackingEnsemble')
                    yp(esc) = obj.stage2.predict(X(esc, :));
                else
                    yp(esc) = predict_model(obj.stage2, X(esc, :));
                end
            end
        end
    end

    methods (Static)
        function t = trainStage1(X, isF, hp)
            t = fitctree(X, double(isF), 'SplitCriterion', 'gdi', 'MinLeafSize', hp.minLeaf, ...
                         'MaxNumSplits', hp.maxSplits, 'ClassNames', [0 1]);
        end

        function p = faultProb(t, X)
            [~, s] = predict(t, X);
            p = s(:, 2);
        end

        function tau = thresholdFor(p, isF, target)
            %THRESHOLDFOR  Largest threshold that keeps at least the target
            %   share of the fault windows at or above it.
            pf = sort(p(isF), 'descend');
            if isempty(pf), tau = 0.5;  return; end
            k = min(numel(pf), max(1, ceil(target * numel(pf))));
            tau = pf(k);
            % A threshold of 0 would escalate every window (all probabilities
            % are >= 0). If the target can only be met that way, keep the
            % windows with probability 0 at stage 1 instead: the recall is then
            % the largest one stage 1 can reach without escalating everything.
            tau = max(tau, 1e-6);
        end
    end
end
