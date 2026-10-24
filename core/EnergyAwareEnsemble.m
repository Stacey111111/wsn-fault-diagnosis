classdef EnergyAwareEnsemble < handle
%ENERGYAWAREENSEMBLE  Choose which base models an ensemble uses so that
%   the diagnosis is as accurate as possible within an energy budget.
%
%   ea = EnergyAwareEnsemble(names, cost, store, y, opts)
%     names  1 x M base-model names (order of the Week 13 meta-features)
%     cost   1 x M struct array from model_cost (fields ops, bytes)
%     store  Week 13 per-fold data (week13_cv.mat): test indices, out-of-fold
%            training scores Zoof and test scores Zte of every base model
%     y      labels of all windows
%     opts   optional: classes (default 0:8), Eop (energy per operation,
%            J, default 75e-9), tieModel (index of the model that breaks
%            vote ties when it is part of the ensemble, default 4 = RF)
%
%   Combiners of a subset of base models (mask = 1 x M logical):
%     'stack'  logistic-regression meta-learner on the scores of the chosen
%              models (StackingEnsemble.fitMeta), retrained in every fold
%     'vote'   majority vote of the chosen models (one model = the model)
%
%   e  = ea.energy(mask, combiner)     energy per window [J]: base models +
%                                      combiner (Eop x operations)
%   b  = ea.bytes(mask, combiner)      embedded memory [bytes]
%   f  = ea.evaluate(mask, combiner)   macro-F1 on each of the 15 test folds
%                                      (results are cached)
%   R  = ea.enumerate(combiner)        all 2^M - 1 subsets: struct array
%                                      with mask, energy, bytes, foldF1, macroF1
%   best = ea.selectExhaustive(R, budget)  most accurate subset within the
%                                      energy budget [J/window]
%   path = ea.selectGreedy(budget, combiner)  forward selection: start
%                                      empty, repeatedly add the model with
%                                      the largest gain in macro-F1 per
%                                      joule that fits into the budget, stop
%                                      when no model improves macro-F1
%   front = EnergyAwareEnsemble.paretoFront(energy, f1)  indices of the
%                                      points no other point beats in both
%
%   Week 14 deliverable, CS 5395 Independent Study (Xiaohui San).

    properties
        names = {}
        cost = []
        store = {}
        y = []
        classes = 0:8
        Eop = 75e-9
        tieModel = 4
        cache                 % containers.Map: key -> fold macro-F1
    end

    methods
        function obj = EnergyAwareEnsemble(names, cost, store, y, opts)
            if nargin < 5, opts = struct(); end
            obj.names = names;  obj.cost = cost;  obj.store = store;  obj.y = y(:);
            f = fieldnames(opts);
            for i = 1:numel(f), obj.(f{i}) = opts.(f{i}); end
            obj.cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end

        function ops = combinerOps(obj, mask, combiner)
            K = numel(obj.classes);  q = K * sum(mask);
            if strcmp(combiner, 'stack') && sum(mask) > 1
                ops = 2 * q + K * 2 * q;          % standardise + K linear scores
            elseif sum(mask) > 1
                ops = sum(mask) + K;              % count votes, take the largest
            else
                ops = 0;                          % a single model needs no combiner
            end
        end

        function e = energy(obj, mask, combiner)
            e = obj.Eop * (sum([obj.cost(mask).ops]) + obj.combinerOps(mask, combiner));
        end

        function b = bytes(obj, mask, combiner)
            K = numel(obj.classes);
            b = sum([obj.cost(mask).bytes]);
            if strcmp(combiner, 'stack') && sum(mask) > 1
                b = b + 4 * (K * (K * sum(mask) + 1) + 2 * K * sum(mask));
            end
        end

        function f = evaluate(obj, mask, combiner)
            key = sprintf('%s-%s', combiner, sprintf('%d', mask));
            if isKey(obj.cache, key), f = obj.cache(key);  return; end
            nF = numel(obj.store);  f = zeros(nF, 1);
            K = numel(obj.classes);
            sel = find(mask);
            for j = 1:nF
                s = obj.store{j};
                te = s.test;  trn = true(numel(obj.y), 1);  trn(te) = false;
                Zte = double(s.Zte);
                if strcmp(combiner, 'stack') && numel(sel) > 1
                    se = StackingEnsemble(obj.names, {}, struct('classes', obj.classes));
                    se.Zoof = double(s.Zoof);  se.yTrain = obj.y(trn);
                    cols = cell2mat(arrayfun(@(m) (m - 1) * K + (1:K), sel, 'UniformOutput', false));
                    yp = se.predictMeta(se.fitMeta('logistic', cols), Zte);
                else
                    V = zeros(numel(te), numel(sel));
                    for a = 1:numel(sel)
                        [~, jj] = max(Zte(:, (sel(a) - 1) * K + (1:K)), [], 2);
                        V(:, a) = obj.classes(jj);
                    end
                    if any(sel == obj.tieModel), tb = V(:, sel == obj.tieModel); else, tb = V(:, 1); end
                    yp = majority_vote(V, obj.classes, tb);
                end
                Mj = class_metrics(obj.y(te), yp, obj.classes);
                f(j) = Mj.macroF1;
            end
            obj.cache(key) = f;
        end

        function R = enumerate(obj, combiner)
            M = numel(obj.names);  nS = 2 ^ M - 1;
            R = struct('mask', cell(1, nS), 'models', '', 'combiner', combiner, 'energy', 0, ...
                       'bytes', 0, 'foldF1', [], 'macroF1', 0);
            for s = 1:nS
                mask = bitget(s, 1:M) == 1;
                R(s).mask = mask;  R(s).models = strjoin(obj.names(mask), ' + ');
                R(s).energy = obj.energy(mask, combiner);  R(s).bytes = obj.bytes(mask, combiner);
                R(s).foldF1 = obj.evaluate(mask, combiner);  R(s).macroF1 = mean(R(s).foldF1);
            end
        end

        function best = selectExhaustive(~, R, budget)
            ok = [R.energy] <= budget;
            if ~any(ok), best = [];  return; end
            idx = find(ok);
            [~, b] = max([R(idx).macroF1]);
            best = R(idx(b));
        end

        function path = selectGreedy(obj, budget, combiner)
            M = numel(obj.names);  mask = false(1, M);  cur = 0;  curE = 0;
            path = struct('mask', {}, 'models', {}, 'energy', {}, 'macroF1', {});
            while true
                bestGain = 0;  bestM = 0;  bestF = cur;  bestE = curE;
                for m = find(~mask)
                    cand = mask;  cand(m) = true;
                    e = obj.energy(cand, combiner);
                    if e > budget, continue; end
                    fm = mean(obj.evaluate(cand, combiner));
                    gain = (fm - cur) / max(e - curE, eps);
                    if fm > cur && gain > bestGain
                        bestGain = gain;  bestM = m;  bestF = fm;  bestE = e;
                    end
                end
                if bestM == 0, break; end
                mask(bestM) = true;  cur = bestF;  curE = bestE;
                path(end + 1) = struct('mask', mask, 'models', strjoin(obj.names(mask), ' + '), ...
                                       'energy', curE, 'macroF1', cur); %#ok<AGROW>
            end
        end
    end

    methods (Static)
        function front = paretoFront(energy, f1)
            [~, order] = sortrows([energy(:), -f1(:)]);
            front = [];  bestF = -Inf;
            for i = order'
                if f1(i) > bestF + 1e-12
                    front(end + 1) = i; %#ok<AGROW>
                    bestF = f1(i);
                end
            end
        end
    end
end
