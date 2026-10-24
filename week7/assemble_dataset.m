function D = assemble_dataset(runs, opts)
%ASSEMBLE_DATASET  Window table and labelled subset from simulated runs.
%
%   D = assemble_dataset(runs, opts)   (see build_dataset for opts)
%
%   The labelled dataset (D.win.inSet) keeps every fault window, every normal
%   window inside an environmental event (hard negatives: the field changes
%   but no sensor is faulty), and a random sample of opts.normMult x (median
%   number of windows per fault class) of the other ("quiet") normal
%   windows. Transition windows (label -1) are never in the set.
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 2, opts = struct(); end
    d = struct('winLen', 12, 'normMult', 2, 'selSeed', 7);
    f = fieldnames(opts);
    for i = 1:numel(f)
        if isfield(d, f{i}), d.(f{i}) = opts.(f{i}); end
    end
    W = [];
    for k = 1:numel(runs)
        Wk = make_windows(runs(k), k, d.winLen);
        if isempty(W)
            W = Wk;
        else
            fn = fieldnames(W);
            for j = 1:numel(fn), W.(fn{j}) = [W.(fn{j}); Wk.(fn{j})]; end
        end
    end
    % labelled subset
    cnt = accumarray(W.label(W.label > 0), 1, [8 1]);
    nNorm = d.normMult * round(median(cnt));
    isNorm = W.label == 0;
    evNorm = find(isNorm & W.eventFrac > 0);
    other = find(isNorm & W.eventFrac == 0);
    st = rng();  rng(d.selSeed);                        % fixed seed for the selection
    other = other(randperm(numel(other), min(numel(other), nNorm)));
    rng(st);
    W.inSet = W.label > 0;
    W.inSet([evNorm; other]) = true;

    D = struct();
    D.runs = runs;
    D.win = W;
    D.classNames = {'normal', 'offset', 'gain', 'drift', 'stuck', 'spike', 'noise', 'loss', 'battery'};
    D.winLen = d.winLen;
    D.seeds = [runs.seed];
end
