function D = build_dataset(seeds, opts)
%BUILD_DATASET  Simulate several independent networks and build the
%   window-level fault dataset.
%
%   D = build_dataset(seeds, opts)
%     seeds  one random seed per network (default 701:710, i.e. 10 runs)
%     opts   options passed to simulate_run, plus
%            winLen   samples per window (default 12 = 1 h)
%            normMult quiet normal windows kept = normMult x the median
%                     number of windows per fault class (default 2); every
%                     normal window inside an environmental event is kept too
%
%   D.runs     struct array of runs (logs, positions, faults, events)
%   D.win      window table (see make_windows) plus
%              inSet  true for the windows of the labelled dataset
%   D.classNames, D.winLen, D.seeds
%
%   Week 7 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 1 || isempty(seeds), seeds = 701:710; end
    if nargin < 2, opts = struct(); end
    runs = [];
    for k = 1:numel(seeds)
        t0 = tic;
        Rk = simulate_run(seeds(k), opts);
        fprintf('  run %2d (seed %d): %d nodes, %d faults, %.1f s\n', k, seeds(k), ...
                size(Rk.reading, 2), numel(Rk.faults), toc(t0));
        if isempty(runs), runs = Rk; else, runs(k) = Rk; end %#ok<AGROW>
    end
    D = assemble_dataset(runs, opts);
end
