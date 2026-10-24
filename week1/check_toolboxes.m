function report = check_toolboxes()
%CHECK_TOOLBOXES  Week 1 deliverable: verify the MATLAB environment for the
%   "Ensemble Learning-Based Fault Diagnosis System for WSNs" project.
%
%   report = check_toolboxes() prints the MATLAB release, checks that each
%   required and optional toolbox is installed and licensed, runs a short
%   functional smoke test for the key functions used later in the project,
%   and saves the result to toolbox_report.txt in the current folder.
%
%   Required : Statistics and Machine Learning Toolbox (fitcensemble,
%              fitcsvm, fitctree, fitcknn, TreeBagger, fitcnet, perfcurve)
%              fitcnet (shallow neural network) needs MATLAB R2021a or later.
%   Optional : Signal Processing Toolbox   (frequency-domain features, Wk 8)
%              Wavelet Toolbox             (wavelet features, Wk 8)
%              Deep Learning Toolbox       (neural network model, Wk 10)
%              Parallel Computing Toolbox  (faster large experiments, Wk 16)
%
%   Author : Xiaohui San            Course : CS 5395 Independent Study
%   Advisor: Dr. Hongchi Shi        Week   : 1

    % name shown by ver()             license feature name        needed?
    tb = {
      'Statistics and Machine Learning Toolbox', 'Statistics_Toolbox',   true
      'Signal Processing Toolbox',               'Signal_Toolbox',       false
      'Wavelet Toolbox',                         'Wavelet_Toolbox',      false
      'Deep Learning Toolbox',                   'Neural_Network_Toolbox', false
      'Parallel Computing Toolbox',              'Distrib_Computing_Toolbox', false
    };

    v = ver();
    installedNames = {v.Name};

    lines = {};
    lines{end+1} = sprintf('Environment check  -  %s', datestr(now));
    lines{end+1} = sprintf('MATLAB version     : %s', version());
    lines{end+1} = sprintf('Computer           : %s', computer());
    lines{end+1} = '';
    lines{end+1} = sprintf('%-42s %-10s %-10s %s', 'Toolbox', 'Installed', 'Licensed', 'Needed for');

    report = struct('name', {}, 'installed', {}, 'licensed', {}, 'required', {});
    missingRequired = {};
    for i = 1:size(tb, 1)
        name = tb{i,1};
        isInst = any(strcmp(installedNames, name));
        isLic = false;
        try
            isLic = logical(license('test', tb{i,2}));
        catch
            % license() is not available in every environment (e.g. Octave)
        end
        if tb{i,3}, need = 'REQUIRED'; else, need = 'optional'; end
        lines{end+1} = sprintf('%-42s %-10s %-10s %s', name, yn(isInst), yn(isLic), need); %#ok<AGROW>
        report(i) = struct('name', name, 'installed', isInst, 'licensed', isLic, 'required', tb{i,3}); %#ok<AGROW>
        if tb{i,3} && ~(isInst && isLic)
            missingRequired{end+1} = name; %#ok<AGROW>
        end
    end

    % ---- functional smoke test of the functions the project relies on ----
    lines{end+1} = '';
    lines{end+1} = 'Functional smoke test (Fisher iris data, 5-fold CV):';
    tests = {
      'fitctree      (decision tree)', @(X,y) fitctree(X, y)
      'fitcknn       (k-NN)',          @(X,y) fitcknn(X, y, 'NumNeighbors', 5)
      'fitcecoc/SVM  (multiclass SVM)',@(X,y) fitcecoc(X, y)
      'TreeBagger    (bagging / RF)',  @(X,y) TreeBagger(50, X, y, 'OOBPrediction', 'on')
      'fitcensemble  (AdaBoostM2)',    @(X,y) fitcensemble(X, y, 'Method', 'AdaBoostM2', ...
                                          'NumLearningCycles', 50, 'LearnRate', 0.5, ...
                                          'Learners', templateTree('MaxNumSplits', 1))
      'fitcensemble  (RUSBoost)',      @(X,y) fitcensemble(X, y, 'Method', 'RUSBoost', ...
                                          'NumLearningCycles', 100, 'LearnRate', 0.5, ...
                                          'Learners', templateTree('MaxNumSplits', 3))
      'fitcnet       (neural network)',@(X,y) fitcnet(X, y, 'LayerSizes', 10, ...
                                          'Standardize', true)
    };
    try
        S = load('fisheriris');
        X = S.meas; y = S.species;
        rng(1);   % fixed seed: same CV folds and same results on every run
        for i = 1:size(tests, 1)
            try
                mdl = tests{i,2}(X, y);
                if isa(mdl, 'TreeBagger')
                    err = oobError(mdl); err = err(end);
                else
                    err = kfoldLoss(crossval(mdl, 'KFold', 5));
                end
                lines{end+1} = sprintf('  PASS  %-32s CV error = %.3f', tests{i,1}, err); %#ok<AGROW>
            catch ME
                lines{end+1} = sprintf('  FAIL  %-32s %s', tests{i,1}, ME.message); %#ok<AGROW>
            end
        end
    catch ME
        lines{end+1} = ['  Skipped: ' ME.message];
    end

    lines{end+1} = '';
    if isempty(missingRequired)
        lines{end+1} = 'RESULT: environment is ready for the project.';
    else
        lines{end+1} = ['RESULT: missing required toolbox: ' strjoin(missingRequired, ', ')];
        lines{end+1} = '        Install it from Home > Add-Ons, or use the university campus license.';
    end

    txt = strjoin(lines, newline);
    fprintf('%s\n', txt);
    fid = fopen('toolbox_report.txt', 'w');
    if fid > 0
        fprintf(fid, '%s\n', txt);
        fclose(fid);
    end
end

function s = yn(tf)
    if tf, s = 'yes'; else, s = 'no'; end
end
