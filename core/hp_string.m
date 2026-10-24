function s = hp_string(hp)
%HP_STRING  One-line text of a hyperparameter struct, e.g. 'C=10, kernelScale=5'.
%
%   Week 9 deliverable, CS 5395 Independent Study (Xiaohui San).

    f = fieldnames(hp);
    parts = cell(1, numel(f));
    for i = 1:numel(f)
        v = hp.(f{i});
        if ischar(v)
            parts{i} = sprintf('%s=%s', f{i}, v);
        elseif isstruct(v)
            parts{i} = sprintf('%s=(%s)', f{i}, hp_string(v));       % e.g. baseHp of a bag
        elseif isscalar(v)
            parts{i} = sprintf('%s=%g', f{i}, v);
        else
            parts{i} = sprintf('%s=[%s]', f{i}, strtrim(sprintf('%g ', v)));
        end
    end
    s = strjoin(parts, ', ');
end
