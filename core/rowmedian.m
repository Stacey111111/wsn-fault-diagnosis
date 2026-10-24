function m = rowmedian(A)
%ROWMEDIAN  Median of each row of A, ignoring NaN (NaN if a row is all NaN).
%   Works in MATLAB and GNU Octave (Octave's median has no 'omitnan').
    m = nan(size(A, 1), 1);
    S = sort(A, 2);                        % NaNs sort to the end
    n = sum(~isnan(A), 2);
    for i = find(n > 0)'
        k = n(i);
        if mod(k, 2) == 1
            m(i) = S(i, (k + 1) / 2);
        else
            m(i) = (S(i, k / 2) + S(i, k / 2 + 1)) / 2;
        end
    end
end
