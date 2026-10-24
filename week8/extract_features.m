function [X, names, groups] = extract_features(R, s0, nodes, winLen, ctx)
%EXTRACT_FEATURES  Feature vector of each one-hour window of one run.
%
%   [X, names, groups] = extract_features(R, s0, nodes, winLen, ctx)
%     R       one run from simulate_run
%     s0      first sample of each window (column vector)
%     nodes   node of each window
%     winLen  samples per window (default 12 = 1 h)
%     ctx     samples of context ending with the window, used by the trend
%             features (default 72 = 6 h)
%     X       numel(s0) x 29 feature matrix (no NaN; see below)
%     names   1 x 29 cell array of feature names
%     groups  1 x 29 cell array: 'time', 'freq', 'wavelet', 'spatial',
%             'kalman' or 'state'
%
%   Only information the sink has is used: the received readings, the
%   battery voltage carried in the packets, the readings of the node's
%   radio neighbours and the routing tree.
%
%   Time domain (received readings of the window, gaps filled by linear
%   interpolation):
%     mean, std, range, skewAbs (|skewness|; faults of either sign look
%     alike), kurtosis, slope (deg C / h),
%     maxJump (largest |first difference|), diffStd (robust std of the
%     first differences), jumpRatio (maxJump / median |difference|),
%     flatFrac (share of differences below 0.001 deg C), acf1 (lag-1
%     autocorrelation after removing the linear trend)
%   Frequency domain (FFT of the detrended window):
%     hfRatio (power above a quarter of the sampling rate / total power),
%     specCentroid (power-weighted mean frequency, cycles per sample)
%   Wavelet (Haar transform, two levels):
%     d1Rms, d2Rms (RMS of the level-1 and level-2 detail coefficients),
%     d1Max (largest |level-1 coefficient|)
%   Spatial (residual = reading minus the median of the neighbours, see
%   neighbour_residual). Faults have a random sign, so the spatial features
%   are magnitudes or are signed relative to the residual:
%     resAbs (|median residual|), resStd, ratioAbs (|median(reading) /
%     median(neighbours) - 1|), nbCorr (correlation with the neighbour
%     median), resGrowth (rate at which |residual| grows over the context,
%     deg C / h: the residual trend times the sign of the residual),
%     levelAbs (|slope of the residual against the neighbour level| over
%     the context; about |gain - 1| for a gain fault), resChange (|median
%     residual| of this window minus that of the first hour of the context)
%   Kalman filter (kalman_innovations):
%     kalMax, kalMean (largest and mean |normalised innovation|)
%   Node state:
%     ownMiss (share of samples missing that were not lost at a relay or
%     because the node had no route), relayMiss (share lost at a relay or
%     because of no route), vMin (lowest voltage reported in the window,
%     else the last voltage ever reported, else 3.0 V), vDrop (voltage
%     reported 6 h earlier minus vMin)
%
%   A feature that cannot be computed (too few readings) is set to 0; the
%   missing-data features tell the classifier when this happened.
%
%   Week 8 deliverable, CS 5395 Independent Study (Xiaohui San).

    if nargin < 4 || isempty(winLen), winLen = 12; end
    if nargin < 5 || isempty(ctx), ctx = 72; end
    names = {'mean', 'std', 'range', 'skewAbs', 'kurtosis', 'slope', 'maxJump', ...
             'diffStd', 'jumpRatio', 'flatFrac', 'acf1', ...
             'hfRatio', 'specCentroid', ...
             'd1Rms', 'd2Rms', 'd1Max', ...
             'resAbs', 'resStd', 'ratioAbs', 'nbCorr', 'resGrowth', 'levelAbs', 'resChange', ...
             'kalMax', 'kalMean', ...
             'ownMiss', 'relayMiss', 'vMin', 'vDrop'};
    groups = [repmat({'time'}, 1, 11), repmat({'freq'}, 1, 2), repmat({'wavelet'}, 1, 3), ...
              repmat({'spatial'}, 1, 7), repmat({'kalman'}, 1, 2), repmat({'state'}, 1, 4)];
    P = numel(names);

    % ---- per-run quantities, computed once ----
    [res, nbMed] = neighbour_residual(R);
    cutOff = R.parent < 0 & R.energy > 0;             % alive, but no route
    notOwn = R.relayLost | cutOff;
    vRep = R.voltage;  vRep(isnan(R.reading)) = NaN;  % voltage the sink received
    Z = nan(size(R.reading));
    for i = unique(nodes(:))'
        Z(:, i) = abs(kalman_innovations(R.reading(:, i)));
    end
    dtH = R.time(2) - R.time(1);
    tt = (0:winLen - 1)' * dtH;                       % time inside the window [h]

    X = zeros(numel(s0), P);
    for w = 1:numel(s0)
        i = nodes(w);
        s = (s0(w):s0(w) + winLen - 1)';
        c = (max(1, s(end) - ctx + 1):s(end))';
        x = R.reading(s, i);
        got = ~isnan(x);
        f = zeros(1, P);

        % ---- node state ----
        f(26) = mean(~got & ~notOwn(s, i));
        f(27) = mean(~got & notOwn(s, i));
        vw = vRep(s, i);  vw = vw(~isnan(vw));
        vh = vRep(1:s(end), i);  vh = vh(~isnan(vh));
        if ~isempty(vw), f(28) = min(vw); elseif ~isempty(vh), f(28) = vh(end); else, f(28) = 3.0; end
        vOld = vRep(1:max(1, s(end) - ctx), i);  vOld = vOld(~isnan(vOld));
        if ~isempty(vOld), f(29) = vOld(end) - f(28); end

        if sum(got) >= 3
            % fill gaps for the shape features
            xf = interp1(find(got), x(got), (1:winLen)', 'linear');
            xf = fillEnds(xf);
            dx = diff(x(got));                         % differences of received readings
            % ---- time domain ----
            mu = mean(xf);  sd = std(xf);
            f(1) = mu;  f(2) = sd;  f(3) = max(xf) - min(xf);
            if sd > 1e-9
                zc = (xf - mu) / std(xf, 1);
                f(4) = abs(mean(zc .^ 3));  f(5) = mean(zc .^ 4);
            else
                f(5) = 3;                              % kurtosis of a constant: use the Gaussian value
            end
            p = [tt, ones(winLen, 1)] \ xf;  f(6) = p(1);
            f(7) = max(abs(dx));
            f(8) = 1.4826 * median(abs(dx - median(dx)));
            f(9) = f(7) / (median(abs(dx)) + 0.01);
            f(10) = mean(abs(dx) < 1e-3);
            e = xf - [tt, ones(winLen, 1)] * p;        % detrended
            if sum(e .^ 2) > 1e-12
                f(11) = sum(e(1:end - 1) .* e(2:end)) / sum(e .^ 2);
            end
            % ---- frequency domain ----
            Pw = abs(fft(e)) .^ 2;
            Pw = Pw(2:floor(winLen / 2) + 1);           % positive frequencies, no DC
            fr = (1:numel(Pw))' / winLen;               % cycles per sample
            if sum(Pw) > 1e-12
                f(12) = sum(Pw(fr > 0.25)) / sum(Pw);
                f(13) = sum(fr .* Pw) / sum(Pw);
            end
            % ---- Haar wavelet, two levels ----
            n2 = 2 * floor(winLen / 2);
            a1 = (xf(1:2:n2) + xf(2:2:n2)) / sqrt(2);
            d1 = (xf(1:2:n2) - xf(2:2:n2)) / sqrt(2);
            m2 = 2 * floor(numel(a1) / 2);
            d2 = (a1(1:2:m2) - a1(2:2:m2)) / sqrt(2);
            f(14) = sqrt(mean(d1 .^ 2));  f(15) = sqrt(mean(d2 .^ 2));  f(16) = max(abs(d1));
            % ---- Kalman ----
            zw = Z(s, i);  zw = zw(~isnan(zw));
            if ~isempty(zw), f(24) = max(zw);  f(25) = mean(zw); end
        end

        % ---- spatial ----
        rw = res(s, i);  ok = ~isnan(rw);
        hasRes = sum(ok) >= 2;
        medRes = 0;
        if hasRes
            medRes = median(rw(ok));
            f(17) = abs(medRes);  f(18) = std(rw(ok));
            lw = nbMed(s, i);
            f(19) = abs(median(x(ok)) / max(abs(median(lw(ok))), 1) - 1);
            if std(x(ok)) > 1e-9 && std(lw(ok)) > 1e-9
                cc = corrcoef(x(ok), lw(ok));  f(20) = cc(1, 2);
            end
        end
        rc = res(c, i);  lc = nbMed(c, i);  tc = R.time(c);
        ok = ~isnan(rc) & ~isnan(lc);
        if hasRes && sum(ok) >= 6
            X3 = [ones(sum(ok), 1), lc(ok) - mean(lc(ok)), tc(ok) - mean(tc(ok))];
            if rank(X3) == 3
                b = X3 \ rc(ok);
                f(21) = sign(medRes) * b(3);           % growth of |residual|
                f(22) = abs(b(2));
            end
            r0 = rc(1:min(winLen, numel(rc)));  r0 = r0(~isnan(r0));
            if ~isempty(r0), f(23) = abs(medRes) - abs(median(r0)); end
        end
        X(w, :) = f;
    end
    X(~isfinite(X)) = 0;
end

function x = fillEnds(x)
%FILLENDS  Replace leading and trailing NaN by the nearest value.
    k = find(~isnan(x));
    x(1:k(1) - 1) = x(k(1));
    x(k(end) + 1:end) = x(k(end));
end
