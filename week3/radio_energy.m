function E = radio_energy(mode, k, d, P)
%RADIO_ENERGY  First-order radio energy model (Heinzelman et al., 2002).
%
%   E = radio_energy('tx', k, d)   energy [J] to transmit k bits over d metres
%   E = radio_energy('rx', k)      energy [J] to receive k bits
%   E = radio_energy('da', k)      energy [J] to aggregate k bits of data
%   E = radio_energy(..., P)       use parameters in struct P (see below)
%
%   Transmit energy uses the free-space (d^2) model below the crossover
%   distance d0 and the two-ray / multipath (d^4) model at or above it:
%
%       E_tx(k,d) = k*E_elec + k*eps_fs*d^2   if d <  d0
%       E_tx(k,d) = k*E_elec + k*eps_mp*d^4   if d >= d0,   d0 = sqrt(eps_fs/eps_mp)
%       E_rx(k)   = k*E_elec
%
%   Default parameters (P = radio_params()):
%       E_elec = 50 nJ/bit, eps_fs = 10 pJ/bit/m^2,
%       eps_mp = 0.0013 pJ/bit/m^4, E_DA = 5 nJ/bit  ->  d0 ~ 87.7 m
%
%   d may be a vector or matrix; E has the same size.
%
%   CS 5395 Independent Study (Xiaohui San).

    if nargin < 4 || isempty(P)
        P = radio_params();
    end
    switch lower(mode)
        case 'tx'
            d0 = sqrt(P.eps_fs / P.eps_mp);
            amp = P.eps_fs .* d.^2;
            far = d >= d0;
            amp(far) = P.eps_mp .* d(far).^4;
            E = k .* P.E_elec + k .* amp;
        case 'rx'
            E = k .* P.E_elec;
        case 'da'
            E = k .* P.E_DA;
        otherwise
            error('radio_energy:mode', 'mode must be ''tx'', ''rx'' or ''da''.');
    end
end
