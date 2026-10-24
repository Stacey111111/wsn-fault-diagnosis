function P = radio_params()
%RADIO_PARAMS  Default parameters of the first-order radio energy model.
%   Values follow Heinzelman, Chandrakasan & Balakrishnan (2002), the
%   standard setting used in LEACH and most clustering studies.

    P.E_elec = 50e-9;      % J/bit   electronics energy (tx and rx circuitry)
    P.eps_fs = 10e-12;     % J/bit/m^2  free-space amplifier
    P.eps_mp = 0.0013e-12; % J/bit/m^4  multipath amplifier
    P.E_DA   = 5e-9;       % J/bit   data aggregation (per bit per signal)
    P.d0     = sqrt(P.eps_fs / P.eps_mp);   % crossover distance, ~87.7 m
end
