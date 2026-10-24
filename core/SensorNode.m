classdef SensorNode < handle
%SENSORNODE  One battery-powered wireless sensor node.
%
%   node = SensorNode(id, x, y)
%   node = SensorNode(id, x, y, opts)   opts fields override the defaults:
%       E0         initial battery energy [J]              (0.5)
%       noiseStd   std. of normal measurement noise [deg C] (0.12)
%       noiseAR    AR(1) coefficient of the noise           (0.8)
%       range      sensor measurement range [deg C]         ([-40 85])
%       vMax,vMin  battery voltage when full / empty [V]    (3.0, 2.1)
%
%   A node senses the physical field, adds realistic correlated noise,
%   saturates at its measurement range, spends energy on sensing and radio
%   traffic, and reports its battery voltage. Faults are not part of the
%   node itself: FaultInjector.apply() modifies the node's clean reading
%   (Week 6), so the same node model is used with and without faults.
%
%   Key methods
%     r = sense(node, trueValue)     clean reading (noise + range limit)
%     spend(node, E)                 subtract E joules; node dies at 0 J
%     v = voltage(node)              battery voltage from remaining energy
%     d = distanceTo(node, xy)       Euclidean distance to a point
%
%   Week 5 deliverable, CS 5395 Independent Study (Xiaohui San).

    properties
        id            % node identifier (1..N)
        x             % position [m]
        y
        E0            % initial energy [J]
        energy        % remaining energy [J]
        noiseStd      % measurement noise std [deg C]
        noiseAR       % AR(1) coefficient of the measurement noise
        range         % [min max] measurement range of the sensor
        vMax          % battery voltage when full [V]
        vMin          % battery voltage when empty (node browns out) [V]
        isCH = false  % true while the node is a cluster head
        lastCHepoch = -Inf  % last LEACH epoch in which it was a cluster head
        clusterHead = 0     % id of its cluster head (0 = sends to sink)
        state = struct()    % scratch state used by faults (e.g. stuck value)
        txCount = 0   % packets transmitted
        rxCount = 0   % packets received (as cluster head)
    end

    properties (Access = private)
        noiseState = 0      % current value of the AR(1) noise process
    end

    methods
        function obj = SensorNode(id, x, y, opts)
            if nargin < 4, opts = struct(); end
            d = struct('E0', 0.5, 'noiseStd', 0.12, 'noiseAR', 0.8, ...
                       'range', [-40 85], 'vMax', 3.0, 'vMin', 2.1);
            f = fieldnames(opts);
            for i = 1:numel(f), d.(f{i}) = opts.(f{i}); end
            obj.id = id;  obj.x = x;  obj.y = y;
            obj.E0 = d.E0;  obj.energy = d.E0;
            obj.noiseStd = d.noiseStd;  obj.noiseAR = d.noiseAR;
            obj.range = d.range;  obj.vMax = d.vMax;  obj.vMin = d.vMin;
        end

        function r = sense(obj, trueValue)
            %SENSE  Clean reading = true value + AR(1) noise, clipped to range.
            innov = obj.noiseStd * sqrt(1 - obj.noiseAR^2) * randn();
            obj.noiseState = obj.noiseAR * obj.noiseState + innov;
            r = min(max(trueValue + obj.noiseState, obj.range(1)), obj.range(2));
        end

        function spend(obj, E)
            %SPEND  Consume E joules of battery energy.
            obj.energy = max(obj.energy - E, 0);
        end

        function tf = isAlive(obj)
            tf = obj.energy > 0;
        end

        function v = voltage(obj)
            %VOLTAGE  Battery voltage: flat for most of the discharge and
            %   falling quickly near the end, like a real alkaline/Li cell.
            frac = max(obj.energy, 0) / obj.E0;          % state of charge 0..1
            v = obj.vMin + (obj.vMax - obj.vMin) * frac.^0.25;
        end

        function d = distanceTo(obj, xy)
            d = hypot(obj.x - xy(1), obj.y - xy(2));
        end

        function reset(obj)
            %RESET  Restore full battery and clear all run-time state.
            obj.energy = obj.E0;  obj.isCH = false;  obj.lastCHepoch = -Inf;
            obj.clusterHead = 0;  obj.state = struct();
            obj.txCount = 0;  obj.rxCount = 0;  obj.noiseState = 0;
        end
    end
end
