function net = deploy_nodes(N, areaW, areaH, sinkPos, mode, E0)
%DEPLOY_NODES  Place N sensor nodes in an areaW x areaH field.
%
%   net = deploy_nodes(N, areaW, areaH, sinkPos, mode, E0)
%     N       number of sensor nodes (e.g. 50-200)
%     areaW,H field size in metres
%     sinkPos [x y] of the base station (sink)
%     mode    'random' (uniform) or 'grid' (regular grid with small jitter)
%     E0      initial battery energy per node in joules
%
%   Returns struct net with fields
%     x, y        N x 1 node coordinates [m]
%     sink        1 x 2 sink position
%     dSink       N x 1 distance from each node to the sink
%     D           N x N pairwise distance matrix
%     E           N x 1 residual energy [J]
%     id          N x 1 node identifiers
%
%   CS 5395 Independent Study (Xiaohui San).

    if nargin < 5 || isempty(mode), mode = 'random'; end
    if nargin < 6 || isempty(E0),  E0 = 0.5;        end

    switch lower(mode)
        case 'random'
            x = areaW * rand(N, 1);
            y = areaH * rand(N, 1);
        case 'grid'
            nc = ceil(sqrt(N * areaW / areaH));
            nr = ceil(N / nc);
            [gx, gy] = meshgrid(((1:nc) - 0.5) * areaW / nc, ((1:nr) - 0.5) * areaH / nr);
            x = gx(:); y = gy(:);
            x = x(1:N); y = y(1:N);
            jitter = 0.1 * min(areaW / nc, areaH / nr);
            x = min(max(x + jitter * (2 * rand(N, 1) - 1), 0), areaW);
            y = min(max(y + jitter * (2 * rand(N, 1) - 1), 0), areaH);
        otherwise
            error('deploy_nodes:mode', 'mode must be ''random'' or ''grid''.');
    end

    net.id    = (1:N)';
    net.x     = x;
    net.y     = y;
    net.sink  = sinkPos(:)';
    net.dSink = hypot(x - sinkPos(1), y - sinkPos(2));
    dx = x - x';                         % implicit expansion (R2016b+ / Octave)
    dy = y - y';
    net.D     = hypot(dx, dy);
    net.E     = E0 * ones(N, 1);
    net.E0    = E0;
    net.area  = [areaW areaH];
end
