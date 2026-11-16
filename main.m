clear; clc; close all;

%% Config & Parameters
cfg.quick = false;
cfg.seed = 2026;

cfg.rhos = [0, 0.5, 0.9, 0.99];
cfg.Ls = [8, 16, 32, 64, 128];
cfg.N = 20000;
cfg.b = 20;      % block size for MSE averaging
cfg.W = 10;      % consecutive blocks required to hit target
cfg.targets = [1.10, 1.25];
cfg.snr_dB = 20;
cfg.Rdev = 100;
cfg.Rtest = 200;
cfg.B = 1000;
cfg.delta = 0.01;
cfg.eps = 1e-6;

% Hyperparameter grids
cfg.grid.lms = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];   % mu = c/L
cfg.grid.nlms = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8];   % mu = c
cfg.grid.rls = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];     % lambda = 1 - c/L
cfg.hardEdge.lms = [];
cfg.hardEdge.nlms = [];
cfg.hardEdge.rls = 1;

% Timing params
cfg.tLs = [4, 8, 16, 32, 64, 128];
cfg.tN = 20000;
cfg.tReps = 15;

% Theory check params
cfg.thL = 16;
cfg.thMus = [0.0025, 0.005, 0.01, 0.015, 0.02];
cfg.thN = 8000;
cfg.thWin = 2000;
cfg.thR = 200;

% Tracking params
cfg.trL = 16;
cfg.sq2s = logspace(-9, -5, 5);
cfg.trSv2 = 1e-2;
cfg.trN = 4000;
cfg.trWin = 1000;
cfg.trRdev = 30;
cfg.trRtest = 100;
cfg.trGrid.lms = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05];
cfg.trGrid.nlms = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
cfg.trGrid.rls = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 1];
cfg.trHardEdge.lms = [];
cfg.trHardEdge.nlms = [];
cfg.trHardEdge.rls = 9;

if cfg.quick
    cfg.rhos = [0, 0.9];
    cfg.Ls = [8, 16];
    cfg.N = 4000;
    cfg.Rdev = 6;
    cfg.Rtest = 30;
    cfg.B = 200;
    cfg.tLs = [8, 16];
    cfg.tN = 4000;
    cfg.tReps = 5;
    cfg.thR = 20;
    cfg.trRdev = 5;
    cfg.trRtest = 10;
    cfg.sq2s = [1e-8, 1e-6];
end

algs = {'lms', 'nlms', 'rls'};
names = {'LMS', 'NLMS', 'RLS'};
col = [0, 0.447, 0.741; 0.850, 0.325, 0.098; 0.466, 0.674, 0.188];
sty = {'-', '--', ':'};

if ~exist('figs', 'dir'), mkdir('figs'); end
if ~exist('tables', 'dir'), mkdir('tables'); end
if ~exist('results', 'dir'), mkdir('results'); end

%% System Info
env = struct();
env.version = version;
env.computer = computer;
env.date = datestr(now, 'yyyy-mm-dd HH:MM');
env.threads = NaN;
try
    env.threads = maxNumCompThreads;
catch
end
env.cpu = cpuName();
env.precision = 'double';
fprintf('Environment: %s | %s | %s | threads=%g\n', env.version, env.computer, env.cpu, env.threads);

%% Part T: Runtime Benchmarks
fprintf('\n[T] Timing ...\n');
allL = unique([cfg.tLs, cfg.Ls]);
numL = numel(allL);
tMed = zeros(numL, 3);
tQ1 = zeros(numL, 3);
tQ3 = zeros(numL, 3);
tMin = zeros(numL, 3);
tInit = zeros(numL, 3);

for il = 1:numL
    L = allL(il);
    rng(cfg.seed + 7000 + L, 'twister');
    [x, d] = genData(cfg.tN, L, 0, cfg.snr_dB);
    
    for ia = 1:3
        p = midParams(algs{ia}, L, cfg);
        timeRun(algs{ia}, x, d, L, p); % warmup
        
        ts = zeros(cfg.tReps, 1);
        for rep = 1:cfg.tReps
            t0 = tic;
            timeRun(algs{ia}, x, d, L, p);
            ts(rep) = toc(t0);
        end
        
        us = 1e6 * ts / cfg.tN;
        tMed(il, ia) = median(us);
        tMin(il, ia) = min(us);
        tQ1(il, ia) = qtile(us, 0.25);
        tQ3(il, ia) = qtile(us, 0.75);
        
        ti = zeros(200, 1);
        for rep = 1:200
            t0 = tic;
            initOnly(algs{ia}, L, p);
            ti(rep) = toc(t0);
        end
        tInit(il, ia) = 1e6 * median(ti);
    end
end
tPer = @(ia, L) tMed(allL == L, ia);

T = {'L,LMS (us),NLMS (us),RLS (us)'};
for L = cfg.tLs
    il = find(allL == L);
    row = sprintf('%d', L);
    for ia = 1:3
        row = [row, sprintf(',%.2f (%.2f-%.2f)', tMed(il, ia), tQ1(il, ia), tQ3(il, ia))];
    end
    T{end+1} = row;
end
writeLines('tables/tab_runtime.csv', T);

%% Part A: Main Experiment
fprintf('\n[A] Main experiment ...\n');
nr = numel(cfg.rhos);
nL = numel(cfg.Ls);
nT = numel(cfg.targets);
nb = cfg.N / cfg.b;
res = struct();

for ir = 1:nr
    for iL = 1:nL
        rho = cfg.rhos(ir);
        L = cfg.Ls(iL);
        cellId = 100 * ir + iL;
        fprintf('  rho = %.2f, L = %3d ... ', rho, L);
        tc = tic;

        % Dev phase (tuning)
        rng(cfg.seed + 1000 + cellId, 'twister');
        dev = struct();
        for ia = 1:3
            dev.(algs{ia}) = zeros(numel(cfg.grid.(algs{ia})), nb);
        end
        
        for r = 1:cfg.Rdev
            [x, d, ~, sv2] = genData(cfg.N, L, rho, cfg.snr_dB);
            for ia = 1:3
                g = cfg.grid.(algs{ia});
                for ig = 1:numel(g)
                    p = gridParams(algs{ia}, g(ig), L, cfg);
                    eb = runFilter(algs{ia}, x, d, L, p, [], cfg.b);
                    dev.(algs{ia})(ig, :) = dev.(algs{ia})(ig, :) + eb.' / sv2;
                end
            end
        end
        
        pick = zeros(3, nT);
        reachedDev = false(3, nT);
        for ia = 1:3
            C = cleanCurve(dev.(algs{ia}) / cfg.Rdev);
            for it = 1:nT
                [pick(ia, it), reachedDev(ia, it)] = pickSetting(C, cfg.targets(it), cfg.W);
            end
        end

        % Test phase
        rng(cfg.seed + 2000 + cellId, 'twister');
        E = cell(3, nT);
        M = cell(3, nT);
        for ia = 1:3
            for it = 1:nT
                E{ia, it} = zeros(cfg.Rtest, nb, 'single');
                M{ia, it} = E{ia, it};
            end
        end
        
        for r = 1:cfg.Rtest
            [x, d, wo, sv2] = genData(cfg.N, L, rho, cfg.snr_dB);
            for ia = 1:3
                for it = 1:nT
                    j = find(pick(ia, 1:it-1) == pick(ia, it), 1);
                    if ~isempty(j)
                        E{ia, it}(r, :) = E{ia, j}(r, :);
                        M{ia, it}(r, :) = M{ia, j}(r, :);
                        continue;
                    end
                    p = gridParams(algs{ia}, cfg.grid.(algs{ia})(pick(ia, it)), L, cfg);
                    [eb, mb] = runFilter(algs{ia}, x, d, L, p, wo, cfg.b);
                    E{ia, it}(r, :) = single(eb.' / sv2);
                    M{ia, it}(r, :) = single(mb.');
                end
            end
        end

        % Bootstrap evaluation
        rng(cfg.seed + 3000 + cellId, 'twister');
        idx = randi(cfg.Rtest, cfg.Rtest, cfg.B);
        Cnt = zeros(cfg.B, cfg.Rtest);
        for bb = 1:cfg.B
            Cnt(bb, :) = accumarray(idx(:, bb), 1, [cfg.Rtest, 1]).';
        end

        out = struct();
        for it = 1:nT
            Tg = cfg.targets(it);
            Nb = zeros(cfg.B, 3);
            N0 = zeros(1, 3);
            
            for ia = 1:3
                Ei = cleanCurve(double(E{ia, it}));
                curve0 = mean(Ei, 1);
                N0(ia) = firstSustained(curve0, Tg, cfg.W) * cfg.b;
                
                curvesB = (Cnt * Ei) / cfg.Rtest;
                Nb(:, ia) = firstSustained(curvesB, Tg, cfg.W) * cfg.b;
                
                if it == 1
                    out.curve{ia} = curve0;
                    out.band{ia} = [qtile(curvesB, 0.025); qtile(curvesB, 0.975)];
                    Mi = double(M{ia, it});
                    out.msd{ia} = mean(Mi, 1);
                    out.finalNMSE(ia) = mean(curve0(end - cfg.W + 1:end));
                    out.finalMSD(ia) = mean(out.msd{ia}(end - cfg.W + 1:end));
                end
            end
            
            tp = [tPer(1, L), tPer(2, L), tPer(3, L)];
            C0 = N0 .* tp;
            Cb = Nb .* repmat(tp, cfg.B, 1);
            
            [~, w] = min(Cb, [], 2);
            w(all(isinf(Cb), 2)) = 0;
            winFrac = [mean(w == 1), mean(w == 2), mean(w == 3)];
            
            out.t(it).N0 = N0;
            out.t(it).Nci = [qtile(Nb, 0.025); qtile(Nb, 0.975)];
            out.t(it).C0 = C0;
            out.t(it).Cci = [qtile(Cb, 0.025); qtile(Cb, 0.975)];
            out.t(it).nonCross = mean(isinf(Nb), 1);
            out.t(it).winFrac = winFrac;
            out.t(it).winner = winnerLabel(winFrac, names);
            out.t(it).winCode = winnerCode(winFrac);
            out.t(it).tp = tp;
            
            ops = opsPerUpdate(L);
            Ob = Nb .* repmat(ops, cfg.B, 1);
            [~, w] = min(Ob, [], 2);
            w(all(isinf(Ob), 2)) = 0;
            fo = [mean(w == 1), mean(w == 2), mean(w == 3)];
            out.t(it).winnerOps = winnerLabel(fo, names);
            out.t(it).winCodeOps = winnerCode(fo);
        end

        % Timing validation check
        rng(cfg.seed + 8000 + cellId, 'twister');
        out.valRatio = nan(1, 3);
        for ia = 1:3
            N0 = out.t(1).N0(ia);
            if isfinite(N0)
                [x, d] = genData(N0, L, rho, cfg.snr_dB);
                p = gridParams(algs{ia}, cfg.grid.(algs{ia})(pick(ia, 1)), L, cfg);
                timeRun(algs{ia}, x, d, L, p);
                ts = zeros(5, 1);
                for rep = 1:5
                    t0 = tic;
                    initOnly(algs{ia}, L, p);
                    timeRun(algs{ia}, x, d, L, p);
                    ts(rep) = toc(t0);
                end
                out.valRatio(ia) = (1e6 * median(ts)) / (N0 * tPer(ia, L));
            end
        end

        out.pick = pick;
        out.reachedDev = reachedDev;
        out.sv2note = 'sigma_v^2 set per trial for 20 dB output SNR';
        res(ir, iL).out = out;
        fprintf('%.0f s  (T=%.2f winner: %s)\n', toc(tc), cfg.targets(1), out.t(1).winner);
    end
end

% Export main tables as CSV
for it = 1:nT
    T = {'rho,L,N_LMS,N_NLMS,N_RLS,C_LMS (ms) [95% CI],C_NLMS (ms) [95% CI],C_RLS (ms) [95% CI],Lowest cost'};
    for ir = 1:nr
        for iL = 1:nL
            o = res(ir, iL).out.t(it);
            row = sprintf('%.2f,%d', cfg.rhos(ir), cfg.Ls(iL));
            for ia = 1:3
                row = [row, ',', fmtN(o.N0(ia))];
            end
            for ia = 1:3
                row = [row, ',"', fmtCms(o.C0(ia), o.Cci(:, ia)), '"'];
            end
            T{end+1} = [row, ',"', o.winner, '"'];
        end
    end
    writeLines(sprintf('tables/tab_main_T%d.csv', round(100 * cfg.targets(it))), T);
end

T = {'rho,L,LMS (mu*L),NLMS (mu),RLS ((1-lambda)*L)'};
for ir = 1:nr
    for iL = 1:nL
        o = res(ir, iL).out;
        row = sprintf('%.2f,%d', cfg.rhos(ir), cfg.Ls(iL));
        for ia = 1:3
            g = cfg.grid.(algs{ia});
            k = o.pick(ia, 1);
            s = sprintf('%g', g(k));
            if isEdge(k, numel(g), cfg.hardEdge.(algs{ia}))
                s = [s, '*'];
            end
            if ~o.reachedDev(ia, 1)
                s = [s, '+'];
            end
            row = [row, ',', s];
        end
        T{end+1} = row;
    end
end
writeLines('tables/tab_settings.csv', T);

T = {sprintf('rho,L,Measured Time (Target %.2f),Measured Time (Target %.2f),Operation Count (Target %.2f)', ...
    cfg.targets(1), cfg.targets(end), cfg.targets(1))};
for ir = 1:nr
    for iL = 1:nL
        o = res(ir, iL).out;
        T{end+1} = sprintf('%.2f,%d,"%s","%s","%s"', ...
            cfg.rhos(ir), cfg.Ls(iL), o.t(1).winner, o.t(end).winner, o.t(1).winnerOps);
    end
end
writeLines('tables/tab_robust.csv', T);

codes = zeros(nr, nL, nT);
for ir = 1:nr
    for iL = 1:nL
        for it = 1:nT
            codes(ir, iL, it) = res(ir, iL).out.t(it).winCode;
        end
    end
end

%% Part B: LMS Theory Check
% White Gaussian input: J_inf = sv2 / (1 - L*mu*sx2/(2 - mu*sx2))
fprintf('\n[B] LMS theory check ...\n');
L = cfg.thL;
sv2 = 10^(-cfg.snr_dB / 10);
nm = numel(cfg.thMus);
ssr = zeros(cfg.thR, nm);

rng(cfg.seed + 4000, 'twister');
for r = 1:cfg.thR
    [x, d] = genData(cfg.thN, L, 0, cfg.snr_dB);
    for im = 1:nm
        eb = runFilter('lms', x, d, L, struct('mu', cfg.thMus(im)), [], cfg.b);
        ssr(r, im) = mean(eb(end - cfg.thWin / cfg.b + 1:end));
    end
end

Jth = sv2 ./ (1 - (L * cfg.thMus) ./ (2 - cfg.thMus));
Jsim = mean(ssr, 1);

rng(cfg.seed + 4500, 'twister');
bi = randi(cfg.thR, cfg.thR, cfg.B);
Jb = zeros(cfg.B, nm);
for bb = 1:cfg.B
    Jb(bb, :) = mean(ssr(bi(:, bb), :), 1);
end

dB = @(v) 10 * log10(v);
diffCI = [qtile(dB(Jb) - dB(Jth), 0.025); qtile(dB(Jb) - dB(Jth), 0.975)];

T = {'mu,Simulated (dB) [95% CI],Theory (dB),Difference (dB) [95% CI]'};
for im = 1:nm
    T{end+1} = sprintf('%.4f,"%.3f [%.3f, %.3f]",%.3f,"%+.3f [%+.3f, %+.3f]"', ...
        cfg.thMus(im), dB(Jsim(im)), qtile(dB(Jb(:, im)), 0.025), qtile(dB(Jb(:, im)), 0.975), ...
        dB(Jth(im)), dB(Jsim(im)) - dB(Jth(im)), diffCI(1, im), diffCI(2, im));
end
writeLines('tables/tab_theory.csv', T);

%% Part C: Tracking Non-Stationary System
% Random walk model: wo(n+1) = wo(n) + q(n), q ~ N(0, sq2*I)
fprintf('\n[C] Tracking ...\n');
L = cfg.trL;
ns = numel(cfg.sq2s);
trBest = zeros(ns, 3);
trCI = zeros(2, ns, 3);
trPick = zeros(ns, 3);

for is = 1:ns
    sq2 = cfg.sq2s(is);
    rng(cfg.seed + 5000 + is, 'twister');
    
    J = struct();
    for ia = 1:3
        J.(algs{ia}) = zeros(1, numel(cfg.trGrid.(algs{ia})));
    end
    
    for r = 1:cfg.trRdev
        [x, d] = genTrack(cfg.trN, L, sq2, cfg.trSv2);
        for ia = 1:3
            g = cfg.trGrid.(algs{ia});
            for ig = 1:numel(g)
                p = trParams(algs{ia}, g(ig), cfg);
                eb = runFilter(algs{ia}, x, d, L, p, [], cfg.b);
                J.(algs{ia})(ig) = J.(algs{ia})(ig) + mean(eb(end - cfg.trWin / cfg.b + 1:end));
            end
        end
    end
    for ia = 1:3
        [~, trPick(is, ia)] = min(J.(algs{ia}));
    end
    
    rng(cfg.seed + 6000 + is, 'twister');
    ss = zeros(cfg.trRtest, 3);
    for r = 1:cfg.trRtest
        [x, d] = genTrack(cfg.trN, L, sq2, cfg.trSv2);
        for ia = 1:3
            p = trParams(algs{ia}, cfg.trGrid.(algs{ia})(trPick(is, ia)), cfg);
            eb = runFilter(algs{ia}, x, d, L, p, [], cfg.b);
            ss(r, ia) = mean(eb(end - cfg.trWin / cfg.b + 1:end));
        end
    end
    trBest(is, :) = mean(ss, 1);
    
    rng(cfg.seed + 6500 + is, 'twister');
    bi = randi(cfg.trRtest, cfg.trRtest, cfg.B);
    Sb = zeros(cfg.B, 3);
    for bb = 1:cfg.B
        Sb(bb, :) = mean(ss(bi(:, bb), :), 1);
    end
    trCI(:, is, :) = reshape([qtile(Sb, 0.025); qtile(Sb, 0.975)], 2, 1, 3);
end

T = {'sigma_q^2,LMS (dB) [mu] (95% CI),NLMS (dB) [mu] (95% CI),RLS (dB) [lambda] (95% CI)'};
for is = 1:ns
    row = sprintf('1e%d', round(log10(cfg.sq2s(is))));
    for ia = 1:3
        g = cfg.trGrid.(algs{ia});
        k = trPick(is, ia);
        s = sprintf('%g', g(k));
        if isEdge(k, numel(g), cfg.trHardEdge.(algs{ia}))
            s = [s, '*'];
        end
        row = [row, sprintf(',"%.2f [%s] (%.2f, %.2f)"', ...
            dB(trBest(is, ia)), s, dB(trCI(1, is, ia)), dB(trCI(2, is, ia)))];
    end
    T{end+1} = row;
end
writeLines('tables/tab_tracking.csv', T);

%% Plotting
fprintf('\nDrawing figures ...\n');

f = newFig([100, 100, 1100, 330]);
for ir = 1:nr
    ax = subplot(1, nr, ir);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    for ia = 1:3
        c0 = zeros(1, nL);
        lo = zeros(1, nL);
        hi = zeros(1, nL);
        for iL = 1:nL
            o = res(ir, iL).out.t(1);
            c0(iL) = o.C0(ia) / 1000;
            lo(iL) = o.Cci(1, ia) / 1000;
            hi(iL) = o.Cci(2, ia) / 1000;
        end
        ok = isfinite(c0);
        hi(~isfinite(hi)) = NaN;
        ebar(ax, cfg.Ls(ok), c0(ok), c0(ok) - lo(ok), hi(ok) - c0(ok), ['o', sty{ia}], col(ia, :));
    end
    set(ax, 'XScale', 'log', 'YScale', 'log', 'XTick', cfg.Ls);
    xlabel(ax, 'Filter length L');
    if ir == 1
        ylabel(ax, 'C_{target} (ms)');
    end
    title(ax, sprintf('\\rho = %.2f', cfg.rhos(ir)));
    if ir == 1
        legend(ax, names, 'Location', 'northwest');
    end
end
savePdf(f, 'figs/fig_cost_vs_L.pdf');

f = newFig([100, 100, 900, 330]);
cmap = [col; 0.75, 0.75, 0.75; 1, 1, 1];
lab = {'LMS', 'NLMS', 'RLS', 'tie', 'none'};
for it = 1:nT
    ax = subplot(1, nT, it);
    hold(ax, 'on');
    for ir = 1:nr
        for iL = 1:nL
            c = codes(ir, iL, it);
            rectangle('Parent', ax, 'Position', [iL - 0.5, ir - 0.5, 1, 1], ...
                'FaceColor', cmap(c, :), 'EdgeColor', 'k');
            text(iL, ir, lab{c}, 'Parent', ax, 'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    set(ax, 'XTick', 1:nL, 'XTickLabel', cfg.Ls, 'YTick', 1:nr, 'YTickLabel', cfg.rhos, ...
        'XLim', [0.5, nL + 0.5], 'YLim', [0.5, nr + 0.5]);
    xlabel(ax, 'Filter length L');
    ylabel(ax, '\rho');
    title(ax, sprintf('Lowest C_{target}, target J/\\sigma_v^2 \\leq %.2f', cfg.targets(it)));
end
savePdf(f, 'figs/fig_regime.pdf');

rRep = [find(cfg.rhos == 0, 1), find(abs(cfg.rhos - 0.9) < 1e-9, 1)];
if numel(rRep) < 2
    rRep = [1, nr];
end
lRep = [find(cfg.Ls == 16, 1), nL];
if isempty(lRep) || numel(lRep) < 2
    lRep = [1, nL];
end
nIt = (1:nb) * cfg.b;

f = newFig([100, 100, 900, 620]);
k = 0;
for ir = rRep
    for iL = lRep
        k = k + 1;
        ax = subplot(2, 2, k);
        hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        o = res(ir, iL).out;
        hl = zeros(1, 3);
        for ia = 1:3
            xc = nIt * o.t(1).tp(ia) / 1000;
            bandFill(ax, xc, 10 * log10(o.band{ia}), col(ia, :));
            hl(ia) = plot(ax, xc, 10 * log10(o.curve{ia}), sty{ia}, 'Color', col(ia, :), 'LineWidth', 1.3);
        end
        plot(ax, xlim(ax), 10 * log10(cfg.targets(1)) * [1, 1], 'k-.');
        set(ax, 'XScale', 'log');
        ylim(ax, [-0.5, 22]);
        xlabel(ax, 'Computation time (ms)');
        ylabel(ax, 'J(n)/\sigma_v^2 (dB)');
        title(ax, sprintf('\\rho = %.2f, L = %d', cfg.rhos(ir), cfg.Ls(iL)));
        if k == 1
            legend(ax, hl, names, 'Location', 'northeast');
        end
    end
end
savePdf(f, 'figs/fig_tradeoff.pdf');

f = newFig([100, 100, 900, 620]);
iL = lRep(1);
for k = 1:2
    ir = rRep(k);
    o = res(ir, iL).out;
    
    ax = subplot(2, 2, k);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    for ia = 1:3
        bandFill(ax, nIt, 10 * log10(o.band{ia}), col(ia, :));
        plot(ax, nIt, 10 * log10(o.curve{ia}), sty{ia}, 'Color', col(ia, :), 'LineWidth', 1.3);
    end
    plot(ax, [nIt(1), nIt(end)], 10 * log10(cfg.targets(1)) * [1, 1], 'k-.');
    set(ax, 'XScale', 'log');
    ylim(ax, [-0.5, 22]);
    xlabel(ax, 'Iteration n');
    ylabel(ax, 'J(n)/\sigma_v^2 (dB)');
    title(ax, sprintf('\\rho = %.2f, L = %d', cfg.rhos(ir), cfg.Ls(iL)));
    
    ax = subplot(2, 2, k + 2);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    for ia = 1:3
        plot(ax, nIt, 10 * log10(o.msd{ia}), sty{ia}, 'Color', col(ia, :), 'LineWidth', 1.3);
    end
    set(ax, 'XScale', 'log');
    xlabel(ax, 'Iteration n');
    ylabel(ax, 'MSD (dB)');
    if k == 1
        legend(ax, names, 'Location', 'southwest');
    end
end
savePdf(f, 'figs/fig_learning.pdf');

f = newFig([100, 100, 480, 360]);
ax = axes(f);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
plot(ax, cfg.thMus, dB(Jth), 'k-', 'LineWidth', 1.3);
lo = qtile(dB(Jb), 0.025);
hi = qtile(dB(Jb), 0.975);
ebar(ax, cfg.thMus, dB(Jsim), dB(Jsim) - lo, hi - dB(Jsim), 'o', col(1, :));
xlabel(ax, 'Step size \mu');
ylabel(ax, 'Steady-state MSE (dB)');
legend(ax, 'Theory', 'Simulation (95% CI)', 'Location', 'northwest');
savePdf(f, 'figs/fig_ss_theory.pdf');

f = newFig([100, 100, 480, 360]);
ax = axes(f);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
for ia = 1:3
    y = dB(trBest(:, ia)).';
    lo = dB(squeeze(trCI(1, :, ia)));
    hi = dB(squeeze(trCI(2, :, ia)));
    ebar(ax, cfg.sq2s, y, y - lo(:).', hi(:).' - y, ['o', sty{ia}], col(ia, :));
end
set(ax, 'XScale', 'log');
xlabel(ax, 'Per-coefficient random-walk variance \sigma_q^2');
ylabel(ax, 'Steady-state MSE (dB)');
legend(ax, names, 'Location', 'northwest');
savePdf(f, 'figs/fig_tracking.pdf');

f = newFig([100, 100, 480, 360]);
ax = axes(f);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
sel = ismember(allL, cfg.tLs);
for ia = 1:3
    ebar(ax, allL(sel), tMed(sel, ia), tMed(sel, ia) - tQ1(sel, ia), ...
        tQ3(sel, ia) - tMed(sel, ia), ['o', sty{ia}], col(ia, :));
end
set(ax, 'XScale', 'log', 'YScale', 'log', 'XTick', cfg.tLs);
xlabel(ax, 'Filter length L');
ylabel(ax, 'Time per update (\mus)');
legend(ax, names, 'Location', 'northwest');
savePdf(f, 'figs/fig_runtime.pdf');

save('results/results.mat', 'cfg', 'env', 'res', 'allL', 'tMed', 'tQ1', 'tQ3', 'tMin', 'tInit', ...
     'Jsim', 'Jth', 'Jb', 'ssr', 'trBest', 'trCI', 'trPick', '-v7');
fprintf('\nDone. Figures in figs/, tables in tables/, raw data in results/results.mat\n');


%% Helper Functions

function [x, d, wo, sv2] = genData(N, L, rho, snr_dB)
    wo = randn(L, 1);
    wo = wo / norm(wo);
    xi = randn(N, 1);
    if rho == 0
        x = xi;
    else
        x0 = randn;
        x = filter(sqrt(1 - rho^2), [1, -rho], xi, rho * x0);
    end
    R = toeplitz(rho .^ (0:L-1));
    sv2 = (wo.' * R * wo) / 10^(snr_dB / 10);
    d = filter(wo, 1, x) + sqrt(sv2) * randn(N, 1);
end

function [x, d] = genTrack(N, L, sq2, sv2)
    wo = randn(L, 1);
    wo = wo / norm(wo);
    x = randn(N, 1);
    U = zeros(L, N);
    for k = 1:L
        U(k, k:N) = x(1:N-k+1).';
    end
    Wo = wo + cumsum(sqrt(sq2) * randn(L, N), 2);
    d = sum(Wo .* U, 1).' + sqrt(sv2) * randn(N, 1);
end

function p = gridParams(alg, c, L, cfg)
    switch alg
        case 'lms'
            p = struct('mu', c / L);
        case 'nlms'
            p = struct('mu', c, 'eps', cfg.eps);
        case 'rls'
            p = struct('lambda', 1 - c / L, 'delta', cfg.delta);
    end
end

function p = trParams(alg, g, cfg)
    switch alg
        case 'lms'
            p = struct('mu', g);
        case 'nlms'
            p = struct('mu', g, 'eps', cfg.eps);
        case 'rls'
            p = struct('lambda', g, 'delta', cfg.delta);
    end
end

function p = midParams(alg, L, cfg)
    switch alg
        case 'lms'
            p = gridParams(alg, 0.1, L, cfg);
        case 'nlms'
            p = gridParams(alg, 0.2, L, cfg);
        case 'rls'
            p = gridParams(alg, 0.05, L, cfg);
    end
end

function [eb, msdb] = runFilter(alg, x, d, L, p, wo, b)
    N = numel(x);
    nb = N / b;
    eb = zeros(nb, 1);
    msdb = nan(nb, 1);
    w = zeros(L, 1);
    u = zeros(L, 1);
    acc = 0;
    
    kind = find(strcmp(alg, {'lms', 'nlms', 'rls'}));
    if kind == 3
        P = eye(L) / p.delta;
    end
    
    for n = 1:N
        u = [x(n); u(1:L-1)];
        e = d(n) - w.' * u;
        if kind == 1
            w = w + (p.mu * e) * u;
        elseif kind == 2
            w = w + (p.mu * e / (p.eps + u.' * u)) * u;
        else
            Pu = P * u;
            k = Pu / (p.lambda + u.' * Pu);
            w = w + k * e;
            P = (P - k * Pu.') / p.lambda;
            P = (P + P.') / 2;
        end
        
        acc = acc + e * e;
        if mod(n, b) == 0
            j = n / b;
            eb(j) = acc / b;
            acc = 0;
            if ~isempty(wo)
                dw = w - wo;
                msdb(j) = dw.' * dw;
            end
        end
    end
end

function w = timeRun(alg, x, d, L, p)
    N = numel(x);
    w = zeros(L, 1);
    u = zeros(L, 1);
    
    switch alg
        case 'lms'
            for n = 1:N
                u = [x(n); u(1:L-1)];
                e = d(n) - w.' * u;
                w = w + (p.mu * e) * u;
            end
        case 'nlms'
            for n = 1:N
                u = [x(n); u(1:L-1)];
                e = d(n) - w.' * u;
                w = w + (p.mu * e / (p.eps + u.' * u)) * u;
            end
        case 'rls'
            P = eye(L) / p.delta;
            for n = 1:N
                u = [x(n); u(1:L-1)];
                e = d(n) - w.' * u;
                Pu = P * u;
                k = Pu / (p.lambda + u.' * Pu);
                w = w + k * e;
                P = (P - k * Pu.') / p.lambda;
                P = (P + P.') / 2;
            end
    end
end

function ops = opsPerUpdate(L)
    ops = [2*L + 1, 3*L + 2, 4*L^2 + 4*L];
end

function initOnly(alg, L, p)
    w = zeros(L, 1);
    u = zeros(L, 1);
    if strcmp(alg, 'rls')
        P = eye(L) / p.delta;
    end
end

function C = cleanCurve(C)
    C(~isfinite(C)) = 1e6;
end

function k = firstSustained(C, T, W)
    ok = double(C <= T);
    cs = [zeros(size(ok, 1), 1), cumsum(ok, 2)];
    s = cs(:, W+1:end) - cs(:, 1:end-W);
    full = (s == W);
    [hit, k] = max(full, [], 2);
    k = k(:);
    k(~hit(:)) = Inf;
end

function [ix, reached] = pickSetting(C, T, W)
    k = firstSustained(C, T, W);
    if all(isinf(k))
        [~, ix] = min(mean(C(:, end-W+1:end), 2));
        reached = false;
    else
        [~, ix] = min(k);
        reached = true;
    end
end

function e = isEdge(k, n, hard)
    e = (k == 1 || k == n) && ~any(k == hard);
end

function q = qtile(X, p)
    if isvector(X)
        X = X(:);
    end
    X = sort(X, 1);
    n = size(X, 1);
    q = X(min(n, max(1, ceil(p * n))), :);
end

function s = winnerLabel(f, names)
    if sum(f) == 0
        s = 'not reached';
        return;
    end
    f = f / sum(f);
    [m, i] = max(f);
    if m >= 0.95
        s = sprintf('%s (%.0f%%)', names{i}, 100 * m);
    else
        s = ['tie: ', strjoin(names(f >= 0.05), '/')];
    end
end

function c = winnerCode(f)
    if sum(f) == 0
        c = 5;
        return;
    end
    f = f / sum(f);
    [m, i] = max(f);
    if m >= 0.95
        c = i;
    else
        c = 4;
    end
end

function s = fmtN(n)
    if isinf(n)
        s = 'n.r.';
    else
        s = sprintf('%d', round(n));
    end
end

function s = fmtCms(c, ci)
    if isinf(c)
        s = 'n.r.';
        return;
    end
    if isinf(ci(2))
        hiS = 'Inf';
    else
        hiS = sprintf('%.3g', ci(2) / 1000);
    end
    s = sprintf('%.3g [%.3g, %s]', c / 1000, ci(1) / 1000, hiS);
end

function writeLines(fname, C)
    fid = fopen(fname, 'w');
    fprintf(fid, '%s\n', C{:});
    fclose(fid);
end

function s = cpuName()
    s = 'unknown CPU';
    try
        if ispc
            s = getenv('PROCESSOR_IDENTIFIER');
        elseif ismac
            [~, s] = system('sysctl -n machdep.cpu.brand_string');
        else
            [~, s] = system('grep -m1 "model name" /proc/cpuinfo | cut -d: -f2');
        end
        s = strtrim(s);
    catch
    end
end

function f = newFig(pos)
    f = figure('Visible', 'off', 'Color', 'w', 'Position', pos);
end

function bandFill(ax, xv, Y, c)
    lo = Y(1, :);
    hi = Y(2, :);
    hi(~isfinite(hi)) = 60;
    lo(~isfinite(lo)) = 60;
    patch('Parent', ax, 'XData', [xv, fliplr(xv)], 'YData', [lo, fliplr(hi)], ...
        'FaceColor', c, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end

function h = ebar(ax, x, y, lo, hi, spec, c)
    f = ancestor(ax, 'figure');
    set(0, 'CurrentFigure', f);
    set(f, 'CurrentAxes', ax);
    h = errorbar(x, y, lo, hi, spec);
    set(h, 'Color', c, 'LineWidth', 1.3);
end

function savePdf(f, out)
    try
        exportgraphics(f, out, 'ContentType', 'vector');
    catch
        set(f, 'PaperPositionMode', 'auto');
        print(f, out, '-dpdf');
    end
    close(f);
end