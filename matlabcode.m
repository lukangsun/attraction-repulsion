% matlabcode.m
% Reproduces the numerical figures and the refinement study of the manuscript
% "A Nonlocal Attraction-Repulsion Transport Equation".
%
% Part I   : d=1, a>r.  The zero-flux stationary state phi_I on I=[-L,L] is
%            computed from the force-balance condition (7.4) of Proposition 7.1
%            (nonnegative least squares with the mass constraint); the
%            half-width L is selected by minimizing the RMS force residual.
%            The Green-kernel reconstruction (7.3) of the same proposition,
%                phi_I = ( Q_{a,r} - c_{1,r} (-Delta)^sigma H )|_I,
%                H(x)  = c_{1,r}^{-1} int_Omega G_Omega^sigma(x,y) Q_{a,r}(y) dy,
%            is evaluated independently with the exterior interval Green
%            kernel (Boggio/Kelvin formula) and compared with phi_I from (7.4).
%            A long-time particle solution phi_T is compared with both.
%            The differentiated form (7.2) is solved as a cross-check only.
% Part II  : d=2, radial Gaussian background.  Ring-mass zero-flux solve
%            (7.13) with R selected by the residual scan, the disk formula
%            (7.11)-(7.12) evaluated with the exterior Boggio kernel and all
%            constants analytic (no fitted coefficient), and the radial
%            particle evolution.
% Part III : refinement study (set runRefinementStudy=false to skip).  It
%            writes refinement_results.txt, refinement_tables.tex and the figure
%            fig_refinement_study.{pdf,png}.
%
% Implementation notes:
%  * All constants of the Green reconstructions (C_{d,a,r}, gamma_{d,a}/gamma_{d,r},
%    kappa_{d,s}, C_{d,s}) are the analytic ones of Table 1 of the manuscript;
%    nothing is fitted, and the raw (unnormalized) mass of each reconstruction
%    is printed as an independent diagnostic.
%  * Boggio's t-integrals are evaluated exactly with the regularized incomplete
%    beta function betainc (no hypergeom), in one and two dimensions.
%
% Requirements: MATLAB R2019b or later (lsqnonneg, fzero, fminbnd, besselj,
% betainc, quantile, histcounts, xline, yline, sgtitle); no toolbox needed.
% Also runs under GNU Octave >= 8 with the shims xline/yline/sgtitle/histcounts.

clear; close all; clc;

runRefinementStudy = true;

envStr = version;
if exist('OCTAVE_VERSION','builtin') == 5
    envStr = ['GNU Octave ' OCTAVE_VERSION];
end
fprintf('Environment: %s\n', envStr);

%% ========================================================================
% Part I. One-dimensional experiment with a > r
% ========================================================================
% Background omega_m = (m/2) 1_{[-1,1]}, exponents a = 0.8 > r = 0.2.
a1 = 0.80;  r1 = 0.20;  m1 = 1.80;
n1     = 320;      % cells on I=[-L,L] (unknown densities, exploiting evenness)
nTest1 = 240;      % force test points in (0,L)
massW1 = 120.0;    % weight of the appended mass equation
NeDec  = 40;       % exterior cells per decade in the Green reconstruction (7.3)
Ymax1  = 1e4;      % exterior truncation for (7.3); tails treated separately
N1 = 2000;  dt1 = 0.02;  T1 = 50.00;  L0 = 1.40;   % N1 = 2000 particles for a smooth histogram

% --- zero-flux stationary state (7.4): force balance V'(x) = K_r*phi(x) on I,
%     mass one, phi >= 0; the half-width L is selected by minimizing the RMS
%     force residual (scan, then fminbnd).
fprintf('\n1D: scanning the support half-width L with the zero-flux system (7.4)...\n');
tS = tic;
scanL = (0.240:0.0025:0.310)';
scanRms1 = zeros(size(scanL));
for k = 1:numel(scanL)
    [~,~,resk] = solveZeroFlux1D(scanL(k),a1,r1,m1,n1,nTest1,massW1);
    scanRms1(k) = sqrt(mean(resk.^2));
end
[~,kmin] = min(scanRms1);
Lbr = [scanL(max(kmin-1,1)), scanL(min(kmin+1,numel(scanL)))];
Lstat = fminbnd(@(LL) rmsZeroFlux1D(LL,a1,r1,m1,n1,nTest1,massW1), Lbr(1), Lbr(2), optimset('TolX',1e-7));
[phiStat,edges1,res1,xTest1] = solveZeroFlux1D(Lstat,a1,r1,m1,n1,nTest1,massW1);
tS = toc(tS);
centers1 = 0.5*(edges1(1:end-1)+edges1(2:end));
dx1 = edges1(2)-edges1(1);
cdfStat1 = [0; cumsum(phiStat)*dx1];
cdfStat1 = cdfStat1/cdfStat1(end);
fprintf('1D: zero-flux (7.4): L (residual minimizer)  %.8f   (%.0f s)\n', Lstat, tS);
fprintf('1D: zero-flux (7.4): RMS force residual      %.4e\n', sqrt(mean(res1.^2)));
fprintf('1D: zero-flux (7.4): max force residual      %.4e\n', max(abs(res1)));
fprintf('1D: zero-flux (7.4): mass                    %.12f\n', sum(phiStat)*dx1);
% cross-check: the differentiated form (7.2) with L from the mass condition
LRiesz = fzero(@(L) stationaryMassDefect1D(L,a1,r1,m1,n1), [0.16,0.45]);
[~,~,~,phiRiesz,~] = solveStationaryDensity1D(Lstat,a1,r1,m1,n1);
fprintf('1D: cross-check (7.2): L from the mass condition %.8f, |dL| = %.2e; density difference (central 90%%) %.2e\n', ...
    LRiesz, abs(LRiesz-Lstat), max(abs(phiRiesz(17:304)-phiStat(17:304)))/max(phiStat));

% --- Green-kernel reconstruction (7.3) on the same grid and the same L.
fprintf('1D: evaluating the Green-kernel reconstruction (7.3)...\n');
tG = tic;
[phiGreenRaw, infoG] = greenReconstruction1D(Lstat,a1,r1,m1,n1,NeDec,Ymax1);
tG = toc(tG);
phiGreen = max(phiGreenRaw,0);
cdfGreen = [0; cumsum(phiGreen)*dx1];
cdfGreen = cdfGreen/cdfGreen(end);
[greenCdfErr, greenDensErr] = greenMetrics(phiGreenRaw,phiStat,cdfStat1,dx1);
fprintf('1D: Green (7.3): exterior cells per side  %d  (Ymax = %.0e), %.1f s\n', infoG.Ne, Ymax1, tG);
fprintf('1D: Green (7.3): raw mass                 %.8f\n', infoG.rawMass);
fprintf('1D: Green (7.3): negative-part mass       %.3e\n', infoG.negMass);
fprintf('1D: Green (7.3): tail coefficient ratio   %.6f  (computed/asymptotic)\n', infoG.tailRatio);
fprintf('1D: max CDF discrepancy (7.3) vs (7.4)    %.4e\n', greenCdfErr);
fprintf('1D: max rel. density discrepancy (central 90%% of cells) %.4e\n', greenDensErr);

% --- Particle dynamics.
fprintf('1D: evolving %d particles to T = %.0f...\n', N1, T1);
[X, time, diam, q99] = evolveParticles1D(N1,dt1,T1,L0,a1,r1,m1,12);
empAtEdges = arrayfun(@(z) sum(X <= z)/N1, edges1);
cdfErrMax = max(abs(empAtEdges(:)-cdfStat1(:)));
fprintf('1D: final diameter                     %.8f  (2L = %.8f)\n', diam(end), 2*Lstat);
fprintf('1D: max CDF discrepancy particles vs (7.4) %.4e\n', cdfErrMax);

histEdges = linspace(-1.05*Lstat,1.05*Lstat,55);
[counts,histEdges] = histcounts(X,histEdges,'Normalization','pdf');
histCenters = 0.5*(histEdges(1:end-1)+histEdges(2:end));

fig1 = figure('Color','w','Position',[80 100 1250 390]);
subplot(1,3,1);
plot(centers1,phiStat,'LineWidth',2.2); hold on;
plot(centers1,phiGreen,'-.','LineWidth',2.0);
plot(histCenters,counts,'--','LineWidth',1.6);
xline(-Lstat,':','LineWidth',1.4); xline(Lstat,':','LineWidth',1.4);
grid on; box on;
xlim([-0.46,0.46]);
xlabel('x'); ylabel('density');
title('density comparison');
legend({'$\phi_I$ from (7.4)','$\phi_I$ from (7.3)','particle histogram'}, ...
       'Location','north','Interpreter','latex');

subplot(1,3,2);
plot(edges1,cdfStat1,'LineWidth',2.2); hold on;
plot(edges1,cdfGreen,'-.','LineWidth',2.0);
stairs(X,(1:N1)'/N1,'--','LineWidth',1.8);
grid on; box on;
xlim([-0.46,0.46]); ylim([0,1.02]);
xlabel('x'); ylabel('cumulative mass');
title('cumulative mass comparison');
legend({'$\phi_I$ from (7.4)','$\phi_I$ from (7.3)','particle $\phi_T$'}, ...
       'Location','southeast','Interpreter','latex');

subplot(1,3,3);
semilogy(scanL, scanRms1, 'o-', 'LineWidth',1.6, 'MarkerSize',4); hold on;
semilogy(Lstat, sqrt(mean(res1.^2)), 's', 'MarkerSize',9, 'LineWidth',2);
grid on; box on;
xlabel('prescribed half-width L'); ylabel('RMS force residual');
title('zero-flux residual vs. L');
legend({'scan','minimizer $L$'},'Location','north','Interpreter','latex');

saveFigure(fig1,'1');



%% ========================================================================
% Part II. Two-dimensional example with d=2 and a >= r >= 0
% ========================================================================
% Background: omega_m(x) = m/(2*pi)*exp(-|x|^2/2).
m = 1.80;  a = 0.60;  r = 0.20;
nTheta = 160;  nRing = 110;  nTest = 150;  nBg = 550;  Lbg = 6.0;  massW = 120.0;

% --- support radius R: minimize the RMS residual of the zero-flux system (7.13)
fprintf('\n2D: scanning the support radius R with the zero-flux ring system (7.13)...\n');
tS = tic;
scanR = (0.260:0.0025:0.400)';
scanRms2 = zeros(size(scanR)); scanOuter2 = zeros(size(scanR));
for k = 1:numel(scanR)
    [muk,sk,~,resk] = solveRingMasses2D(scanR(k),a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW);
    scanRms2(k) = sqrt(mean(resk.^2)); scanOuter2(k) = sum(muk(sk > 0.9*scanR(k)));
end
[~,kmin] = min(scanRms2);
Rbr = [scanR(max(kmin-1,1)), scanR(min(kmin+1,numel(scanR)))];
R = fminbnd(@(RR) rmsRing2D(RR,a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW), Rbr(1), Rbr(2), optimset('TolX',1e-6));
[mu, sRing, rTest, resid, Amat, bvec] = solveRingMasses2D(R,a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW);
tS = toc(tS);
fprintf('2D: zero-flux (7.13): R (residual minimizer) %.6f   (%.0f s)\n', R, tS);
fprintf('2D: stationary mass          %.15f\n', sum(mu));
fprintf('2D: stationary RMS residual  %.4e\n', sqrt(mean(resid.^2)));
fprintf('2D: stationary max residual  %.4e\n', max(abs(resid)));
fprintf('2D: mass of the ring solution in (0.9R,R]: %.4f\n', sum(mu(sRing > 0.9*R)));

% Direct reconstruction from the disk formula (7.11)-(7.12) at the same R,
% with all constants analytic (no fitted coefficient), on a radial mesh graded
% towards the boundary rho = R (IntDec cells per decade of R-rho).
NeDec2 = 40;  Ymax2 = 1e4;  nAng2 = 64;  IntDec2 = 40;
fprintf('2D: evaluating the Green-kernel disk formula (7.11)-(7.12)...\n');
tG = tic;
rhoEdges2 = interiorMesh2D(R,IntDec2);
[phiDisk, infoG2] = greenReconstruction2D(R,a,r,m,rhoEdges2,NeDec2,Ymax2,nAng2);
tG = toc(tG);
nuDisk = infoG2.nu/sum(infoG2.nu);
Adisk = ringKernelMatrix(rTest, infoG2.rc, r, nTheta);
residDisk = Adisk*nuDisk - bvec;
fprintf('2D: Green (7.11): exterior cells %d, interior cells %d, %.1f s\n', infoG2.Ne, numel(phiDisk), tG);
fprintf('2D: Green (7.11): raw mass                 %.6f\n', infoG2.rawMass);
fprintf('2D: Green (7.11): negative-part mass       %.3e\n', infoG2.negMass);
fprintf('2D: Green (7.11): boundary exponent        %.3f  (expected -s = %.2f)\n', infoG2.boundaryExponent, -(1+r)/2);
fprintf('2D: Green (7.11): force residual of (7.13) RMS %.3e, max %.3e\n', sqrt(mean(residDisk.^2)), max(abs(residDisk)));

% Radial particle dynamics.
fprintf('2D: evolving the radial particle system...\n');
Npart = 180;  dt = 0.05;  nSteps = 1000;
[phiRad, tabs2D] = evolveRadial2D(Npart,dt,nSteps,a,r,m,nBg,Lbg,[]);

rPlot = linspace(0,0.66,500)';
cdfStat = cumulativeRingMass(mu,sRing,rPlot);
cdfFormula = cumulativeRingMass(nuDisk,infoG2.rc,rPlot);
cdfDyn = arrayfun(@(rr) mean(phiRad <= rr), rPlot);
% Two distances between cumulative radial masses: the sup norm and the L^1
% distance W1 = int |F_1 - F_2| d rho (the Wasserstein-1 distance of the radial
% mass distributions).  The sup norm is dominated by the steep, oscillatory
% boundary layer of the ring-mass solution near rho = R, whereas W1 measures
% the mass allocation itself.
fprintf('2D: (7.11) vs (7.13):   sup|dF| = %.3e   W1 = %.3e\n', max(abs(cdfFormula-cdfStat)), trapz(rPlot,abs(cdfFormula-cdfStat)));
fprintf('2D: particles vs (7.13): sup|dF| = %.3e   W1 = %.3e\n', max(abs(cdfDyn-cdfStat)), trapz(rPlot,abs(cdfDyn-cdfStat)));

golden = pi*(3-sqrt(5));
angles = mod((0:Npart-1)'*golden, 2*pi);
Xcloud = [phiRad.*cos(angles), phiRad.*sin(angles)];

fig2 = figure('Color','w','Position',[80 100 1250 360]);
subplot(1,3,1);
plot(Xcloud(:,1),Xcloud(:,2),'o','MarkerSize',3); hold on;
th = linspace(0,2*pi,400);
plot(R*cos(th),R*sin(th),'--','LineWidth',1.5);
axis equal; grid on; box on;
xlim([-0.55,0.55]); ylim([-0.55,0.55]);
xlabel('x_1'); ylabel('x_2');
title('particle cloud');
legend({'particle $\phi_T$',sprintf('$R\\approx%.2f$',R)}, ...
       'Location','northeast','Interpreter','latex');

subplot(1,3,2);
plot(rPlot,cdfStat,'LineWidth',2); hold on;
plot(rPlot,cdfFormula,'-.','LineWidth',2);
stairs(sort(phiRad), (1:Npart)'/Npart, '--', 'LineWidth',2);
xline(R,':','LineWidth',1.5);
grid on; box on;
xlim([0,0.66]); ylim([0,1.05]);
xlabel('radius'); ylabel('cumulative mass');
title('cumulative radial mass comparison');
legend({'$\phi_R$ from (7.13)','$\phi_R$ from (7.11)','particle $\phi_T$',sprintf('$R\\approx%.2f$',R)}, ...
       'Location','southeast','Interpreter','latex');

subplot(1,3,3);
semilogy(scanR, scanRms2, 'o-', 'LineWidth',1.6, 'MarkerSize',4); hold on;
semilogy(R, sqrt(mean(resid.^2)), 's', 'MarkerSize',9, 'LineWidth',2);
grid on; box on;
xlabel('prescribed radius R'); ylabel('RMS force residual');
title('zero-flux residual vs. R');
legend({'scan','minimizer $R$'},'Location','north','Interpreter','latex');

saveFigure(fig2,'2');



%% ========================================================================
% Part III. Refinement study
% ========================================================================
if runRefinementStudy
    fprintf('\n=== Part III: refinement study ===\n');
    fid = fopen('refinement_results.txt','w');
    ftex = fopen('refinement_tables.tex','w');
    logf(fid,'Refinement study, generated %s with %s\n', datestr(now), envStr);
    fprintf(ftex,'%% Auto-generated by matlabcode.m on %s (%s).\n%% Each table body is stored in a macro; \\input this file in the preamble.\n', datestr(now), envStr);
    S = struct();

    % ---- support scans of Parts I and II (tables) ----
    S.scan1 = writeScan1D(scanL,scanRms1,Lstat,sqrt(mean(res1.^2)),LRiesz,fid,ftex);
    S.scan2 = writeScan2D(scanR,scanRms2,scanOuter2,R,sqrt(mean(resid.^2)),fid,ftex);

    % ---- III.1  1D zero-flux (7.4): cell refinement, L and CDF convergence ----
    nList = [40 80 160 320 640 1280];
    S.zf = refine1DZeroFlux(a1,r1,m1,nList,massW1,fid,ftex);

    % ---- III.2  1D Green reconstruction (7.3): exterior resolution, truncation, cells ----
    S.green = refine1DGreen(Lstat,a1,r1,m1,n1,nTest1,massW1,NeDec,Ymax1,S.zf.n,S.zf.L,fid,ftex);

    % ---- III.3  1D particles: N, dt, T ----
    S.part1 = refine1DParticles(a1,r1,m1,L0,T1,edges1,cdfStat1,Lstat,fid,ftex);

    % ---- III.4  2D ring refinement at the selected R, and 2D Green reconstruction refinement ----
    S.ring  = refine2DRing(R,a,r,m,nBg,Lbg,massW,rPlot,fid,ftex);
    S.green2 = refine2DGreen(R,a,r,m,NeDec2,Ymax2,nAng2,IntDec2,rPlot,cdfStat,rTest,bvec,nTheta,fid,ftex);

    % ---- III.5  2D radial particles: N, dt, T ----
    S.part2 = refine2DParticles(a,r,m,nBg,Lbg,rPlot,cdfStat,fid,ftex);

    fclose(fid); fclose(ftex);
    save('refinement_results.mat','S','-v7');

    % ---- figure ----
    fig3 = figure('Color','w','Position',[60 60 1250 720]);
    subplot(2,3,1);
    loglog(S.zf.n(1:end-1), abs(S.zf.L(1:end-1)-S.zf.L(end)),'o-','LineWidth',1.8); hold on;
    loglog(S.zf.n(1:end-1), S.zf.cdfErr(1:end-1),'s--','LineWidth',1.8);
    nn = S.zf.n(1:end-1); loglog(nn, S.zf.cdfErr(1)*(nn/nn(1)).^(-1),':','Color',[0.4 0.4 0.4]);
    grid on; xlabel('cells n'); ylabel(sprintf('error vs n=%d',S.zf.n(end)));
    legend({sprintf('$|L_n-L_{%d}|$',S.zf.n(end)),'max CDF error','slope $-1$'},'Location','southwest','Interpreter','latex');
    title('1D zero-flux state (7.4)');

    subplot(2,3,2);
    loglog(S.green.NeDec, S.green.cdfErrNe,'o-','LineWidth',1.8); hold on;
    loglog(S.green.NeDec, abs(S.green.massNe-1),'s--','LineWidth',1.8);
    grid on; xlabel('exterior cells per decade'); ylabel('discrepancy');
    legend({'max CDF, (7.3) vs (7.4)','$|M_G-1|$'},'Location','southwest','Interpreter','latex');
    title('1D Green kernel (7.3): exterior resolution');

    subplot(2,3,3);
    loglog(S.green.Ymax, S.green.cdfErrY,'o-','LineWidth',1.8); hold on;
    loglog(S.green.Ymax, S.green.cdfErrYnoTail,'s--','LineWidth',1.8);
    grid on; xlabel('truncation Y_{max}'); ylabel('max CDF discrepancy');
    legend({'with tail terms','without tail terms'},'Location','southwest');
    title('1D Green kernel (7.3): truncation');

    subplot(2,3,4);
    loglog(S.part1.N, S.part1.cdfErrN,'o-','LineWidth',1.8); hold on;
    loglog(S.part1.N, S.part1.cdfErrN(1)*(S.part1.N/S.part1.N(1)).^(-1),':','Color',[0.4 0.4 0.4]);
    grid on; xlabel('particles N'); ylabel('max CDF error vs (7.4)');
    legend({'$E_{\rm CDF}(N)$, $\Delta t=0.02$','slope $-1$'},'Location','southwest','Interpreter','latex');
    title('1D particles, T=50');

    subplot(2,3,5);
    semilogy(S.ring.level(2:end), S.ring.w1DiffPrev(2:end),'o-','LineWidth',1.8); hold on;
    semilogy(S.ring.level, S.ring.rmsRes,'s--','LineWidth',1.8);
    grid on; xlabel('refinement level'); ylabel('value');
    legend({'W_1 change vs previous level','RMS force residual'},'Location','northeast');
    title('2D ring masses (7.13)');
    set(gca,'XTick',S.ring.level);

    subplot(2,3,6);
    loglog(S.part2.N, S.part2.w1N,'o-','LineWidth',1.8); hold on;
    loglog(S.part2.N([1 end]), S.ring.w1DiffFinest(2)*[1 1],':','Color',[0.4 0.4 0.4],'LineWidth',1.5);
    grid on; xlabel('radial nodes N'); ylabel('W_1 error vs (7.13)');
    legend({'$W_1(N)$, $\Delta t=0.05$','$W_1$(base ring level, finest ring level)'},'Location','southwest','Interpreter','latex');
    title('2D radial particles, T=50');

    saveFigure(fig3,'fig_refinement_study');
    fprintf('=== refinement study written to refinement_results.txt / refinement_tables.tex ===\n');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions: one-dimensional stationary problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function defect = stationaryMassDefect1D(L,a,r,m,n)
    [mass,~,~,~,~] = solveStationaryDensity1D(L,a,r,m,n);
    defect = mass - 1;
end

function [phi,edges,resid,xTest] = solveZeroFlux1D(L,a,r,m,n,nTest,massW)
% Zero-flux stationary state (7.4) on I=[-L,L]:  V'(x) = (1+r) int_I sgn(x-y)|x-y|^r phi(y) dy
% for x in (-L,L), with phi >= 0 and mass one.  Piecewise constant phi on n uniform
% cells; the density is even, so the n/2 right-half values are the unknowns and
% the force (odd in x) is tested at nTest midpoints in (0,L).  The cell integrals
% of the odd kernel are exact, the overdetermined system is solved by
% nonnegative least squares with an appended mass equation of weight massW,
% and the result is normalized to mass one.  resid = force residual at xTest.
    if mod(n,2) ~= 0, error('n must be even'); end
    edges = linspace(-L,L,n+1)';  h = 2*L/n;  nh = n/2;
    xTest = (0.5:nTest-0.5)'*(L/nTest);
    Afull = (1+r)*sgnPowerCellIntegral(xTest, edges(1:end-1)', edges(2:end)', r);
    A = Afull(:,nh+1:end) + Afull(:,nh:-1:1);        % even density: pair mirrored cells
    b = uniformBackgroundForce1D(xTest,a,m);
    Aaug = [A; massW*2*h*ones(1,nh)];
    baug = [b; massW];
    if exist('lsqnonneg','file') == 2
        c = lsqnonneg(Aaug,baug);
    else
        c = projectedLeastSquares(Aaug,baug,800);
    end
    c = max(c,0);
    c = c/(2*h*sum(c));
    resid = A*c - b;
    phi = [c(end:-1:1); c];
end

function v = rmsZeroFlux1D(L,a,r,m,n,nTest,massW)
    [~,~,resid] = solveZeroFlux1D(L,a,r,m,n,nTest,massW);
    v = sqrt(mean(resid.^2));
end

function M = sgnPowerCellIntegral(x, l, u, r)
% M(i,j) = int_{l(j)}^{u(j)} sgn(x(i)-y) |x(i)-y|^r dy  (closed form, r > -1).
    x = x(:); l = l(:)'; u = u(:)';
    q = r+1;
    Xm = repmat(x,1,numel(l)); Lm = repmat(l,numel(x),1); Um = repmat(u,numel(x),1);
    M = zeros(size(Xm));
    left  = Xm >= Um;
    right = Xm <= Lm;
    mid   = ~(left | right);
    M(left)  =  ((Xm(left)-Lm(left)).^q - (Xm(left)-Um(left)).^q)/q;
    M(right) = -((Um(right)-Xm(right)).^q - (Lm(right)-Xm(right)).^q)/q;
    M(mid)   =  ((Xm(mid)-Lm(mid)).^q - (Um(mid)-Xm(mid)).^q)/q;
end

function [mass,edges,centers,phi,resid] = solveStationaryDensity1D(L,a,r,m,n)
% Solves the finite-interval Riesz equation (7.2) for d=1, a>r>0:
%   r(1+r) int_{-L}^L |x-y|^{r-1} phi(y) dy = G_a(x) = V''(x),
% by midpoint collocation with piecewise constant densities on n uniform cells.
    edges = linspace(-L,L,n+1)';
    centers = 0.5*(edges(1:end-1)+edges(2:end));
    A = r*(1+r)*absPowerCellIntegral(centers, edges(1:end-1)', edges(2:end)', r-1);
    b = uniformBackgroundSecond1D(centers,a,m);
    phi = A\b;
    dx = diff(edges);
    mass = sum(phi.*dx);
    resid = norm(A*phi-b)/norm(b);
end

function M = absPowerCellIntegral(x, l, u, p)
% M(i,j) = int_{l(j)}^{u(j)} |x(i)-y|^p dy, evaluated in closed form (p > -1).
% x is a column, l and u are rows.  The diagonal (singular) cells are exact.
    x = x(:); l = l(:)'; u = u(:)';
    q = p+1;
    Xm = repmat(x,1,numel(l)); Lm = repmat(l,numel(x),1); Um = repmat(u,numel(x),1);
    M = zeros(size(Xm));
    left  = Xm >= Um;              % cell entirely to the left of x
    right = Xm <= Lm;              % cell entirely to the right of x
    mid   = ~(left | right);
    M(left)  = ((Xm(left)-Lm(left)).^q - (Xm(left)-Um(left)).^q)/q;
    M(right) = ((Um(right)-Xm(right)).^q - (Lm(right)-Xm(right)).^q)/q;
    M(mid)   = ((Xm(mid)-Lm(mid)).^q + (Um(mid)-Xm(mid)).^q)/q;
end

function G = uniformBackgroundSecond1D(x,a,m)
% G = V'' = (psi_a*omega_m)'' for omega_m=(m/2)1_{[-1,1]}.
    G = 0.5*m*(1+a)*(sign(x+1).*abs(x+1).^a - sign(x-1).*abs(x-1).^a);
end

function F = uniformBackgroundForce1D(x,a,m)
% V' for omega_m=(m/2)1_{[-1,1]}:  F(x) = (m/2)(|x+1|^{1+a}-|x-1|^{1+a}).
    F = 0.5*m*(abs(x+1).^(1+a) - abs(x-1).^(1+a));
end

function [Q, cQ] = sourceQ1D(y,a,r,m)
% Q_{a,r} = C_{1,a,r} omega_m * |x|^{-(1-a+r)}   (eq. (6.10) of the manuscript),
% evaluated in closed form for omega_m = (m/2) 1_{[-1,1]} with beta = a-r > 0:
%   Q(y) = cQ [ sgn(y+1)|y+1|^beta - sgn(y-1)|y-1|^beta ],   cQ = C_{1,a,r} m/(2 beta),
% where C_{1,a,r} is the constant of Table 1 of the manuscript (d=1).  For
% |y| > 2 the difference is evaluated in the cancellation-free form
%   |y|^beta [ expm1(beta log1p(1/|y|)) - expm1(beta log1p(-1/|y|)) ].
    bt = a - r;
    C1ar = gamma((1-bt)/2)*gamma((2+a)/2)*gamma(-(1+r)/2) / ...
           (sqrt(pi)*gamma(bt/2)*gamma(-(1+a)/2)*gamma((2+r)/2));
    cQ = C1ar*m/(2*bt);
    Q = zeros(size(y));
    near = abs(y) <= 2;
    yn = y(near);
    Q(near) = sign(yn+1).*abs(yn+1).^bt - sign(yn-1).*abs(yn-1).^bt;
    yf = abs(y(~near));
    Q(~near) = yf.^bt.*(expm1(bt*log1p(1./yf)) - expm1(bt*log1p(-1./yf)));
    Q = cQ*Q;
end

function [phi, info] = greenReconstruction1D(L,a,r,m,n,NeDec,Ymax)
% Green-kernel reconstruction (7.3) of Proposition 7.1 on I=[-L,L]:
%   phi_I = ( Q_{a,r} - c_{1,r} (-Delta)^sigma H )|_I,      sigma = r/2,
%   H(y)  = c_{1,r}^{-1} int_Omega G_Omega^sigma(y,z) Q_{a,r}(z) dz,   y in Omega,
%   H = 0 on I,   Omega = R \ I.
% Since H vanishes on I, for x in I the fractional Laplacian is the integral
%   (-Delta)^sigma H(x) = -C_{1,sigma} int_Omega H(y) |x-y|^{-1-2sigma} dy,
% so, with W := c_{1,r} H = int_Omega G Q dz (the constant c_{1,r} cancels),
%   phi_I(x) = Q_{a,r}(x) + C_{1,sigma} int_Omega W(y) |x-y|^{-1-2sigma} dy.
% Exterior Green kernel (Boggio/Kelvin form given in Proposition 7.1):
%   G(y,z) = kappa_{1,sigma} |y-z|^{r-1} int_0^{eta} t^{sigma-1}(1+t)^{-1/2} dt,
%   eta    = (y^2-L^2)(z^2-L^2) / (L^2 (y-z)^2),
%   int_0^{eta} t^{sigma-1}(1+t)^{-1/2} dt = B(eta/(1+eta); sigma, 1/2-sigma)
%          = betainc(eta/(1+eta), sigma, 1/2-sigma) * beta(sigma, 1/2-sigma).
% Discretization: n uniform cells on I (midpoints x_i); on the exterior
% t = z-L in (0, Ymax-L] a first cell [0, h/4] followed by geometrically graded
% cells with NeDec cells per decade (the point z=1, where Q has a cusp, is a
% cell edge).  Both exterior integrals use product integration: the smooth
% factors are frozen at cell midpoints and the singular factors |y-z|^{r-1}
% and (y-x)^{-1-2sigma} are integrated exactly over each cell, so that the
% diagonal cells of the first integral are exact.  W is symmetric, so only the
% right half-line is stored; the mirrored half-line enters through the
% kernels G(y,-z) and (y+x)^{-1-2sigma}.
% Tails beyond Ymax: for W, the z-integral over (Ymax, inf) is computed with a
% Gauss-Legendre rule in the mapped variable z = Ymax (1-s)^{-1/(1-a)}; for
% phi, W is extrapolated with its exact decay W(y) ~ y^{-(1-a)} and the
% y-integral is again a mapped Gauss-Legendre rule.
    sigma = r/2;  bt = a - r;  h = 2*L/n;
    edges = linspace(-L,L,n+1)';
    x = 0.5*(edges(1:end-1)+edges(2:end));

    kappa = 1/(4^sigma*gamma(sigma)^2);                            % kappa_{1,sigma}
    Cfl   = 4^sigma*gamma(0.5+sigma)/(sqrt(pi)*abs(gamma(-sigma))); % C_{1,sigma}
    Binf  = beta(sigma, 0.5-sigma);
    boggio = @(xx) Binf*betainc(xx, sigma, 0.5-sigma);              % xx = eta/(1+eta)

    % ---- exterior mesh on the right half-line ----
    tmin = h/4;  Te = Ymax - L;
    K = max(4, round(NeDec*log10(Te/tmin)));
    zE = L + [0; tmin*(Te/tmin).^((0:K)'/K)];
    if Ymax > 1
        zE = sort([zE; 1]);
        zE = zE([true; diff(zE) > 1e-9*max(1,abs(zE(2:end)))]);
    end
    zl = zE(1:end-1);  zu = zE(2:end);  zc = 0.5*(zl+zu);  Ne = numel(zc);
    Qc = sourceQ1D(zc,a,r,m);

    % ---- step 1: W(y_j) = int_Omega G(y_j,z) Q(z) dz at exterior midpoints ----
    [Y,Z]  = ndgrid(zc, zc);
    Nn  = (Y.^2-L^2).*(Z.^2-L^2);
    Bp  = boggio(Nn./(Nn + L^2*(Y-Z).^2));   % same side; diagonal -> B(inf)
    Bm  = boggio(Nn./(Nn + L^2*(Y+Z).^2));   % mirrored side
    Wp  = absPowerCellIntegral(zc, zl', zu', r-1);
    ZL = repmat(zl',Ne,1); ZU = repmat(zu',Ne,1);
    Wm  = ((Y+ZU).^r - (Y+ZL).^r)/r;         % int_{cell} (y+z)^{r-1} dz
    Gmat = kappa*(Bp.*Wp + Bm.*Wm);
    W0 = Gmat*Qc;

    % tail of the z-integral beyond Ymax (mapped Gauss-Legendre)
    pz = 1/(1-a);
    [sg, wg] = gaussLegendre01(48);
    zt  = Ymax*(1-sg).^(-pz);
    dzt = Ymax*pz*(1-sg).^(-pz-1).*wg;
    [Yt,Zt] = ndgrid(zc, zt);
    Nt  = (Yt.^2-L^2).*(Zt.^2-L^2);
    Gt  = kappa*( boggio(Nt./(Nt+L^2*(Yt-Zt).^2)).*abs(Yt-Zt).^(r-1) + ...
                  boggio(Nt./(Nt+L^2*(Yt+Zt).^2)).*(Yt+Zt).^(r-1) );
    tailW = Gt*(sourceQ1D(zt,a,r,m).*dzt);
    W = W0 + tailW;

    % ---- step 2: phi(x_i) = Q(x_i) + C_{1,sigma} int_Omega W(y)|x_i-y|^{-1-2sigma} dy ----
    [Xi, ZLi] = ndgrid(x, zl');  ZUi = repmat(zu',n,1);
    Vp = ((ZLi - Xi).^(-r) - (ZUi - Xi).^(-r))/r;    % int_{cell}(y-x)^{-1-r} dy
    Vm = ((ZLi + Xi).^(-r) - (ZUi + Xi).^(-r))/r;    % mirrored half-line
    corr0 = (Vp + Vm)*W0;
    corr  = (Vp + Vm)*W;

    % tail of the y-integral beyond Ymax: W(y) ~ W(zc_end) (y/zc_end)^{-(1-a)}
    gam = 1 - a;
    py = 1/(gam + r);
    yt  = Ymax*(1-sg).^(-py);
    dyt = Ymax*py*(1-sg).^(-py-1).*wg;
    Wt  = W(end)*(yt/zc(end)).^(-gam);
    tailPhi = ( (yt'-x).^(-1-r) + (yt'+x).^(-1-r) ) * (Wt.*dyt);   % n x 1

    Qx = sourceQ1D(x,a,r,m);
    phi        = Qx + Cfl*(corr + tailPhi);
    phiNoTail  = Qx + Cfl*corr0;

    % asymptotic constant of W (from the Green representation) for a check
    [~, cQ] = sourceQ1D(0,a,r,m);
    Aasym = kappa*Binf*2*bt*cQ*(beta(bt,r) + beta(1-bt-r,r) + beta(bt,1-bt-r));
    info = struct();
    info.edges = edges; info.centers = x; info.h = h;
    info.zc = zc; info.zEdges = zE; info.W = W; info.Ne = Ne; info.K = K;
    info.rawMass = sum(phi)*h;
    info.negMass = sum(max(-phi,0))*h;
    info.phiNoTail = phiNoTail;
    info.rawMassNoTail = sum(phiNoTail)*h;
    info.tailRatio = W(end)*zc(end)^gam/Aasym;
    info.constants = struct('kappa',kappa,'Cfl',Cfl,'Binf',Binf,'Aasym',Aasym);
end

function [x,w] = gaussLegendre01(nq)
% Gauss-Legendre nodes and weights on [0,1] (Golub-Welsch).
    k = (1:nq-1)';  bb = k./sqrt(4*k.^2-1);
    J = diag(bb,1)+diag(bb,-1);
    [V,D] = eig(J);
    [xg,idx] = sort(diag(D));
    wg = 2*(V(1,idx).^2)';
    x = 0.5*(xg+1);  w = 0.5*wg;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions: one-dimensional particle dynamics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, time, diam, q99] = evolveParticles1D(N,dt,T,L0,a,r,m,seed)
% Explicit Euler for  X_i' = -V'(X_i) + (1/N) sum_j K_r(X_i-X_j),  K_r(z)=(1+r)sgn(z)|z|^r.
    nSteps = round(T/dt);
    rng(seed);
    X = linspace(-L0,L0,N)' + 0.01*randn(N,1);
    X = sort(X);
    nRec = floor(nSteps/4);
    time = zeros(nRec,1); diam = zeros(nRec,1); q99 = zeros(nRec,1); ir = 0;
    for k = 1:nSteps
        X = X + dt*(-uniformBackgroundForce1D(X,a,m) + repulsionForce1D(X,r));
        if mod(k,20)==0
            X = sort(X);
        end
        if mod(k,4)==0
            ir = ir+1;
            time(ir) = k*dt;
            diam(ir) = max(X)-min(X);
            q99(ir)  = 2*quantile(abs(X),0.99);
        end
    end
    X = sort(X);
    time = time(1:ir); diam = diam(1:ir); q99 = q99(1:ir);
end

function F = repulsionForce1D(x,r)
    Z = x - x';
    K = (1+r)*sign(Z).*abs(Z).^r;
    K(1:size(K,1)+1:end) = 0;
    F = mean(K,2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions: two-dimensional radial problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = ringKernelMatrix(rVec, sVec, alpha, nTheta)
% A(i,j) is the radial component at radius rVec(i) generated by one unit of
% mass uniformly distributed on the circle of radius sVec(j) for the vector
% kernel K_alpha(z) = (1+alpha) z |z|^{alpha-1}.
    rVec = rVec(:);
    sVec = sVec(:)';
    theta = ((1:nTheta)-0.5)*(2*pi/nTheta);
    cth = cos(theta);
    A = zeros(numel(rVec), numel(sVec));
    chunk = 80;
    eps2 = 1e-14;
    for j0 = 1:chunk:numel(sVec)
        j1 = min(j0+chunk-1,numel(sVec));
        ss = sVec(j0:j1);
        block = zeros(numel(rVec),numel(ss));
        for k = 1:nTheta
            c = cth(k);
            d2 = max(rVec.^2 + ss.^2 - 2*rVec*ss*c, eps2);
            block = block + (1+alpha)*(rVec - ss*c).*d2.^((alpha-1)/2);
        end
        A(:,j0:j1) = block/nTheta;
    end
end

function [mu, sRing, rTest, resid, A, b] = solveRingMasses2D(R,a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW)
% Nonnegative least-squares solution of the radial zero-flux system (7.13)
% for ring masses on the disk B_R, with an appended mass equation of weight massW.
    sRing = ((1:nRing)' - 0.5) * R/nRing;
    rTest = ((1:nTest)' - 0.5) * R/nTest;
    A = ringKernelMatrix(rTest, sRing, r, nTheta);
    sBg  = ((1:nBg)' - 0.5) * Lbg/nBg;
    dsBg = Lbg/nBg;
    wBg  = m * sBg .* exp(-0.5*sBg.^2) * dsBg;         % ring mass of m/(2 pi) e^{-|x|^2/2}
    B = ringKernelMatrix(rTest, sBg, a, nTheta);
    b = B*wBg;
    Aaug = [A; massW*ones(1,nRing)];
    baug = [b; massW];
    if exist('lsqnonneg','file') == 2
        mu = lsqnonneg(Aaug,baug);
    else
        warning('lsqnonneg not found; using a projected least-squares fallback.');
        mu = projectedLeastSquares(Aaug,baug,800);
    end
    mu = max(mu,0);
    mu = mu/sum(mu);
    resid = A*mu - b;
end

function v = rmsRing2D(R,a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW)
    [~,~,~,resid] = solveRingMasses2D(R,a,r,m,nRing,nTest,nTheta,nBg,Lbg,massW);
    v = sqrt(mean(resid.^2));
end

function cdf = cumulativeRingMass(mu,sRing,rQuery)
    cdf = zeros(size(rQuery));
    for i = 1:numel(rQuery)
        cdf(i) = sum(mu(sRing <= rQuery(i)));
    end
end

function [phiRad, tabs] = evolveRadial2D(Npart,dt,nSteps,a,r,m,nBg,Lbg,tabs)
% Radial particle dynamics  R_i' = - int A_a(R_i,s) d omega_m(s) + (1/N) sum_j A_r(R_i,R_j)
% with interpolation tables for the ring kernels (tables reused if supplied).
    if isempty(tabs)
        rMaxGrid = 3.2; nGrid = 260;
        tabs.rGrid = linspace(0,rMaxGrid,nGrid)';
        tabs.Kself = ringKernelMatrix(tabs.rGrid, tabs.rGrid, r, 96);
        sBg  = ((1:nBg)' - 0.5) * Lbg/nBg;
        wBg  = m * sBg .* exp(-0.5*sBg.^2) * (Lbg/nBg);
        tabs.Bgrid = ringKernelMatrix(tabs.rGrid, sBg, a, 96) * wBg;
    end
    u = ((1:Npart)' - 0.5)/Npart;
    phiRad = sort(1.1 * sqrt(-2*log(1 - 0.98*u)));   % nonstationary initial radii
    for k = 1:nSteps
        Fbg = interp1(tabs.rGrid, tabs.Bgrid, phiRad, 'linear', 'extrap');
        [RR,SS] = ndgrid(phiRad,phiRad);
        Kmat = interp2(tabs.rGrid, tabs.rGrid, tabs.Kself, SS, RR, 'linear', 0);
        Fphi = mean(Kmat,2);
        phiRad = max(phiRad + dt*(-Fbg + Fphi), 0);
        phiRad = sort(phiRad);
    end
end

function Q = sourceQ2D(rho,a,r,m)
% Q_{a,r} for omega_m = m/(2 pi) exp(-|x|^2/2) in d=2 (eq. (6.10) / Fourier form):
%   Q(rho) = (gamma_{2,a}/gamma_{2,r}) (m/2pi) int_0^inf k^{1-beta} e^{-k^2/2} J_0(k rho) dk
%          = (gamma_{2,a}/gamma_{2,r}) (m/2pi) 2^{-beta/2} Gamma(alpha) 1F1(alpha;1;-rho^2/2),
%   beta = a-r, alpha = 1-beta/2.  1F1 by Kummer's series (z<40) or its large-z expansion.
    bt = a - r;  al = 1 - bt/2;
    gr = 2^bt*gamma((3+a)/2)*gamma(-(1+r)/2)/(gamma(-(1+a)/2)*gamma((3+r)/2));   % gamma_{2,a}/gamma_{2,r}
    z = rho(:).^2/2;
    F = zeros(size(z));
    small = z < 40;
    % Kummer transformation: 1F1(al;1;-z) = e^{-z} 1F1(1-al;1;z), series with positive terms
    zs = z(small); term = ones(size(zs)); acc = ones(size(zs));
    for k = 0:200
        term = term.*(1-al+k).*zs/((k+1)^2);
        acc = acc + term;
        if all(abs(term) <= 1e-17*abs(acc)), break; end
    end
    F(small) = exp(-zs).*acc;
    % large z: 1F1(al;1;-z) ~ z^{-al}/Gamma(1-al) sum_k (al)_k^2/(k! z^k)
    zl = z(~small); term = ones(size(zl)); acc = ones(size(zl));
    for k = 0:40
        term = term.*((al+k)^2)./((k+1)*zl);
        acc = acc + term;
        if all(abs(term) <= 1e-16*abs(acc)), break; end
    end
    F(~small) = zl.^(-al)/gamma(1-al).*acc;
    Q = gr*(m/(2*pi))*2^(-bt/2)*gamma(al)*F;
    Q = reshape(Q,size(rho));
end

function rhoE = interiorMesh2D(R,IntDec)
% Interior radial cell edges on [0,R], geometrically graded towards rho = R:
% R - rho runs over seven decades with IntDec cells per decade.
    g = 10.^(-(0:7*IntDec)'/IntDec);
    rhoE = unique([0; R*(1-g(2:end)); R]);
end

function [phi, info] = greenReconstruction2D(R,a,r,m,rhoE,NeDec,Ymax,nAng)
% Disk formula (7.11)-(7.12) of Proposition 7.3 with all constants analytic:
%   phi_R = ( Q_{a,r} - c_{2,r} (-Delta)^s H_R )|_{B_R},  s = (1+r)/2,
%   H_R(y) = c_{2,r}^{-1} int_{B_R^c} G_R^s(y,z) Q_{a,r}(z) dz,  H_R = 0 in B_R.
% As in one dimension, H_R = 0 in B_R gives (-Delta)^s H_R(x) = -C_{2,s} int_{B_R^c} H_R(y)|x-y|^{-2-2s} dy
% for x in B_R, the constant c_{2,r} cancels, and with W := c_{2,r} H_R = int G Q dz
%   phi_R(x) = Q(x) + C_{2,s} int_{B_R^c} W(y)|x-y|^{-2-2s} dy,  W(y) = int_{B_R^c} G_R^s(y,z) Q(z) dz,
%   G_R^s(y,z) = kappa_{2,s}|y-z|^{2s-2} int_0^{eta_R} t^{s-1}(1+t)^{-1} dt,  s=(1+r)/2.
% rhoE: interior cell edges (0 = rhoE(1) < ... < rhoE(end) = R); phi at the midpoints.
    s = (1+r)/2;  bt = a - r;
    kappa = 1/(4^s*pi*gamma(s)^2);                       % kappa_{2,s}
    Cfl   = 4^s*gamma(1+s)/(pi*abs(gamma(-s)));           % C_{2,s}
    Bc    = beta(s,1-s);
    boggio = @(xx) Bc*betainc(min(max(xx,0),1), s, 1-s);
    % angular rule on [0,pi], graded at theta=0:  theta = pi u^5
    [ug, wg] = gaussLegendre01(nAng);
    th = pi*ug.^5;  wth = 2*pi*5*ug.^4.*wg;               % factor 2: both half circles
    sh2 = sin(th/2).^2;                                  % d^2 = (rho-eta)^2 + 4 rho eta sin^2(theta/2)
    % exterior radial mesh
    tmin = 1e-7*R;  Te = Ymax - R;
    K = max(4, round(NeDec*log10(Te/tmin)));
    zE = R + [0; tmin*(Te/tmin).^((0:K)'/K)];
    zl = zE(1:end-1); zu = zE(2:end); zc = 0.5*(zl+zu); dz = zu - zl; Ne = numel(zc);
    Qc = sourceQ2D(zc,a,r,m);
    % ---- step 1: W at exterior midpoints
    [Y,Z] = ndgrid(zc,zc);  Nn = (Y.^2-R^2).*(Z.^2-R^2);
    KG = zeros(Ne,Ne);
    for l = 1:nAng
        d2 = (Y-Z).^2 + 4*Y.*Z*sh2(l);
        KG = KG + wth(l)*kappa*d2.^(s-1).*boggio(Nn./(Nn + R^2*d2));
    end
    W0 = KG*(Qc.*zc.*dz);
    % tail of the z-integral: z = Ymax (1-s)^{-p}, integrand ~ z^{-(2-2s)-(2-bt)+1} = z^{a-2}... use p = 1/(1-a)
    pz = 1/(1-a);  [sg, wq] = gaussLegendre01(48);
    zt = Ymax*(1-sg).^(-pz);  dzt = Ymax*pz*(1-sg).^(-pz-1).*wq;
    [Yt,Zt] = ndgrid(zc,zt);  Nt = (Yt.^2-R^2).*(Zt.^2-R^2);
    KGt = zeros(size(Yt));
    for l = 1:nAng
        d2 = (Yt-Zt).^2 + 4*Yt.*Zt*sh2(l);
        KGt = KGt + wth(l)*kappa*d2.^(s-1).*boggio(Nt./(Nt + R^2*d2));
    end
    W = W0 + KGt*(sourceQ2D(zt,a,r,m).*zt.*dzt);
    % ---- step 2: phi at interior midpoints
    rc = 0.5*(rhoE(1:end-1)+rhoE(2:end)); nI = numel(rc);
    [X,Zi] = ndgrid(rc,zc);
    K2 = zeros(nI,Ne);
    for l = 1:nAng
        d2 = (X-Zi).^2 + 4*X.*Zi*sh2(l);
        K2 = K2 + wth(l)*d2.^(-1-s);
    end
    corr0 = K2*(W0.*zc.*dz);
    corr  = K2*(W.*zc.*dz);
    % tail of the y-integral: W(y) ~ W_end (y/zc_end)^{a-1}; integrand ~ y^{a-1-2-2s+1} = y^{a-2-2s}
    gam = 1 - a;  py = 1/(gam + 1 + 2*s - 1);   % exponent of decay = gam+2s ... integrand y^{-(gam+2s+1)} dy -> p = 1/(gam+2s)
    py = 1/(gam + 2*s);
    yt = Ymax*(1-sg).^(-py);  dyt = Ymax*py*(1-sg).^(-py-1).*wq;
    Wt = W(end)*(yt/zc(end)).^(-gam);
    [Xt,Yt2] = ndgrid(rc,yt);
    K2t = zeros(size(Xt));
    for l = 1:nAng
        d2 = (Xt-Yt2).^2 + 4*Xt.*Yt2*sh2(l);
        K2t = K2t + wth(l)*d2.^(-1-s);
    end
    tailPhi = K2t*(Wt.*yt.*dyt);
    Qi = sourceQ2D(rc,a,r,m);
    phi       = Qi + Cfl*(corr + tailPhi);
    phiNoTail = Qi + Cfl*corr0;
    % ---- masses with the boundary weight (R-rho)^{-s} integrated exactly
    tu = R - rhoE(1:end-1); tl = R - rhoE(2:end);          % t = R - rho on each cell
    cellW = 2*pi*( R*(tu.^(1-s)-tl.^(1-s))/(1-s) - (tu.^(2-s)-tl.^(2-s))/(2-s) );
    nu = phi.*(R-rc).^s.*cellW;                              % ring masses
    info = struct();
    info.rc = rc; info.zc = zc; info.W = W; info.Ne = Ne; info.nu = nu;
    info.rawMass = sum(nu); info.negMass = sum(max(-nu,0));
    info.rawMassNoTail = sum(phiNoTail.*(R-rc).^s.*cellW);
    info.phiNoTail = phiNoTail;
    % boundary exponent fit on 1e-5 < (R-rho)/R < 1e-2
    msk = (R-rc)/R > 1e-5 & (R-rc)/R < 1e-2 & phi > 0;
    p = polyfit(log(R-rc(msk)), log(phi(msk)), 1); info.boundaryExponent = p(1);
    info.constants = struct('kappa',kappa,'Cfl',Cfl,'s',s);
end
function x = projectedLeastSquares(A,b,nIter)
% Fallback for nonnegative least squares if lsqnonneg is unavailable.
    x = max(A\b,0);
    L = norm(A)^2 + 1e-12;
    tau = 0.9/L;
    for it = 1:nIter
        x = max(x - tau*A'*(A*x-b), 0);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions: refinement study
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function logf(fid, varargin)
    fprintf(varargin{:});
    fprintf(fid, varargin{:});
end

function F = cdfAt(edges, cdf, xq)
% Piecewise linear CDF with values 0 left of the support and 1 right of it.
    F = interp1(edges, cdf, xq, 'linear');
    F(xq <= edges(1)) = 0;
    F(xq >= edges(end)) = 1;
end

function out = writeScan1D(scanL,scanRms,Lstar,rmsStar,LRiesz,fid,ftex)
    logf(fid,'\n[Part I] 1D zero-flux (7.4): RMS force residual versus prescribed half-width L (n=320, 240 test points)\n');
    logf(fid,'%10s %14s\n','L','RMS resid');
    for k = 1:numel(scanL), logf(fid,'%10.4f %14.4e\n', scanL(k), scanRms(k)); end
    logf(fid,'minimizer L = %.8f with RMS residual %.4e; L from the mass condition of the differentiated form (7.2): %.8f\n', Lstar, rmsStar, LRiesz);
    out = struct('L',scanL,'rms',scanRms,'Lstar',Lstar,'rmsStar',rmsStar,'LRiesz',LRiesz);
    fprintf(ftex,'%% Table: 1D scan\n\\newcommand{\\tableScanOneD}{%%\n\\begin{tabular}{lc@{\\hspace{2em}}lc@{\\hspace{2em}}lc}\n\\toprule\n$L$ & RMS residual & $L$ & RMS residual & $L$ & RMS residual\\\\\n\\midrule\n');
    nr = ceil(numel(scanL)/3);
    for k = 1:nr
        row = {};
        for c = 0:2
            j = k + c*nr;
            if j <= numel(scanL), row{end+1} = sprintf('%.4f & %s', scanL(j), texnum(scanRms(j))); else, row{end+1} = ' & '; end
        end
        fprintf(ftex,'%s\\\\\n', strjoin(row,' & '));
    end
    fprintf(ftex,'\\midrule\n$L^*=%.6f$ & %s & \\multicolumn{4}{l}{$L$ from the mass condition of (7.2): %.6f}\\\\\n\\bottomrule\n\\end{tabular}}\n\n', Lstar, texnum(rmsStar), LRiesz);
end

function out = writeScan2D(scanR,scanRms,scanOuter,Rstar,rmsStar,fid,ftex)
    logf(fid,'\n[Part II] 2D zero-flux (7.13): RMS force residual versus prescribed radius R (nRing=110, nTest=150, q=160)\n');
    logf(fid,'%10s %14s %14s\n','R','RMS resid','mass in (0.9R,R]');
    for k = 1:numel(scanR), logf(fid,'%10.4f %14.4e %14.6f\n', scanR(k), scanRms(k), scanOuter(k)); end
    logf(fid,'minimizer R = %.6f with RMS residual %.4e\n', Rstar, rmsStar);
    out = struct('R',scanR,'rms',scanRms,'outer',scanOuter,'Rstar',Rstar,'rmsStar',rmsStar);
    fprintf(ftex,'%% Table: 2D scan\n\\newcommand{\\tableScan}{%%\n\\begin{tabular}{lcc@{\\hspace{2em}}lcc}\n\\toprule\n$R$ & RMS residual & mass in $(0.9R,R]$ & $R$ & RMS residual & mass in $(0.9R,R]$\\\\\n\\midrule\n');
    idx = 1:2:numel(scanR);            % every 0.005 in the table
    nr = ceil(numel(idx)/2);
    for k = 1:nr
        row = {};
        for c = 0:1
            j = k + c*nr;
            if j <= numel(idx), jj = idx(j); row{end+1} = sprintf('%.3f & %s & %.4f', scanR(jj), texnum(scanRms(jj)), scanOuter(jj)); else, row{end+1} = ' & & '; end
        end
        fprintf(ftex,'%s\\\\\n', strjoin(row,' & '));
    end
    fprintf(ftex,'\\midrule\n$R^*=%.5f$ & %s & \\multicolumn{4}{l}{}\\\\\n\\bottomrule\n\\end{tabular}}\n\n', Rstar, texnum(rmsStar));
end

function out = refine1DZeroFlux(a,r,m,nList,massW,fid,ftex)
    logf(fid,'\n[III.1] 1D zero-flux (7.4): cell refinement (L = residual minimizer at each n; nTest = 3n/4)\n');
    logf(fid,'%8s %14s %14s %12s %12s %12s %10s\n','n','L_n','L_n (7.2)','RMS resid','max resid','phi_n(0)','time[s]');
    nL = numel(nList);
    L = zeros(nL,1); LR = zeros(nL,1); rms = zeros(nL,1); mx = zeros(nL,1); phi0 = zeros(nL,1); tim = zeros(nL,1);
    E = cell(nL,1); C = cell(nL,1);
    Lprev = 0.2727;
    for k = 1:nL
        n = nList(k); nT = round(0.75*n); t0 = tic;
        f = @(LL) rmsZeroFlux1D(LL,a,r,m,n,nT,massW);
        L(k) = fminbnd(f, Lprev-0.004, Lprev+0.004, optimset('TolX',1e-7));
        [phi,E{k},res] = solveZeroFlux1D(L(k),a,r,m,n,nT,massW);
        rms(k) = sqrt(mean(res.^2)); mx(k) = max(abs(res));
        C{k} = [0; cumsum(phi)*(E{k}(2)-E{k}(1))]; C{k} = C{k}/C{k}(end);
        phi0(k) = 0.5*(phi(n/2)+phi(n/2+1));
        LR(k) = fzero(@(LL) stationaryMassDefect1D(LL,a,r,m,n), [0.16,0.45]);
        tim(k) = toc(t0);
        logf(fid,'%8d %14.10f %14.10f %12.3e %12.3e %12.8f %10.1f\n', n, L(k), LR(k), rms(k), mx(k), phi0(k), tim(k));
        Lprev = L(k);
    end
    xq = linspace(-0.3,0.3,6001)';
    Fref = cdfAt(E{end},C{end},xq);
    cdfErr = zeros(nL,1);
    for k = 1:nL, cdfErr(k) = max(abs(cdfAt(E{k},C{k},xq)-Fref)); end
    logf(fid,'errors against the finest grid n=%d:\n', nList(end));
    logf(fid,'%8s %14s %14s %14s\n','n','|L_n-L_ref|','maxCDFerr','|phi_n(0)-phi_ref(0)|');
    for k = 1:nL-1
        logf(fid,'%8d %14.3e %14.3e %14.3e\n', nList(k), abs(L(k)-L(end)), cdfErr(k), abs(phi0(k)-phi0(end)));
    end
    out = struct('n',nList(:),'L',L,'LRiesz',LR,'rms',rms,'max',mx,'phi0',phi0,'cdfErr',cdfErr,'time',tim);
    fprintf(ftex,'%% Table: 1D zero-flux refinement\n\\newcommand{\\tableZeroFlux}{%%\n\\begin{tabular}{rllcccc}\n\\toprule\n');
    fprintf(ftex,'$n$ & $L_n$ & $L_n$ from (7.2) & RMS residual & $\\phi_n(0)$ & $|L_n-L_{%d}|$ & $\\max_x|F_n-F_{%d}|$\\\\\n\\midrule\n', nList(end), nList(end));
    for k = 1:nL
        if k < nL
            fprintf(ftex,'%d & %.7f & %.7f & %s & %.6f & %s & %s\\\\\n', nList(k), L(k), LR(k), texnum(rms(k)), phi0(k), texnum(abs(L(k)-L(end))), texnum(cdfErr(k)));
        else
            fprintf(ftex,'%d & %.7f & %.7f & %s & %.6f & -- & --\\\\\n', nList(k), L(k), LR(k), texnum(rms(k)), phi0(k));
        end
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function out = refine1DGreen(L,a,r,m,n0,nTest0,massW,NeDec0,Ymax0,nListG,Lzf,fid,ftex)
    [phiR,edges] = solveZeroFlux1D(L,a,r,m,n0,nTest0,massW);
    h0 = edges(2)-edges(1);
    cdfR = [0; cumsum(phiR)*h0]; cdfR = cdfR/cdfR(end);

    % (a) exterior resolution
    NeDecList = [5 10 20 40 80 160];
    logf(fid,'\n[III.2a] 1D Green kernel (7.3): exterior cells per decade (n=%d, Ymax=%.0e)\n', n0, Ymax0);
    logf(fid,'%8s %8s %14s %12s %12s %12s %10s\n','Ne/dec','Ne','raw mass','neg. mass','maxCDFdisc','maxDensDisc','time[s]');
    nA = numel(NeDecList); massNe = zeros(nA,1); cdfErrNe = zeros(nA,1); densErrNe = zeros(nA,1); NeA = zeros(nA,1); negA = zeros(nA,1); timA = zeros(nA,1);
    for k = 1:nA
        t0 = tic;
        [phiG, info] = greenReconstruction1D(L,a,r,m,n0,NeDecList(k),Ymax0);
        timA(k) = toc(t0);
        [cdfErrNe(k), densErrNe(k)] = greenMetrics(phiG,phiR,cdfR,h0);
        massNe(k) = info.rawMass; NeA(k) = info.Ne; negA(k) = info.negMass;
        logf(fid,'%8d %8d %14.8f %12.3e %12.3e %12.3e %10.2f\n', NeDecList(k), info.Ne, info.rawMass, info.negMass, cdfErrNe(k), densErrNe(k), timA(k));
    end
    % (b) truncation radius
    YmaxList = [1e1 1e2 1e3 1e4 1e5 1e6];
    logf(fid,'\n[III.2b] 1D Green kernel (7.3): exterior truncation Ymax (n=%d, %d cells/decade); "no tail" drops both tail terms\n', n0, NeDec0);
    logf(fid,'%10s %8s %14s %12s %14s %12s\n','Ymax','Ne','raw mass','maxCDFdisc','raw mass(noT)','maxCDF(noT)');
    nB = numel(YmaxList); massY = zeros(nB,1); cdfErrY = zeros(nB,1); massYnt = zeros(nB,1); cdfErrYnt = zeros(nB,1); NeB = zeros(nB,1);
    for k = 1:nB
        [phiG, info] = greenReconstruction1D(L,a,r,m,n0,NeDec0,YmaxList(k));
        [cdfErrY(k), ~] = greenMetrics(phiG,phiR,cdfR,h0);
        [cdfErrYnt(k), ~] = greenMetrics(info.phiNoTail,phiR,cdfR,h0);
        massY(k) = info.rawMass; massYnt(k) = info.rawMassNoTail; NeB(k) = info.Ne;
        logf(fid,'%10.0e %8d %14.8f %12.3e %14.8f %12.3e\n', YmaxList(k), info.Ne, massY(k), cdfErrY(k), massYnt(k), cdfErrYnt(k));
    end
    % (c) interior cells at fixed L
    logf(fid,'\n[III.2c] 1D Green kernel (7.3) vs zero-flux state (7.4) on the same grid: interior cells n, L = L_n of [III.1] (%d cells/decade, Ymax=%.0e)\n', NeDec0, Ymax0);
    logf(fid,'%8s %12s %8s %14s %12s %12s %12s %10s\n','n','L_n','Ne','raw mass','neg. mass','maxCDFdisc','maxDensDisc','time[s]');
    nC = numel(nListG); massN = zeros(nC,1); cdfErrN = zeros(nC,1); densErrN = zeros(nC,1); NeC = zeros(nC,1); timC = zeros(nC,1); negC = zeros(nC,1);
    for k = 1:nC
        n = nListG(k); t0 = tic;
        [phiRn,edg] = solveZeroFlux1D(Lzf(k),a,r,m,n,round(0.75*n),massW);
        hn = edg(2)-edg(1); cdfRn = [0; cumsum(phiRn)*hn]; cdfRn = cdfRn/cdfRn(end);
        [phiG, info] = greenReconstruction1D(Lzf(k),a,r,m,n,NeDec0,Ymax0);
        timC(k) = toc(t0);
        [cdfErrN(k), densErrN(k)] = greenMetrics(phiG,phiRn,cdfRn,hn);
        massN(k) = info.rawMass; NeC(k) = info.Ne; negC(k) = info.negMass;
        logf(fid,'%8d %12.8f %8d %14.8f %12.3e %12.3e %12.3e %10.2f\n', n, Lzf(k), info.Ne, massN(k), negC(k), cdfErrN(k), densErrN(k), timC(k));
    end
    out = struct('NeDec',NeDecList(:),'Ne',NeA,'massNe',massNe,'cdfErrNe',cdfErrNe,'densErrNe',densErrNe, ...
                 'Ymax',YmaxList(:),'massY',massY,'cdfErrY',cdfErrY,'massYnoTail',massYnt,'cdfErrYnoTail',cdfErrYnt, ...
                 'n',nListG(:),'massN',massN,'cdfErrN',cdfErrN,'densErrN',densErrN);
    % LaTeX tables
    fprintf(ftex,'%% Table: 1D Green (7.3), exterior resolution\n\\newcommand{\\tableGreenNe}{%%\n\\begin{tabular}{rrlccc}\n\\toprule\n');
    fprintf(ftex,'cells/decade & $N_e$ & $M_G$ & $M_-$ & $\\max|F_G-F_I|$ & $\\max|\\phi_G-\\phi_I|/\\max\\phi_I$\\\\\n\\midrule\n');
    for k = 1:nA
        fprintf(ftex,'%d & %d & %.8f & %s & %s & %s\\\\\n', NeDecList(k), NeA(k), massNe(k), texnum(negA(k)), texnum(cdfErrNe(k)), texnum(densErrNe(k)));
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
    fprintf(ftex,'%% Table: 1D Green (7.3), truncation\n\\newcommand{\\tableGreenY}{%%\n\\begin{tabular}{lrlclc}\n\\toprule\n');
    fprintf(ftex,'$Y_{\\max}$ & $N_e$ & $M_G$ & $\\max|F_G-F_I|$ & $M_G$ (no tail) & $\\max|F_G-F_I|$ (no tail)\\\\\n\\midrule\n');
    for k = 1:nB
        fprintf(ftex,'$10^{%d}$ & %d & %.8f & %s & %.6f & %s\\\\\n', round(log10(YmaxList(k))), NeB(k), massY(k), texnum(cdfErrY(k)), massYnt(k), texnum(cdfErrYnt(k)));
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
    fprintf(ftex,'%% Table: 1D Green (7.3), interior cells\n\\newcommand{\\tableGreenN}{%%\n\\begin{tabular}{rlrlcc}\n\\toprule\n');
    fprintf(ftex,'$n$ & $L_n$ & $N_e$ & $M_G$ & $\\max|F_G-F_I|$ & $\\max|\\phi_G-\\phi_I|/\\max\\phi_I$\\\\\n\\midrule\n');
    for k = 1:nC
        fprintf(ftex,'%d & %.7f & %d & %.8f & %s & %s\\\\\n', nListG(k), Lzf(k), NeC(k), massN(k), texnum(cdfErrN(k)), texnum(densErrN(k)));
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function [cdfErr, densErr] = greenMetrics(phiG,phiR,cdfR,h)
% cdfErr : max |F_G - F_h| on the cell edges (both CDFs normalized);
% densErr: max |phi_G - phi_h| / max phi_h over the central 90% of the cells.
    cdfG = [0; cumsum(max(phiG,0))*h]; cdfG = cdfG/cdfG(end);
    cdfErr = max(abs(cdfG-cdfR));
    n = numel(phiR); k0 = round(0.05*n);
    ii = (k0+1):(n-k0);
    densErr = max(abs(phiG(ii)-phiR(ii)))/max(phiR);
end

function out = refine1DParticles(a,r,m,L0,T,edgesRef,cdfRef,Lstat,fid,ftex)
    NList = [125 250 500 1000 2000];
    logf(fid,'\n[III.3a] 1D particles: number of particles N (dt=0.02, T=%.0f), error against the zero-flux CDF (7.4) (n=%d)\n', T, numel(edgesRef)-1);
    logf(fid,'%8s %12s %14s %14s %10s\n','N','maxCDFerr','final diam','diam-2L','time[s]');
    nN = numel(NList); errN = zeros(nN,1); diamN = zeros(nN,1); timN = zeros(nN,1);
    for k = 1:nN
        t0 = tic;
        [X,~,diam,~] = evolveParticles1D(NList(k),0.02,T,L0,a,r,m,12);
        timN(k) = toc(t0);
        emp = arrayfun(@(z) sum(X <= z)/NList(k), edgesRef);
        errN(k) = max(abs(emp(:)-cdfRef(:))); diamN(k) = diam(end);
        logf(fid,'%8d %12.3e %14.8f %14.3e %10.1f\n', NList(k), errN(k), diamN(k), diamN(k)-2*Lstat, timN(k));
    end
    dtList = [0.08 0.04 0.02 0.01];
    logf(fid,'\n[III.3b] 1D particles: time step dt (N=500, T=%.0f)\n', T);
    logf(fid,'%8s %12s %14s %14s %10s\n','dt','maxCDFerr','final diam','diam-2L','time[s]');
    nD = numel(dtList); errDt = zeros(nD,1); diamDt = zeros(nD,1); timDt = zeros(nD,1);
    for k = 1:nD
        t0 = tic;
        [X,~,diam,~] = evolveParticles1D(500,dtList(k),T,L0,a,r,m,12);
        timDt(k) = toc(t0);
        emp = arrayfun(@(z) sum(X <= z)/500, edgesRef);
        errDt(k) = max(abs(emp(:)-cdfRef(:))); diamDt(k) = diam(end);
        logf(fid,'%8.3f %12.3e %14.8f %14.3e %10.1f\n', dtList(k), errDt(k), diamDt(k), diamDt(k)-2*Lstat, timDt(k));
    end
    TList = [50 100 150 200];
    logf(fid,'\n[III.3c] 1D particles: terminal time T (N=500, dt=0.02)\n');
    logf(fid,'%8s %12s %14s %14s %10s\n','T','maxCDFerr','final diam','diam-2L','time[s]');
    nT = numel(TList); errT = zeros(nT,1); diamT = zeros(nT,1);
    for k = 1:nT
        t0 = tic;
        [X,~,diam,~] = evolveParticles1D(500,0.02,TList(k),L0,a,r,m,12);
        emp = arrayfun(@(z) sum(X <= z)/500, edgesRef);
        errT(k) = max(abs(emp(:)-cdfRef(:))); diamT(k) = diam(end);
        logf(fid,'%8d %12.3e %14.8f %14.3e %10.1f\n', TList(k), errT(k), diamT(k), diamT(k)-2*Lstat, toc(t0));
    end
    out = struct('N',NList(:),'cdfErrN',errN,'diamN',diamN,'dt',dtList(:),'cdfErrDt',errDt,'diamDt',diamDt,'T',TList(:),'cdfErrT',errT,'diamT',diamT);
    fprintf(ftex,'%% Table: 1D particles\n\\newcommand{\\tablePartOneD}{%%\n\\begin{tabular}{rcc@{\\hspace{1.5em}}rcc@{\\hspace{1.5em}}rcc}\n\\toprule\n');
    fprintf(ftex,'$N$ & $E_{\\rm CDF}$ & $D(T)-2L$ & $\\Delta t$ & $E_{\\rm CDF}$ & $D(T)-2L$ & $T$ & $E_{\\rm CDF}$ & $D(T)-2L$\\\\\n\\midrule\n');
    for k = 1:max([nN,nD,nT])
        if k <= nN, s1 = sprintf('%d & %s & %s', NList(k), texnum(errN(k)), texnum(diamN(k)-2*Lstat)); else, s1 = ' & & '; end
        if k <= nD, s2 = sprintf('%.2f & %s & %s', dtList(k), texnum(errDt(k)), texnum(diamDt(k)-2*Lstat)); else, s2 = ' & & '; end
        if k <= nT, s3 = sprintf('%d & %s & %s', TList(k), texnum(errT(k)), texnum(diamT(k)-2*Lstat)); else, s3 = ' & & '; end
        fprintf(ftex,'%s & %s & %s\\\\\n', s1, s2, s3);
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function out = refine2DRing(R,a,r,m,nBg0,Lbg,massW,rPlot,fid,ftex)
    levels = [55 75 80 275; 110 150 160 550; 220 300 320 1100; 440 600 640 2200];
    nLev = size(levels,1);
    logf(fid,'\n[III.4b] 2D ring masses (7.13): refinement of (nRing, nTest, nTheta, nBg) at R=%.2f\n', R);
    logf(fid,'%6s %6s %6s %7s %6s %12s %12s %12s %12s %10s\n','level','nRing','nTest','nTheta','nBg','RMS resid','max resid','sup vs prev','W1 vs prev','time[s]');
    cdfs = cell(nLev,1); rms = zeros(nLev,1); mx = zeros(nLev,1); dprev = zeros(nLev,1); wprev = zeros(nLev,1); tim = zeros(nLev,1);
    for k = 1:nLev
        t0 = tic;
        [mu,sRing,~,resid] = solveRingMasses2D(R,a,r,m,levels(k,1),levels(k,2),levels(k,3),levels(k,4),Lbg,massW);
        tim(k) = toc(t0);
        cdfs{k} = cumulativeRingMass(mu,sRing,rPlot);
        rms(k) = sqrt(mean(resid.^2)); mx(k) = max(abs(resid));
        if k > 1
            dprev(k) = max(abs(cdfs{k}-cdfs{k-1}));
            wprev(k) = trapz(rPlot, abs(cdfs{k}-cdfs{k-1}));
        end
        logf(fid,'%6d %6d %6d %7d %6d %12.4e %12.4e %12.4e %12.4e %10.1f\n', k, levels(k,1), levels(k,2), levels(k,3), levels(k,4), rms(k), mx(k), dprev(k), wprev(k), tim(k));
    end
    dfin = zeros(nLev,1); wfin = zeros(nLev,1);
    for k = 1:nLev-1
        dfin(k) = max(abs(cdfs{k}-cdfs{end}));
        wfin(k) = trapz(rPlot, abs(cdfs{k}-cdfs{end}));
    end
    logf(fid,'against the finest level: sup %s   W1 %s\n', mat2str(dfin(1:end-1)',3), mat2str(wfin(1:end-1)',3));
    out = struct('level',(1:nLev)','levels',levels,'rmsRes',rms,'maxRes',mx,'cdfDiffPrev',dprev,'w1DiffPrev',wprev,'cdfDiffFinest',dfin,'w1DiffFinest',wfin,'time',tim);
    fprintf(ftex,'%% Table: 2D ring refinement\n\\newcommand{\\tableRing}{%%\n\\begin{tabular}{ccccccccc}\n\\toprule\n');
    fprintf(ftex,'level & $n_{\\rm ring}$ & $n_{\\rm test}$ & $q$ & $n_{\\rm bg}$ & RMS residual & max residual & $\\sup_\\rho|F_\\ell-F_{\\ell-1}|$ & $W_1(F_\\ell,F_{\\ell-1})$\\\\\n\\midrule\n');
    for k = 1:nLev
        if k == 1, sd = '--'; sw = '--'; else, sd = texnum(dprev(k)); sw = texnum(wprev(k)); end
        fprintf(ftex,'%d & %d & %d & %d & %d & %s & %s & %s & %s\\\\\n', k, levels(k,1), levels(k,2), levels(k,3), levels(k,4), texnum(rms(k)), texnum(mx(k)), sd, sw);
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function out = refine2DGreen(R,a,r,m,NeDec0,Ymax0,nAng0,IntDec0,rPlot,cdfRef,rTest,bvec,nTheta,fid,ftex)
% Convergence of the analytic disk formula (7.11)-(7.12): exterior cells per decade,
% truncation, angular nodes, interior cells per decade.  Compared with the ring-mass
% CDF in W1 and through the force residual of (7.13).
    cases = {'exterior cells/decade', [10 20 40 80], 1; '$Y_{\max}$', [1e2 1e3 1e4 1e5 1e6], 2; ...
             'angular nodes', [16 32 64 128], 3; 'interior cells/decade', [10 20 40 80], 4};
    logf(fid,'\n[III.4c] 2D Green-kernel disk formula (7.11): refinement (base: %d ext. cells/decade, Ymax=%.0e, %d angular nodes, %d int. cells/decade)\n', NeDec0, Ymax0, nAng0, IntDec0);
    logf(fid,'%-22s %10s %6s %6s %10s %10s %10s %10s %10s %8s\n','parameter','value','Ne','nI','raw mass','noTail','expo','W1 vs ring','force RMS','time[s]');
    fprintf(ftex,'%% Table: 2D Green refinement\n\\newcommand{\\tableGreenTwoD}{%%\n\\begin{tabular}{llrrlllcc}\n\\toprule\n');
    fprintf(ftex,'parameter & value & $N_e$ & $n_I$ & $M_G$ & $M_G$ (no tail) & exponent & $W_1$ & RMS residual\\\\\n\\midrule\n');
    out = struct('name',{{}},'value',{{}},'mass',{{}},'w1',{{}},'res',{{}});
    for c = 1:size(cases,1)
        vals = cases{c,2}; nm = cases{c,1};
        for v = vals
            p = [NeDec0 Ymax0 nAng0 IntDec0]; p(cases{c,3}) = v;
            t0 = tic;
            rhoE = interiorMesh2D(R,p(4));
            [phi,info] = greenReconstruction2D(R,a,r,m,rhoE,p(1),p(2),p(3));
            nu = info.nu/sum(info.nu);
            cdfG = cumulativeRingMass(nu,info.rc,rPlot);
            w1 = trapz(rPlot,abs(cdfG-cdfRef));
            A2 = ringKernelMatrix(rTest, info.rc, r, nTheta);
            res = sqrt(mean((A2*nu-bvec).^2));
            tm = toc(t0);
            logf(fid,'%-22s %10.4g %6d %6d %10.6f %10.6f %10.3f %10.3e %10.3e %8.1f\n', nm, v, info.Ne, numel(phi), info.rawMass, info.rawMassNoTail, info.boundaryExponent, w1, res, tm);
            if cases{c,3} == 2, vs = sprintf('$10^{%d}$', round(log10(v))); else, vs = sprintf('%d', v); end
            fprintf(ftex,'%s & %s & %d & %d & %.5f & %.5f & %.3f & %s & %s\\\\\n', nm, vs, info.Ne, numel(phi), info.rawMass, info.rawMassNoTail, info.boundaryExponent, texnum(w1), texnum(res));
            out.name{end+1} = nm; out.value{end+1} = v; out.mass{end+1} = info.rawMass; out.w1{end+1} = w1; out.res{end+1} = res;
        end
        if c < size(cases,1), fprintf(ftex,'\\midrule\n'); end
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function out = refine2DParticles(a,r,m,nBg,Lbg,rPlot,cdfRef,fid,ftex)
    NList = [90 180 360 720];
    logf(fid,'\n[III.5a] 2D radial particles: number of nodes N (dt=0.05, T=50), error against the ring-mass CDF (base level)\n');
    logf(fid,'%8s %12s %12s %14s %10s\n','N','sup|dF|','W1','max radius','time[s]');
    tabs = [];
    nN = numel(NList); errN = zeros(nN,1); w1N = zeros(nN,1); rmaxN = zeros(nN,1); timN = zeros(nN,1);
    for k = 1:nN
        t0 = tic;
        [phiRad, tabs] = evolveRadial2D(NList(k),0.05,1000,a,r,m,nBg,Lbg,tabs);
        timN(k) = toc(t0);
        cdfDyn = arrayfun(@(rr) mean(phiRad <= rr), rPlot);
        errN(k) = max(abs(cdfDyn-cdfRef)); w1N(k) = trapz(rPlot,abs(cdfDyn-cdfRef)); rmaxN(k) = max(phiRad);
        logf(fid,'%8d %12.3e %12.3e %14.6f %10.1f\n', NList(k), errN(k), w1N(k), rmaxN(k), timN(k));
    end
    dtList = [0.1 0.05 0.025];
    logf(fid,'\n[III.5b] 2D radial particles: time step dt (N=180, T=50)\n');
    logf(fid,'%8s %12s %12s %14s %10s\n','dt','sup|dF|','W1','max radius','time[s]');
    nD = numel(dtList); errDt = zeros(nD,1); w1Dt = zeros(nD,1); rmaxDt = zeros(nD,1);
    for k = 1:nD
        t0 = tic;
        [phiRad, tabs] = evolveRadial2D(180,dtList(k),round(50/dtList(k)),a,r,m,nBg,Lbg,tabs);
        cdfDyn = arrayfun(@(rr) mean(phiRad <= rr), rPlot);
        errDt(k) = max(abs(cdfDyn-cdfRef)); w1Dt(k) = trapz(rPlot,abs(cdfDyn-cdfRef)); rmaxDt(k) = max(phiRad);
        logf(fid,'%8.3f %12.3e %12.3e %14.6f %10.1f\n', dtList(k), errDt(k), w1Dt(k), rmaxDt(k), toc(t0));
    end
    TList = [50 100 150 200];
    logf(fid,'\n[III.5c] 2D radial particles: terminal time T (N=180, dt=0.05)\n');
    logf(fid,'%8s %12s %12s %14s %10s\n','T','sup|dF|','W1','max radius','time[s]');
    nT = numel(TList); errT = zeros(nT,1); w1T = zeros(nT,1); rmaxT = zeros(nT,1);
    for k = 1:nT
        t0 = tic;
        [phiRad, tabs] = evolveRadial2D(180,0.05,round(TList(k)/0.05),a,r,m,nBg,Lbg,tabs);
        cdfDyn = arrayfun(@(rr) mean(phiRad <= rr), rPlot);
        errT(k) = max(abs(cdfDyn-cdfRef)); w1T(k) = trapz(rPlot,abs(cdfDyn-cdfRef)); rmaxT(k) = max(phiRad);
        logf(fid,'%8d %12.3e %12.3e %14.6f %10.1f\n', TList(k), errT(k), w1T(k), rmaxT(k), toc(t0));
    end
    out = struct('N',NList(:),'cdfErrN',errN,'w1N',w1N,'rmaxN',rmaxN,'dt',dtList(:),'cdfErrDt',errDt,'w1Dt',w1Dt,'rmaxDt',rmaxDt,'T',TList(:),'cdfErrT',errT,'w1T',w1T,'rmaxT',rmaxT);
    fprintf(ftex,'%% Table: 2D particles\n\\newcommand{\\tablePartTwoD}{%%\n\\begin{tabular}{rcc@{\\hspace{1.5em}}rcc@{\\hspace{1.5em}}rcc}\n\\toprule\n');
    fprintf(ftex,'$N$ & $W_1$ & $\\max_i\\rho_i(T)$ & $\\Delta t$ & $W_1$ & $\\max_i\\rho_i(T)$ & $T$ & $W_1$ & $\\max_i\\rho_i(T)$\\\\\n\\midrule\n');
    for k = 1:max([nN,nD,nT])
        if k <= nN, s1 = sprintf('%d & %s & %.5f', NList(k), texnum(w1N(k)), rmaxN(k)); else, s1 = ' & & '; end
        if k <= nD, s2 = sprintf('%.3f & %s & %.5f', dtList(k), texnum(w1Dt(k)), rmaxDt(k)); else, s2 = ' & & '; end
        if k <= nT, s3 = sprintf('%d & %s & %.5f', TList(k), texnum(w1T(k)), rmaxT(k)); else, s3 = ' & & '; end
        fprintf(ftex,'%s & %s & %s\\\\\n', s1, s2, s3);
    end
    fprintf(ftex,'\\bottomrule\n\\end{tabular}}\n\n');
end

function s = texnum(v)
% Number in LaTeX scientific notation, e.g. 1.23\times10^{-4}.
    if v == 0
        s = '0'; return;
    end
    e = floor(log10(abs(v)));
    mant = v/10^e;
    if abs(abs(mant)-10) < 5e-3, mant = mant/10; e = e+1; end
    s = sprintf('$%.2f\\times10^{%d}$', mant, e);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions: figure export
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saveFigure(fig,basename)
% Save both PDF and PNG, with a print fallback for older MATLAB versions.
    try
        exportgraphics(fig,[basename '.pdf'],'ContentType','vector');
        exportgraphics(fig,[basename '.png'],'Resolution',220);
    catch
        print(fig,[basename '.png'],'-dpng','-r220');
        print(fig,[basename '.pdf'],'-dpdf');
    end
end
