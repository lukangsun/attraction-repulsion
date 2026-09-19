% verify_T50_corrected_poisson_formula516.m
% Reproduces the numerical figures in the revised manuscript.
% Terminal times: T=50 for both the 2D and 1D experiments.
%
% Correction relative to the earlier version:
% In formula516Density1D, the exterior sigma-harmonic extension needed for
% formula (5.16) is computed with the EXTERIOR interval Poisson kernel
%
%   P_{I^c}(y,x) = sin(pi*sigma)/pi * (((y-A)*(y-B))/((x-A)*(B-x)))^sigma / |y-x|,
%
% for y outside I=[A,B] and x inside I.
%
% The previous code used the reciprocal ratio.  That is the interior-domain
% Poisson kernel and gives the wrong exterior tail.
%
% Part I: d=2, 0 <= r <= a < 1.  The stationary radial measure is
% computed in two ways:
%   (a) the radial zero-flux Galerkin system (the radial verification
%       condition), and
%   (b) a direct quadrature of the explicit reconstruction formula (6.17)
%       with R=0.33, using the exterior Green kernel and the interior
%       fractional-Laplacian singular integral.
% Both theoretical cumulative masses are compared with the long-time radial
% particle solution phi_T.
%
% Part II: d=1, a > r.  The theoretical one-dimensional stationary
% distribution is computed from the finite Riesz formula (5.15).  We also
% independently evaluate the Green/Poisson fractional-Laplacian formula
% (5.16) at L=0.2727 and plot its cumulative mass.

clear; close all; clc;

%% ========================================================================
% Part I. Two-dimensional example with d=2 and a >= r >= 0
% ========================================================================
% Background: omega_m(x) = m/(2*pi)*exp(-|x|^2/2).
% Nontrivial exponents satisfying a > r.
m = 1.80;
a = 0.60;
r = 0.20;
R = 0.33;        % selected by a coarse residual scan of the free boundary

nTheta = 160;
nRing  = 110;
nTest  = 150;
nBg    = 550;
Lbg    = 6.0;
massW  = 120.0;

sRing = ((1:nRing)' - 0.5) * R/nRing;
rTest = ((1:nTest)' - 0.5) * R/nTest;

fprintf('2D: assembling radial Galerkin discretization of the radial zero-flux system...\n');
A = ringKernelMatrix(rTest, sRing, r, nTheta);

sBg  = ((1:nBg)' - 0.5) * Lbg/nBg;
dsBg = Lbg/nBg;
% Ring mass of m/(2*pi)*exp(-|x|^2/2) in two dimensions.
wBg  = m * sBg .* exp(-0.5*sBg.^2) * dsBg;
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
fprintf('2D: stationary mass          %.15f\n', sum(mu));
fprintf('2D: stationary RMS residual  %.4e\n', sqrt(mean(resid.^2)));
fprintf('2D: stationary max residual  %.4e\n', max(abs(resid)));

% Direct reconstruction curve from formula (6.17) with R=0.33.
% This is separate from the zero-flux Galerkin curve above.
muFormula617 = reconstructFormula6172D(sRing,R,a,r,m,A,b);

% Evolve the radial particle dynamics
%   R_i' = - int A_a(R_i,s) d omega_m(s) + (1/N) sum_j A_r(R_i,R_j).
fprintf('2D: precomputing interpolation tables for phi_T evolution...\n');
Npart = 180;
rMaxGrid = 3.2;
nGrid = 260;
rGridTab = linspace(0,rMaxGrid,nGrid)';
Kself = ringKernelMatrix(rGridTab, rGridTab, r, 96);
Bgrid = ringKernelMatrix(rGridTab, sBg, a, 96) * wBg;

u = ((1:Npart)' - 0.5)/Npart;
phiRad = 1.1 * sqrt(-2*log(1 - 0.98*u));   % nonstationary initial radii
phiRad = sort(phiRad);

dt = 0.05;
nSteps = 1000;
for k = 1:nSteps
    Fbg = interp1(rGridTab, Bgrid, phiRad, 'linear', 'extrap');
    [RR,SS] = ndgrid(phiRad,phiRad);
    Kmat = interp2(rGridTab, rGridTab, Kself, SS, RR, 'linear', 0);
    Fphi = mean(Kmat,2);
    phiRad = max(phiRad + dt*(-Fbg + Fphi), 0);
    phiRad = sort(phiRad);
    if mod(k,300)==0
        fprintf('2D: step %4d/%4d, max radius %.4f\n', k, nSteps, max(phiRad));
    end
end

rPlot = linspace(0,0.66,500)';
cdfStat = zeros(size(rPlot));
cdfFormula617 = zeros(size(rPlot));
cdfDyn  = zeros(size(rPlot));
for i = 1:numel(rPlot)
    cdfStat(i) = sum(mu(sRing <= rPlot(i)));
    cdfFormula617(i) = sum(muFormula617(sRing <= rPlot(i)));
    cdfDyn(i)  = mean(phiRad <= rPlot(i));
end

% Produce a representative particle cloud from radial particles by assigning
% deterministic angles.  This is only for visualization; the dynamics are radial.
golden = pi*(3-sqrt(5));
angles = mod((0:Npart-1)'*golden, 2*pi);
Xcloud = [phiRad.*cos(angles), phiRad.*sin(angles)];

fig = figure('Color','w','Position',[80 100 1250 360]);
subplot(1,3,1);
plot(Xcloud(:,1),Xcloud(:,2),'o','MarkerSize',3); hold on;
th = linspace(0,2*pi,400);
plot(R*cos(th),R*sin(th),'--','LineWidth',1.5);
axis equal; grid on; box on;
xlim([-0.55,0.55]); ylim([-0.55,0.55]);
xlabel('x_1'); ylabel('x_2');
title('particle $\phi_T$','Interpreter','latex');

subplot(1,3,2);
plot(rPlot,cdfStat,'LineWidth',2); hold on;
plot(rPlot,cdfFormula617,'-.','LineWidth',2);
stairs(sort(phiRad), (1:Npart)'/Npart, '--', 'LineWidth',2);
xline(R,':','LineWidth',1.5);
grid on; box on;
xlim([0,0.66]); ylim([0,1.05]);
xlabel('radius'); ylabel('cum. mass');
title(sprintf('$d=2$, $a=%.1f$, $r=%.1f$, $T=%.0f$',a,r,dt*nSteps),'Interpreter','latex');
legend({'zero-flux Galerkin','$\phi_R$ from (6.17)','particle $\phi_T$',sprintf('$R=%.2f$',R)}, 'Location','southeast','Interpreter','latex');

subplot(1,3,3);
plot(rTest, abs(resid), 'LineWidth',2);
grid on; box on;
xlabel('test radius'); ylabel('force residual');
title('stationary residual');

saveFigure(fig,'fig_2d_ar_phi_compare');

%% ========================================================================
% Part II. One-dimensional experiment with a > r, using formula (5.15)
% ========================================================================
% The stationary distribution is computed from the direct one-dimensional
% interval formula (5.15).  For the symmetric background
% omega_m=(m/2)1_{[-1,1]} and a symmetric support [-L,L], the equation is
%
%   r(1+r) int_{-L}^L |x-y|^{r-1} phi_L(y) dy = (psi_a*omega_m)''(x),
%
% and L is determined by int phi_L = 1.  The particle evolution is then
% compared to this derived stationary distribution.
a1 = 0.80;
r1 = 0.20;
m1 = 1.80;
N1 = 500;
L0 = 1.40;
dt1 = 0.02;
T1 = 50.00;
nSteps1 = round(T1/dt1);

fprintf('1D: solving the interval characterization for a > r...\n');
Lstat = fzero(@(L) stationaryMassDefect1D(L,a1,r1,m1,140), [0.16,0.45]);
[massStat,edges1,centers1,phiStat,res1] = solveStationaryDensity1D(Lstat,a1,r1,m1,320);
dx1 = diff(edges1);
cdfStat1 = [0; cumsum(phiStat.*dx1)];
cdfStat1 = cdfStat1/cdfStat1(end);

% Direct Green/Poisson formula (5.16) at the displayed support L=0.2727.
% This now uses the exterior Poisson kernel for I^c, not the reciprocal
% interior kernel used in the earlier code.
Lformula = 0.2727;
[edges516,centers516,phi516] = formula516Density1D(Lformula,a1,r1,m1,320);
dx516 = diff(edges516);
cdf516 = [0; cumsum(phi516.*dx516)];
cdf516 = cdf516/cdf516(end);
fprintf('1D: support [-L,L], L       %.8f\n', Lstat);
fprintf('1D: stationary mass          %.15f\n', massStat);
fprintf('1D: Riesz-equation residual  %.4e\n', res1);
% Diagnostic comparing the finite-Riesz formula (5.15) and the corrected
% Green/Poisson formula (5.16) on the common displayed interval.
if numel(edges516)==numel(edges1) && max(abs(edges516-edges1)) < 1e-12
    formula516Err = max(abs(cdf516(:)-cdfStat1(:)));
else
    cdfStatOn516 = interp1(edges1,cdfStat1,edges516,'linear','extrap');
    formula516Err = max(abs(cdf516(:)-cdfStatOn516(:)));
end
fprintf('1D: max CDF discrepancy (5.15) vs corrected (5.16) %.4e\n', formula516Err);

rng(12);
X = linspace(-L0,L0,N1)' + 0.01*randn(N1,1);
X = sort(X);

time = [];
diam = [];
q99 = [];

for k = 1:nSteps1
    X = X + dt1*(-uniformBackgroundForce1D(X,a1,m1) + repulsionForce1D(X,r1));
    if mod(k,20)==0
        X = sort(X);
    end
    if mod(k,4)==0
        t = k*dt1;
        absX = abs(X);
        time(end+1,1) = t; %#ok<SAGROW>
        diam(end+1,1) = max(X)-min(X); %#ok<SAGROW>
        q99(end+1,1) = 2*quantile(absX,0.99); %#ok<SAGROW>
    end
end
X = sort(X);
empAtEdges = arrayfun(@(z) sum(X <= z)/N1, edges1);
cdfErrMax = max(abs(empAtEdges(:)-cdfStat1(:)));
fprintf('1D: final diameter           %.8f\n', diam(end));
fprintf('1D: max CDF discrepancy      %.4e\n', cdfErrMax);

histEdges = linspace(-1.05*Lstat,1.05*Lstat,55);
[counts,histEdges] = histcounts(X,histEdges,'Normalization','pdf');
histCenters = 0.5*(histEdges(1:end-1)+histEdges(2:end));

fig2 = figure('Color','w','Position',[80 100 1250 390]);
subplot(1,3,1);
plot(centers1,phiStat,'LineWidth',2.2); hold on;
plot(histCenters,counts,'--','LineWidth',1.8);
xline(-Lstat,':','LineWidth',1.4); xline(Lstat,':','LineWidth',1.4);
grid on; box on;
xlim([-0.46,0.46]);
xlabel('x'); ylabel('density');
title('density comparison');
legend({'density from (5.15)','particle histogram'},'Location','north','Interpreter','latex');

subplot(1,3,2);
plot(edges1,cdfStat1,'LineWidth',2.2); hold on;
plot(edges516,cdf516,'-.','LineWidth',2.0);
stairs(X,(1:N1)'/N1,'--','LineWidth',1.8);
grid on; box on;
xlim([-0.46,0.46]); ylim([0,1.02]);
xlabel('x'); ylabel('cumulative mass');
title(sprintf('$a=%.1f$, $r=%.1f$, $T=%.0f$',a1,r1,T1),'Interpreter','latex');
legend({'$\phi_L$ from (5.15)','$\phi_I$ from (5.16), $L=0.2727$','particle $\phi_T$'},'Location','southeast','Interpreter','latex');

subplot(1,3,3);
plot(time,diam,'LineWidth',2); hold on;
plot(time,q99,'--','LineWidth',2);
yline(2*Lstat,':','LineWidth',1.8);
grid on; box on;
xlabel('time'); ylabel('length scale');
title('support convergence');
legend({'diameter','twice 99% radius','2L from characterization'},'Location','northeast');

sgtitle(sprintf('1D attractive-dominant experiment: m=%.1f, L=%.4f, max CDF error=%.2e',m1,Lstat,cdfErrMax));
saveFigure(fig2,'fig_1d_a_greater_r_phi_compare');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = ringKernelMatrix(rVec, sVec, alpha, nTheta)
% A(i,j) is the radial component at radius rVec(i) generated by one unit
% of mass uniformly distributed on the circle of radius sVec(j) for the
% vector kernel K_alpha(z) = (1+alpha) z |z|^{alpha-1}.
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

function F = uniformBackgroundForce1D(x,a,m)
% omega = (m/2) 1_{[-1,1]}.  The integral of
% (1+a) sign(x-y)|x-y|^a against omega is exact:
% F(x) = (m/2)(|x+1|^{1+a}-|x-1|^{1+a}).
    F = 0.5*m*(abs(x+1).^(1+a) - abs(x-1).^(1+a));
end

function F = repulsionForce1D(x,r)
    Z = x - x';
    K = (1+r)*sign(Z).*abs(Z).^r;
    K(1:size(K,1)+1:end) = 0;
    F = mean(K,2);
end

function defect = stationaryMassDefect1D(L,a,r,m,n)
    [mass,~,~,~,~] = solveStationaryDensity1D(L,a,r,m,n);
    defect = mass - 1;
end

function [mass,edges,centers,phi,resid] = solveStationaryDensity1D(L,a,r,m,n)
% Solves the finite-interval Riesz equation for d=1, a>r>0:
%   r(1+r) int_{-L}^L |x-y|^{r-1} phi(y) dy = (psi_a*omega_m)''(x),
% using piecewise constant densities on uniform cells.
    edges = linspace(-L,L,n+1)';
    centers = 0.5*(edges(1:end-1)+edges(2:end));
    A = finiteRieszMatrix1D(centers,edges,r);
    b = uniformBackgroundSecond1D(centers,a,m);
    phi = A\b;
    dx = diff(edges);
    mass = sum(phi.*dx);
    resid = norm(A*phi-b)/norm(b);
end

function A = finiteRieszMatrix1D(centers,edges,r)
% A(i,j) = r(1+r) int_{cell j} |x_i-y|^{r-1} dy.
    centers = centers(:);
    l = edges(1:end-1)';
    u = edges(2:end)';
    A = zeros(numel(centers),numel(l));
    for i = 1:numel(centers)
        x = centers(i);
        vals = zeros(size(l));
        left = x >= u;
        right = x <= l;
        mid = ~(left | right);
        vals(left) = ((x-l(left)).^r - (x-u(left)).^r)/r;
        vals(right) = ((u(right)-x).^r - (l(right)-x).^r)/r;
        vals(mid) = ((x-l(mid)).^r + (u(mid)-x).^r)/r;
        A(i,:) = r*(1+r)*vals;
    end
end

function G = uniformBackgroundSecond1D(x,a,m)
% G=(psi_a*omega_m)'' for omega_m=(m/2)1_{[-1,1]}.
    G = 0.5*m*(1+a)*(sign(x+1).*abs(x+1).^a - sign(x-1).*abs(x-1).^a);
end


function muFormula = reconstructFormula6172D(sRing,R,a,r,m,Aforce,bforce)
% Direct radial quadrature of the reconstruction formula (6.17):
%   phi_R = (Q_{a,r} - lambda_{2,r}(-Delta)^s H_R)|_{B_R}.
% The source Q_{a,r} is evaluated from its Fourier multiplier form
% |xi|^{r-a} \hat omega, H_R is approximated by the exterior Green formula,
% and the interior fractional Laplacian is evaluated by the singular integral
% because H_R=0 inside B_R.  The relative Fourier-normalization constant is
% fixed by mass normalization and by minimizing the residual of the radial
% zero-flux condition.
    s = (1+r)/2;
    qIn = radialQShape2D(sRing,a,r,m);

    nExt = 85; Lext = 4.0; pow = 1.35;
    u = ((1:nExt)' - 0.5)/nExt;
    rhoExt = R + (Lext-R)*u.^pow;
    drExt = (Lext-R)*pow*u.^(pow-1)/nExt;
    qExt = radialQShape2D(rhoExt,a,r,m);

    nTheta = 96;
    theta = ((1:nTheta)-0.5)*(2*pi/nTheta);
    cth = cos(theta);

    H = zeros(nExt,1);
    for i = 1:nExt
        rho = rhoExt(i);
        acc = 0;
        for j = 1:nExt
            eta = rhoExt(j);
            d2 = max(rho^2 + eta^2 - 2*rho*eta*cth, 1e-12);
            etaGreen = ((rho^2-R^2)*(eta^2-R^2))./(R^2*d2);
            G = d2.^(-(1-r)/2).*greenEtaIntegral(etaGreen,s);
            acc = acc + mean(G)*qExt(j)*(2*pi*eta*drExt(j));
        end
        H(i) = acc;
    end

    corr = zeros(numel(sRing),1);
    for i = 1:numel(sRing)
        xi = sRing(i);
        acc = 0;
        for j = 1:nExt
            eta = rhoExt(j);
            d2 = max(xi^2 + eta^2 - 2*xi*eta*cth, 1e-12);
            K = mean(d2.^(-(3+r)/2));
            acc = acc + K*H(j)*(2*pi*eta*drExt(j));
        end
        corr(i) = acc;
    end

    dr = sRing(2)-sRing(1);
    objective = @(logc) formula617Residual(exp(logc),qIn,corr,sRing,dr,Aforce,bforce);
    try
        logc = fminbnd(objective,-10,10);
        coeff = exp(logc);
    catch
        coeff = 1;
    end
    dens = max(qIn + coeff*corr,0);
    muFormula = dens.*(2*pi*sRing*dr);
    muFormula = muFormula/sum(muFormula);
end

function val = formula617Residual(coeff,qIn,corr,sRing,dr,Aforce,bforce)
    dens = max(qIn + coeff*corr,0);
    mu = dens.*(2*pi*sRing*dr);
    if sum(mu) <= 0
        mu = ones(size(mu))/numel(mu);
    else
        mu = mu/sum(mu);
    end
    val = sqrt(mean((Aforce*mu-bforce).^2));
end

function q = radialQShape2D(rho,a,r,m)
% Shape of Q_{a,r}=F^{-1}(|xi|^{r-a} \hat omega) for radial Gaussian omega.
% Constants are suppressed because the displayed curve is mass-normalized.
    rho = rho(:);
    k = linspace(1e-5,16,2500)';
    dk = k(2)-k(1);
    w = k.^(1+r-a).*exp(-0.5*k.^2)*dk;
    q = zeros(size(rho));
    for i = 1:numel(rho)
        q(i) = m*sum(w.*besselj(0,k*rho(i)));
    end
    q = max(q,0);
end

function I = greenEtaIntegral(eta,s)
% Integral_0^eta t^{s-1}/(1+t) dt = eta^s/s * 2F1(s,1;s+1;-eta).
    eta = max(eta,0);
    I = zeros(size(eta));
    mask = eta > 0;
    if any(mask(:))
        z = eta(mask);
        I(mask) = z.^s/s .* hypergeom([s,1],s+1,-z);
    end
end

function [edges,centers,phi] = formula516Density1D(L,a,r,m,n)
% Direct evaluation of formula (5.16): phi_I = c_r (-Delta)^sigma g_tilde.
% Here sigma=r/2 and g=(psi_a*omega)''/[r(1+r)] on I=[A,B].
%
% IMPORTANT: g_tilde must be the sigma-harmonic extension to the EXTERIOR
% domain I^c.  For y outside I and x inside I, the exterior Poisson kernel is
%
%   P_{I^c}(y,x) = sin(pi*sigma)/pi * (((y-A)*(y-B))/((x-A)*(B-x)))^sigma / |y-x|.
%
% This is the opposite ratio from the interior Poisson kernel.  The earlier
% code used the reciprocal ratio, which is why the (5.16) curve was shifted.
%
% The fractional Laplacian is then approximated by the singular-integral
% formula.  The multiplicative constant is irrelevant for the plotted CDF
% because the density is normalized to total mass one.
    sigma = r/2;
    A = -L; B = L;
    edges = linspace(A,B,n+1)';
    centers = 0.5*(edges(1:end-1)+edges(2:end));
    dx = edges(2)-edges(1);
    g = uniformBackgroundSecond1D(centers,a,m)/(r*(1+r));

    nOut = 1400; Ymax = 10.0; pow = 1.8;
    u = ((1:nOut)' - 0.5)/nOut;
    z = (Ymax-L)*u.^pow + 1e-6;
    dz = (Ymax-L)*pow*u.^(pow-1)/nOut;
    yLeft = A - z;
    yRight = B + z;

    gLeft = zeros(nOut,1); gRight = zeros(nOut,1);
    for j = 1:nOut
        gLeft(j) = poissonExtensionIntervalExterior(centers,g,dx,A,B,sigma,yLeft(j));
        gRight(j) = poissonExtensionIntervalExterior(centers,g,dx,A,B,sigma,yRight(j));
    end

    phiRaw = zeros(n,1);
    for i = 1:n
        x = centers(i);
        delta = x - centers;
        mask = abs(delta) > 1e-12;

        inside = sum((g(i)-g(mask))./abs(delta(mask)).^(1+2*sigma))*dx;
        outsideL = sum((g(i)-gLeft)./abs(x-yLeft).^(1+2*sigma).*dz);
        outsideR = sum((g(i)-gRight)./abs(x-yRight).^(1+2*sigma).*dz);

        phiRaw(i) = inside + outsideL + outsideR;
    end

    % Small negative values can occur from quadrature error near the two
    % endpoint singularities.  The theoretical density is nonnegative.
    negMass = sum(max(-phiRaw,0))*dx;
    if negMass > 1e-4
        fprintf('1D: warning, negative quadrature mass in (5.16) = %.3e\n', negMass);
    end
    phi = max(phiRaw,0);
    if sum(phi)*dx <= 0
        warning('formula516Density1D produced nonpositive mass; falling back to max(g,0).');
        phi = max(g,0);
    end
    phi = phi/(sum(phi)*dx);
end

function val = poissonExtensionIntervalExterior(x,g,dx,A,B,sigma,y)
% Exterior Poisson kernel for the fractional Laplacian on I^c.
% Input x are quadrature points inside I=[A,B]; y is outside I.
    c_sigma = sin(pi*sigma)/pi;
    num = (y-A)*(y-B);          % positive for y in I^c
    den = (x-A).*(B-x);         % positive for x in I
    den = max(den,eps);
    P = c_sigma * (num./den).^sigma ./ abs(x-y);
    val = sum(P.*g)*dx;
end

function x = projectedLeastSquares(A,b,nIter)
% Simple fallback for nonnegative least-squares if lsqnonneg is unavailable.
    x = max(A\b,0);
    L = norm(A)^2 + 1e-12;
    tau = 0.9/L;
    for it = 1:nIter
        x = max(x - tau*A'*(A*x-b), 0);
    end
end

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
