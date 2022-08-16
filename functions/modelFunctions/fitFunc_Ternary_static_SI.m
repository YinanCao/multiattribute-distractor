
function [output,p] = fitFunc_Ternary_static_SI(attribute, p_data, Nfit, SI_flag, distort_flag, mono_flag, parmEst)

obFunc = @(x) LL_func(attribute, p_data, x(1),x(2),x(3),x(4),x(5),x(6),x(7));

if isempty(parmEst)
w     = [0,1];
w_lim = [0,1];

beta  = [1e-5, 20];
beta_lim  = [0, 20];

if SI_flag
    wSI = [0,1];
    wSI_lim = [0,1];
else
    wSI = [1,1];
    wSI_lim = [1,1];
end

if distort_flag
    eta = [0,10];
    eta_lim = [0,50];
    k0 = [0.01,1-0.01];
    k0_lim = [0.01,1-0.01];
else
    eta = [1,1];
    eta_lim = [1,1];
    k0 = [0.5,0.5];
    k0_lim = [0.5,0.5];
end

B = [w; beta; wSI; wSI; eta; k0; eta];
B_lim = [w_lim; beta_lim; wSI_lim; wSI_lim; eta_lim; k0_lim; eta_lim];

LB = B_lim(:,1); UB = B_lim(:,2); 

conA = []; conb = [];
if mono_flag
    conA = zeros(1,size(B,1));
    conA(1,3) = -1; conA(1,4) = 1;
    conb = zeros(size(conA,1),1);
end

% grid search starting points:
Nall = 100;
X0 = zeros(Nall,size(B,1));
for i = 1:size(B,1)
    a = B(i,1); b = B(i,2);
    X0(:,i) = a + (b-a).*rand(Nall,1);
end
Np = sum(std(X0)~=0); % number of free parameters
feval = [5000, 5000]; % max number of function evaluations and iterations
options = optimset('MaxFunEvals',feval(1),'MaxIter',feval(2),'TolFun',1e-10,'TolX',1e-10,'Display','none');

X0_valid = [];
for iter = 1:Nall
    init_fval = obFunc(X0(iter,:));
    if isreal(init_fval) && ~isnan(init_fval) && ~isinf(init_fval)
        X0_valid = [X0_valid; X0(iter,:)];
    end
end

tic
parfor iter = 1:Nfit
    [Xfit_grid(iter,:), NegLL_grid(iter,1)] = fmincon(obFunc,X0_valid(iter,:),conA,conb,[],[],LB,UB,[],options);
end
toc

[~,best] = min(NegLL_grid);
Xfit = Xfit_grid(best,:);
NegLL = NegLL_grid(best);
LL = -NegLL;

[~,p_pred,relacc] = obFunc(Xfit);

n = sum(~isnan(relacc));
BIC = Np*log(n) + 2*NegLL;
AIC = Np*2 + 2*NegLL;
AICc = 2*NegLL + 2*Np + 2*Np*(Np+1)/(n-Np-1);

output.Xfit = Xfit;
output.Xfit_grid = Xfit_grid;
output.NegLL_grid = NegLL_grid;
output.LL = LL;
output.BIC = BIC;
output.AIC = AIC;
output.AICc = AICc;
output.pout = p_pred;
output.relacc = relacc;

else
    [output,p] = obFunc(parmEst);
    output = -output;
end

end


% log-likelihood function:
function [negLL, p_pred, rel_acc] = LL_func(attribute,p_data,w,beta,wSI2,wSI3,eta_p,k0_p,eta_x)

miss = isnan(p_data(:,1));

att_P = attribute(:,[1,2,3]);
att_X = attribute(:,[4,5,6]);

% subjective distortion function:
k = att_P;
k(k==1) = 0.999;
y = eta_p.*log(k./(1-k)) + (1-eta_p).*log(k0_p/(1-k0_p));
att_P_bar = exp(y)./(1+exp(y));

att_X_bar = att_X.^eta_x;

% Selective integration: ternay
p_SI = SI_helper(att_P_bar,wSI2,wSI3);
x_SI = SI_helper(att_X_bar,wSI2,wSI3);

U = w*x_SI + (1-w)*p_SI;
v = U(:,1:2)*beta;
v = bsxfun(@minus, v, prctile(v,100,2));
p_pred = exp(v) ./ nansum(exp(v),2);

L = p_pred.^p_data(:,1:2);
L_valid = L;
L_valid(miss,:) = [];
L_valid(L_valid==0) = eps;
negLL = -sum(nansum(log(L_valid),2));
if isnan(negLL)
    negLL = 1e10;
end

rel_acc = p_pred(:,1)./sum(p_pred(:,1:2),2);
rel_acc(isnan(p_data(:,1))) = NaN;

end


%%
function ABD_SI = SI_helper(ABD,w2,w3)

% SI:
w1 = 1;
tmp = tiedrank(ABD')';
SIw = [];
for trl = 1:size(tmp,1)
    temp = tmp(trl,:);
    temp(temp==1) = w3; % lowest
    temp(temp==3) = w1; % highest
    temp(temp==2) = w2; % mid
    temp(temp==1.5) = w3; % take care of a tie
    temp(temp==2.5) = w1; % take care of a tie
    SIw(trl,:) = temp;
end
ABD_SI = ABD.*SIw;

end