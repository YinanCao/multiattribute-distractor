function [output,trl_likelihood] = fitFunc_Ternary_AU_dyna_rankSI(attribute, p_data, rt_data, minRT, Nfit,...
    distort_flag, SI_flag, parmEst, FFI_type, I0_flag, Tnd_flag, ffi_flag, mono_flag)

obFunc = @(x) LL_Wald(attribute,p_data,rt_data,FFI_type,x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11));

if isempty(parmEst) % fit model to training set

    dr = [0.1,10];
    dr_lim = [1e-4,20];

    bound = [.1,3];
    bound_lim = [1e-3,10];

    lambda = [0,1];
    lambda_lim = [0,1];

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

    if SI_flag
        wSI = [0,1];
        wSI_lim = [0,1];
    else
        wSI = [1,1];
        wSI_lim = [1,1];
    end

    if I0_flag
        I0 = [-10, 10];
        I0_lim = [-10, 10];
    else
        I0 = [0, 0];
        I0_lim = [0, 0];
    end

    if Tnd_flag
        Tnd = [0, minRT-eps];
        Tnd_lim = [0, minRT-eps];
    else
        Tnd = [0, 0];
        Tnd_lim = [0, 0];
    end

    if ffi_flag == 2 % free
        c = [0,1];
        c_lim = [0,1];
    else % fixed value
        % when c=0, it's a Race model
        % when c=1, it's a DDM
        c = [ffi_flag, ffi_flag];
        c_lim = c;
    end

    B = [lambda; wSI; wSI; dr; bound; I0; Tnd; c; eta; k0; eta];
    B_lim = [lambda_lim; wSI_lim; wSI_lim; dr_lim; bound_lim; I0_lim; Tnd_lim; c_lim; eta_lim; k0_lim; eta_lim];

    LB = B_lim(:,1); UB = B_lim(:,2);
    
    conA = []; conb = [];
    if mono_flag
        conA = zeros(1,size(B,1));
        conA(1,2) = -1; conA(1,3) = 1;
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
        [Xfit_grid(iter,:), NegLL_grid(iter)] = fmincon(obFunc,X0_valid(iter,:),conA,conb,[],[],LB,UB,[],options);
    end
    toc

    [~,best] = min(NegLL_grid);
    Xfit = Xfit_grid(best,:);
    NegLL = NegLL_grid(best);
    LL = -NegLL;

    predFunc = @(x) Wald_predict(attribute,p_data,FFI_type,x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10),x(11));
    [pout,rtout,relacc,mu] = predFunc(Xfit);

    n = sum(~isnan(relacc));
    BIC = Np*log(n) + 2*NegLL;
    AIC = Np*2 + 2*NegLL;
    AICc = 2*NegLL + 2*Np + 2*Np*(Np+1)/(n-Np-1);

    output.mu = mu;
    output.Xfit = Xfit;
    output.Xfit_grid = Xfit_grid;
    output.NegLL_grid = NegLL_grid;
    output.LL = LL;
    output.BIC = BIC;
    output.AIC = AIC;
    output.AICc = AICc;
    output.pout = pout;
    output.rtout = rtout;
    output.relacc = relacc;
    trl_likelihood = [];
else % cross-validate model in test set:
    output = -obFunc(parmEst);
    [~,trl_likelihood] = obFunc(parmEst);
end

end


%%
function [negLL,L] = LL_Wald(attribute,p_data,rt_data,FFI_type,lambda,wSI2,wSI3,dr,theta,I0,Tnd,c,eta_p,k0_p,eta_x)

Nopt = 2;

att_P = attribute(:,[1,2,3]);
att_X = attribute(:,[4,5,6]);

% subjective distortion function:
k = att_P;
k(k==1) = 0.999;
y = eta_p.*log(k./(1-k)) + (1-eta_p).*log(k0_p/(1-k0_p));
att_P_bar = exp(y)./(1+exp(y));

att_X_bar = att_X.^eta_x;

% Selective integration:
p_SI = SI_helper(att_P_bar,wSI2,wSI3);
x_SI = SI_helper(att_X_bar,wSI2,wSI3);

% Utility:
U = lambda*x_SI + (1-lambda)*p_SI;
ntrl = size(U,1);

[~,ch] = max(p_data,[],2);
miss = isnan(p_data(:,1));
rt_data(miss) = NaN;

walddist = @(t,theta,mu,var) theta./sqrt(2.*pi.*var.*t.^3).*exp(-(theta-mu.*t).^2./(2.*var.*t));
Gfun = @(t,theta,mu,var) 1-normcdf((mu.*t-theta)./sqrt(var.*t))-normcdf((-mu.*t-theta)./sqrt(var.*t)).*exp(2.*theta.*mu./var);

% Feedforward-inhibition:
if FFI_type==1
U_ffi(:,1) = U(:,1) - c*max(U(:,[2,3]),[],2);
U_ffi(:,2) = U(:,2) - c*max(U(:,[1,3]),[],2);
else
U_ffi(:,1) = U(:,1) - c*mean(U(:,[2,3]),2);
U_ffi(:,2) = U(:,2) - c*mean(U(:,[1,3]),2);
end

mu = dr*U_ffi + I0;
% unit variance Wiener process:
variance = 1;
L = NaN(ntrl,1);
% trial-by-trial:
for trl = 1:ntrl
    mu1 = mu(trl, ch(trl)); % chosen
    mu2 = mu(trl, setdiff(1:Nopt, ch(trl))); % unchosen
    T = rt_data(trl) - Tnd;
    L(trl,1) = walddist(T,theta,mu1,variance) .* Gfun(T,theta,mu2,variance);
end

L(L==0) = eps;
L_valid = L;
L_valid(miss) = [];
negLL = -sum(log(L_valid));
if isnan(negLL)
    negLL = 1e10;
end

end

%%
function [p_pred, RT_pred, rel_acc, mu] = Wald_predict(attribute,p_data,FFI_type,lambda,wSI2,wSI3,dr,theta,I0,Tnd,c,eta_p,k0_p,eta_x)

Nopt = 2;

att_P = attribute(:,[1,2,3]);
att_X = attribute(:,[4,5,6]);

% subjective distortion function:
k = att_P;
k(k==1) = 0.999;
y = eta_p.*log(k./(1-k)) + (1-eta_p).*log(k0_p/(1-k0_p));
att_P_bar = exp(y)./(1+exp(y));

att_X_bar = att_X.^eta_x;

% Selective integration:
p_SI = SI_helper(att_P_bar,wSI2,wSI3);
x_SI = SI_helper(att_X_bar,wSI2,wSI3);

% Utility:
U = lambda*x_SI + (1-lambda)*p_SI;
ntrl = size(U,1);

[~,ch] = max(p_data,[],2);
miss = isnan(p_data(:,1));

walddist = @(t,theta,mu,var) theta./sqrt(2.*pi.*var.*t.^3).*exp(-(theta-mu.*t).^2./(2.*var.*t));
Gfun = @(t,theta,mu,var) 1-normcdf((mu.*t-theta)./sqrt(var.*t))-normcdf((-mu.*t-theta)./sqrt(var.*t)).*exp(2.*theta.*mu./var);

% Feedforward-inhibition:
if FFI_type==1
U_ffi(:,1) = U(:,1) - c*max(U(:,[2,3]),[],2);
U_ffi(:,2) = U(:,2) - c*max(U(:,[1,3]),[],2);
else
U_ffi(:,1) = U(:,1) - c*mean(U(:,[2,3]),2);
U_ffi(:,2) = U(:,2) - c*mean(U(:,[1,3]),2);
end

mu = dr*U_ffi + I0;

delta_t = 0.001;
t_max = 100;
ts = (1:ceil(t_max / delta_t)) * delta_t;

variance = 1;
p_pred = zeros(ntrl,3);
RT_pred = nan(ntrl,1);
for trl = 1:ntrl
    mu1 = mu(trl, ch(trl)); % chosen
    un_ch = setdiff(1:Nopt, ch(trl));
    mu2 = mu(trl, un_ch); % unchosen
    gG = walddist(ts,theta,mu1,variance) .* Gfun(ts,theta,mu2,variance);
    p_pred(trl,ch(trl)) = trapz(ts,gG); % choice probability
    p_pred(trl,un_ch) = 1-trapz(ts,gG); % choice probability
    RT_pred(trl,1) = trapz(ts,ts.*gG) / trapz(ts,gG) + Tnd;
end
rel_acc = p_pred(:,1);
rel_acc(miss) = NaN;
RT_pred(miss) = NaN;



end


%%
function ABD_SI = SI_helper(ABD,w2,w3)

% Selective-integration (SI) filtering
w1 = 1;
tmp = tiedrank(ABD')';
SIw = [];
for trl = size(tmp,1):-1:1
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