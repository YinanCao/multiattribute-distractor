function output = fitFunc_Ternary_AU_dyna_DualRoute(attribute, p_data, rt_data, minRT, Nfit,...
    distort_flag, parmEst, MI_flag, I0_flag, Tnd_flag)

obFunc = @(x) LL_Wald(attribute,p_data,rt_data,x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10));

if isempty(parmEst)
    
    lambda = [0,1];
    lambda_lim = [0,1];

    dr = [0.1,2];
    dr_lim = [1e-4,20];

    bound = [1e-3,2];
    bound_lim = [1e-3,10];

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
    
    if MI_flag == 1 % free
        frac = [0.2,0.8];
        frac_lim = [1e-5,1-1e-5];
    else % fixed
        frac = [MI,MI];
        frac_lim = [MI,MI];
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
    
    B = [lambda; dr; dr; bound; frac; I0; Tnd; eta; k0; eta];
    B_lim = [lambda_lim; dr_lim; dr_lim; bound_lim; frac_lim; I0_lim; Tnd_lim; eta_lim; k0_lim; eta_lim];

    LB = B_lim(:,1); UB = B_lim(:,2);

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
        [Xfit_grid(iter,:), NegLL_grid(iter)] = fmincon(obFunc,X0_valid(iter,:),[],[],[],[],LB,UB,[],options);
    end
    toc

    [~,best] = min(NegLL_grid);
    Xfit = Xfit_grid(best,:);
    NegLL = NegLL_grid(best);
    LL = -NegLL;

    predFunc = @(x) Wald_predict(attribute,p_data,x(1),x(2),x(3),x(4),x(5),x(6),x(7),x(8),x(9),x(10));
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
else
    output = -obFunc(parmEst);
end

end


%%
function [negLL,L] = LL_Wald(attribute,p_data,rt_data,lambda,dr,dr_dn,theta,f,I0,Tnd,eta_p,k0_p,eta_x)

Nopt = 2;

att_P = attribute(:,[1,2,3]);
att_X = attribute(:,[4,5,6]);

% subjective distortion function:
k = att_P;
k(k==1) = 0.999;
y = eta_p.*log(k./(1-k)) + (1-eta_p).*log(k0_p/(1-k0_p));
att_P_bar = exp(y)./(1+exp(y));

att_X_bar = att_X.^eta_x;

[~,ch] = max(p_data,[],2);
miss = isnan(p_data(:,1));
rt_data(miss) = NaN;

walddist = @(t,theta,mu,var) theta./sqrt(2.*pi.*var.*t.^3).*exp(-(theta-mu.*t).^2./(2.*var.*t));
Gfun = @(t,theta,mu,var) 1-normcdf((mu.*t-theta)./sqrt(var.*t))-normcdf((-mu.*t-theta)./sqrt(var.*t)).*exp(2.*theta.*mu./var);

% drift rate of raw inputs:
value = att_P_bar*(1-lambda) + att_X_bar*lambda;
ntrl = size(value,1);
mu = dr*(value - f*mean(value,2)) + I0; 
value_dn = bsxfun(@times, value, 1./sum(value,2));
% drift rate of divisively normalized inputs:
mu_dn = dr_dn*(value_dn - f*mean(value_dn,2)) + I0;

% unit variance Wiener process:
variance = 1;
L = NaN(ntrl,1);
% trial-by-trial:
for trl = 1:ntrl
    un_ch = setdiff(1:Nopt, ch(trl));
    mu1 = mu(trl, ch(trl)); % chosen
    mu2 = mu(trl, un_ch); % unchosen
    mu1_dn = mu_dn(trl, ch(trl));
    mu2_dn = mu_dn(trl, un_ch);
    
    T = rt_data(trl) - Tnd;
    gG1 = walddist(T,theta,mu1,variance).*Gfun(T,theta,mu2,variance)...
        .*Gfun(T,theta,mu1_dn,variance).*Gfun(T,theta,mu2_dn,variance);
    gG1_dn = walddist(T,theta,mu1_dn,variance).*Gfun(T,theta,mu1,variance)...
        .*Gfun(T,theta,mu2,variance).*Gfun(T,theta,mu2_dn,variance);
    L(trl,1) = gG1 + gG1_dn;
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
function [p_pred, RT_pred, rel_acc,mu] = Wald_predict(attribute,p_data,lambda,dr,dr_dn,theta,f,I0,Tnd,eta_p,k0_p,eta_x)

Nopt = 2;

att_P = attribute(:,[1,2,3]);
att_X = attribute(:,[4,5,6]);

% subjective distortion function:
k = att_P;
k(k==1) = 0.999;
y = eta_p.*log(k./(1-k)) + (1-eta_p).*log(k0_p/(1-k0_p));
att_P_bar = exp(y)./(1+exp(y));

att_X_bar = att_X.^eta_x;

[~,ch] = max(p_data,[],2);
miss = isnan(p_data(:,1));

walddist = @(t,theta,mu,var) theta./sqrt(2.*pi.*var.*t.^3).*exp(-(theta-mu.*t).^2./(2.*var.*t));
Gfun = @(t,theta,mu,var) 1-normcdf((mu.*t-theta)./sqrt(var.*t))-normcdf((-mu.*t-theta)./sqrt(var.*t)).*exp(2.*theta.*mu./var);

% drift rate of raw inputs:
value = att_P_bar*(1-lambda) + att_X_bar*lambda;
ntrl = size(value,1);
mu = dr*(value - f*mean(value,2)) + I0; 
value_dn = bsxfun(@times, value, 1./sum(value,2));
% drift rate of divisively normalized inputs:
mu_dn = dr_dn*(value_dn - f*mean(value_dn,2)) + I0;

delta_t = 0.001;
t_max = 100;
ts = (1:ceil(t_max / delta_t)) * delta_t;

variance = 1;
p_pred = zeros(ntrl,3);
RT_pred = nan(ntrl,1);
for trl = 1:ntrl
    un_ch = setdiff(1:Nopt, ch(trl));
    mu1 = mu(trl, ch(trl)); % chosen
    mu2 = mu(trl, un_ch); % unchosen
    mu1_dn = mu_dn(trl, ch(trl));
    mu2_dn = mu_dn(trl, un_ch);
    
    gG1 = walddist(ts,theta,mu1,variance).*Gfun(ts,theta,mu2,variance)...
        .*Gfun(ts,theta,mu1_dn,variance).*Gfun(ts,theta,mu2_dn,variance);
    gG1_dn = walddist(ts,theta,mu1_dn,variance).*Gfun(ts,theta,mu1,variance)...
        .*Gfun(ts,theta,mu2,variance).*Gfun(ts,theta,mu2_dn,variance);
    gG = gG1 + gG1_dn;
    p_pred(trl,ch(trl)) = trapz(ts,gG); % choice probability
    p_pred(trl,un_ch) = 1-trapz(ts,gG); % choice probability
    RT_pred(trl,1) = trapz(ts,ts.*gG) / trapz(ts,gG) + Tnd;
end
rel_acc = p_pred(:,1);
rel_acc(miss) = NaN;
RT_pred(miss) = NaN;

end


