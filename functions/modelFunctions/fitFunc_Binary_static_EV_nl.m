function output = fitFunc_Binary_static_EV_nl(attribute, p_data, Nfit, parmEst, fpar)

obFunc = @(x) LL_func(attribute,p_data,x(1),x(2),x(3));

if isempty(parmEst)
    
    beta  = [1e-5, 5];
    beta_lim  = [0, 1e3];

    eta = [0,10];
    eta_lim = [0,50];

    B = [beta; eta.^fpar(1); eta.^fpar(2)];
    B_lim = [beta_lim; eta_lim.^fpar(1); eta_lim.^fpar(2)];
    
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
    output = -obFunc(parmEst);
end

end

%%
% log-likelihood function:
function [negLL, p_pred, rel_acc] = LL_func(attribute,p_data, beta,tau,alpha)

miss = isnan(p_data(:,1));

att_P = attribute(:,[1,2]);
att_X = attribute(:,[3,4]);

% subjective distortion function:
att_P_bar = (att_P.^tau)./((att_P.^tau + (1-att_P).^tau).^(1/tau));
att_X_bar = att_X.^alpha;

% Utility:
U = att_P_bar.*att_X_bar;
v = U*beta;
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


