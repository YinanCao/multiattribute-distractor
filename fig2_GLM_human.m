clear;clc;close all;

% choose which distractor regressor
whichDvar = 'DV-HV'; 
% whichDvar = 'DV';

datadir = './datasets/';
datafile = {
       'behav_fmri.mat'
       'gluth_exp4.mat'
       'gluth_exp1.mat'
       'gluth_exp3.mat'
       'gluth_exp2_HP.mat' % gluth high pressure
       };
% aggregating data
accuracy = [];trial_type = [];
probs = [];rews = [];RT = [];
for whichf = 1:length(datafile) % aggregating data
    D1 = load([datadir,datafile{whichf}]);
    accuracy = cat(2,accuracy,D1.behavior.accuracy);
    trial_type = cat(2,trial_type,D1.behavior.trial_type);
    probs = cat(2,probs,D1.behavior.probs);
    rews = cat(2,rews,D1.behavior.rews);
    RT = cat(2,RT,D1.behavior.RT); % in ms
end
n_subj = length(rews);

for s = n_subj:-1:1
    
    disp(['subj: ',num2str(s)])
    % trial type (2 = ternary trial, 1 = binary)
    % 1 = binary
    tt = trial_type{s};
    if length(unique(tt))>2 % some datasets have irrelevant conditions
        tt(tt~=0&tt~=10) = nan;
        tt(tt==0) = 2;
        tt(tt==10) = 1;
    end
    P = probs{s}; % reward probability (H, L, D)
    X = rews{s};  % reward magnitude (H, L, D)
    P(P<=0) = nan;
    X(X<=0) = nan;
    % rescale to (0,1]
    Pnorm = bsxfun(@times, P, 1./prctile(P,100));
    Xnorm = bsxfun(@times, X, 1./prctile(X,100));
    
    data_acc = accuracy{s}; % p(H over L) relative accuracy

    % for regression:
    Ttrl = [Pnorm(tt==2,:),Xnorm(tt==2,:),data_acc(tt==2,1)]; % ternary,last col = accuacy
    Btrl = [Pnorm(tt==1,:),Xnorm(tt==1,:),data_acc(tt==1,1)]; % binary

    ntrl = size(Ttrl,1);
    regreMatrix = []; % now, let's prepare stuff for the regression
    for trl = ntrl:-1:1 % each T trial, find matched B trials
        bi_id = find(sum(abs(bsxfun(@minus,Btrl(:,[1,2,4,5]),Ttrl(trl,[1,2,4,5]))),2)<1e-10);
        Bmatch = Btrl(bi_id,end); % B responses, 1 or 0
        if ~isempty(Bmatch(~isnan(Bmatch)))
            Hchoice = nansum(Bmatch); % n of H choice
            Ntot = nansum(~isnan(Bmatch)); % total N observations
        else % only 1 trl, but missed response
            Hchoice = nan;
            Ntot = nan;
        end
        Ntot_ternary = 1;
        regreMatrix(trl,:) = [Ttrl(trl,:),Ntot_ternary,Hchoice,Ntot];
    end

    % choice response:
    % two-column for glmfit: number of successes in the corresponding number of trials in n
    Ty = regreMatrix(:,7:8);  % two-column input, ternary
    By = regreMatrix(:,9:10); % two-column input, binary
    
    % prepare GLM regressors
    % regreMatrix: first 6 cols are: ph, pl, pd, xh, xl, xd
    hv = regreMatrix(:,1).*regreMatrix(:,4); % get EV
    lv = regreMatrix(:,2).*regreMatrix(:,5);
    dv = regreMatrix(:,3).*regreMatrix(:,6);
    HVmLV = hv-lv;
    
    switch whichDvar
        case 'DV-HV'
        D = dv-hv;
        case 'DV'
        D = dv;
    end

    % regression for T and B individually
    Xreg = [HVmLV,D];
    Xreg = zscore(Xreg);  % normalisation
    Interaction = Xreg(:,1).*Xreg(:,end); % interaction term
    X = [Xreg,Interaction]; % GLM regressors
    linkFun = 'logit';
    glm1(s,:) = glmfit(X,Ty,'binomial',linkFun); % T
    glm2(s,:) = glmfit(X,By,'binomial',linkFun); % B
    % regression for T & B combined
    C = [zeros(150,1);ones(150,1)]; % 'Condition' (D present=1)
    Xcomb = [repmat(X,2,1),bsxfun(@times,repmat(X,2,1),C),C];
    ycomb = [By;Ty];
    glm3(s,:) = glmfit(Xcomb,ycomb,'binomial',linkFun); % T & B combined
end

% do the plots:
out = {glm1,glm2,glm3};

switch whichDvar
    case 'DV-HV'
    varnameall = {
    {'HV-LV', 'DV-HV', '(HV-LV) x (DV-HV)','','','',''}
    {'HV-LV', 'DV-HV', '(HV-LV) x (DV-HV)','','','',''}
    {'HV-LV', 'DV-HV', '(HV-LV) x (DV-HV)','(HV-LV) x C','(DV-HV) x C', '(HV-LV) x (DV-HV) x C','C'}
    };
    case 'DV'
    varnameall = {
    {'HV-LV', 'DV', '(HV-LV) x DV','','','',''}
    {'HV-LV', 'DV', '(HV-LV) x DV','','','',''}
    {'HV-LV', 'DV', '(HV-LV) x DV','(HV-LV) x C','DV x C', '(HV-LV) x DV x C','C'}
    };
end

close all
grey = [ones(1,3)*0.25;
        ones(1,3)*0.75;
        ones(1,3)*0.5];

figure('position',[617   604   729   343])
np = numel(out);
for panel = 1:np
    varname = varnameall{panel};
    subplot(1,np,panel)
    
    tmp = out{panel};
    % stats:
    beta = tmp(:,2:end); nbeta = size(beta,2);
    [~,P0,~,S] = ttest(beta,0,'tail','both');
    P = Bonf_Holm_yc(P0); % bonf-holm correction for p value
    sig = P<.05;
    array2table([S.tstat;P0;P;sig])

    M = nanmean(tmp,1); se = nanstd(tmp,1)./sqrt(n_subj);
    M(1) = []; se(1) = []; m_err = M; se_err = se;
    b = bar([m_err,zeros(1,8-length(m_err))],'facecolor','flat','edgecolor','flat');

    for i = 1:size(b.CData,1)
        b.CData(i,:) = grey(panel,:);
    end
    hold on; plot([-10,10],[0,0],'linewidth',0.5,'color','k')
    err = [m_err + se_err; m_err - se_err];
    x = 1:length(m_err);
    for k = 1:size(err,2)
        plot(ones(1,2)*(x(k)),err(:,k),'linewidth',2,'color',b.CData(k,:)*.8)
        hold on
    end
    for si = 1:length(sig)
        off = [1.5,2.8]; sig_i = 2;
        if sign(m_err(si))>0,sig_i = 1;end
           sigy = m_err(si)+sign(m_err(si))*(se_err(si)*off(sig_i));
        if sig(si)
           text(si,sigy,'*','HorizontalAlignment','Center',...
            'BackGroundColor','none','VerticalAlignment','middle',...
            'fontsize',16,'color','k'); hold on;
        end
    end
    
    if panel==3
        % prepend a color for each tick label
        ticklabels_new = varname;
        for i = 5:6
            ticklabels_new{i} = ['\color{blue} ' varname{i}];
        end
        varname = ticklabels_new;
    end
   
    set(gca,'xticklabel',varname,'xticklabelrotation',45)
    xlim([0.5,8-0.5])
    set(gca,'box','off','linewidth',1,'tickdir','out','xtick',1:length(m_err))
    set(gca,'fontsize',14)
    if panel <= 1
        ylabel('GLM \beta')
    end
    set(gca, 'color', 'none');
    ylim([-0.25,.72]);
    set(gca,'ytick',[-0.2,0,0.2,0.4,0.6])
end



function [p_corr,h] = Bonf_Holm_yc(p,alpha)
    % Holm's seq Bonferroni correction
    % Yinan Cao
    % oxford 2019
    if sum(isnan(p))>0, error('NaN found in p list!'); end
    if size(p,1)>size(p,2), p = p'; end
    if nargin < 2
        alpha = .05;
    end
    Np = length(p);
    c_alpha = alpha./(Np:-1:1);
    [p_sorted, idx] = sort(p,'ascend'); % sort p vals
    i = find(p_sorted>c_alpha,1); h = ones(size(p));
    if ~isempty(i)
        h(idx(i:end)) = 0;
    end
    for i = Np:-1:1
        p_corr(idx(i)) = max(min([p_sorted(1:i).*(Np+1-(1:i));ones(1,i)],[],1));
    end
end
