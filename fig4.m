clear;clc;close all;
addpath(genpath('./modelfits/'))
addpath(genpath('./functions/'))
modeldir = './modelfits/';

Tmodel = load([modeldir,'Ternary_models_[2022_5_27_17_38_25].mat']);
Bmodel = load([modeldir,'Binary_models_[2022_5_27_16_9_53].mat']);

datadir = './datasets/';
datafile = {
       'behav_fmri.mat'
       'gluth_exp4.mat'
       'gluth_exp1.mat'
       'gluth_exp3.mat'
       'gluth_exp2_HP.mat'
       };

% aggregating data
accuracy = []; trial_type = []; probs = []; rews = [];
RT = [];
for whichf = 1:length(datafile)
    D1 = load([datadir,datafile{whichf}]);
    accuracy = cat(2,accuracy,D1.behavior.accuracy);
    trial_type = cat(2,trial_type,D1.behavior.trial_type);
    probs = cat(2,probs,D1.behavior.probs);
    rews = cat(2,rews,D1.behavior.rews);
    RT = cat(2,RT,D1.behavior.RT); % in ms
end
n_subj = length(rews);

for s = n_subj:-1:1
    % trial type (2 = distractor trial)
    tt = trial_type{s}; 
    if length(unique(tt))>2
        tt(tt~=0&tt~=10) = -99;
        tt(tt==0) = 2;
        tt(tt==10) = 1;
    end
    P = probs{s}; % reward probability (HV, LV, D)
    X = rews{s};  % reward magnitude (HV, LV, D)
    P(P<=0) = nan;
    X(X<=0) = nan;
    % normalise reward P and X to (0,1]
    Pnorm = bsxfun(@times, P, 1./prctile(P,100));
    Xnorm = bsxfun(@times, X, 1./prctile(X,100));
    attribute = [Pnorm(tt==2,:),Xnorm(tt==2,:)];
    data_acc = accuracy{s}; % p(H over L)
    data_rt = RT{s}/1000;
    data_rt(data_rt<0.1) = nan;
    miss = isnan(data_acc) | isnan(data_rt);
    data_acc(miss) = nan;
    
    m_t = [Tmodel.AU_CI_noD.outputFull_T{s}.relacc,...
           Tmodel.rankSI.outputFull_T{s}.relacc,...
           Tmodel.adapGain.outputFull_T{s}.relacc,...
           Tmodel.EV_DR.outputFull_T{s}.relacc,...
           ];
       
    m_b = [Bmodel.AU_CI.outputFull_B{s}.relacc,...
           Bmodel.rankSI.outputFull_B{s}.relacc,...
           Bmodel.adapGain.outputFull_B{s}.relacc,...
           Bmodel.EV_DR.outputFull_B{s}.relacc,...
          ];
    
    Y_b = [data_acc(tt==1),m_b]; % binary
    Y_t = [data_acc(tt==2),m_t]; % ternary
    
    % group repeated binary trials into unique conditions based on
    % P and X of H and L:
    x = [Pnorm(tt==1,1:2),Xnorm(tt==1,1:2),Y_b];
    [~,~,t3] = unique(x(:,1:4),'rows');
    tmp = [];
    for k = unique(t3)'
        tmp = [tmp; nanmean(x(t3==k,:),1)];
    end
    binary_data = tmp;
    % For each ternary trial, find the matched binary trial:
    x = [Pnorm(tt==2,:),Xnorm(tt==2,:),Y_t];
    out = [];
    for trl = size(x,1):-1:1 % loop over every ternary trial
        % find the binary condition with matching hv and lv:
        ter_hvlv = x(trl,[1,2,4,5]); % prob and mag of hv and lv of ternay trial
        bi_id = sum(abs(bsxfun(@minus,binary_data(:,1:4),ter_hvlv)),2)<1e-5;
        bi = binary_data(bi_id,5:end); % binary condition data
        ter = x(trl,7:end);
        out(trl,:,1) = ter;
        out(trl,:,2) = bi;
    end
    
    % condition types
    Pdom = attribute(:,1)>attribute(:,2)  & attribute(:,4)<=attribute(:,5);
    Xdom = attribute(:,1)<=attribute(:,2) & attribute(:,4)>attribute(:,5);
    
    % if use all trials:
    useall = 1;
    if useall
        Pdom = true(150,1);
        Xdom = true(150,1);
    end
    
    hv = attribute(:,[1,4]); lv = attribute(:,[2,5]); dd = attribute(:,[3,6]);
    % D dominance:
    Dloc = [sum(dd,2)>sum(hv,2) & sum(dd,2)>sum(lv,2),...
            sum(dd,2)<sum(lv,2) & sum(dd,2)<sum(hv,2)];
    distance = [];
    for trl = size(hv,1):-1:1
        distance(trl,1) = pdist([hv(trl,:); dd(trl,:)]) > pdist([lv(trl,:); dd(trl,:)]);
    end
    Call = [];
    for kk = 1:size(Dloc,2)
        C = [Pdom & distance, Pdom & ~distance,...
             Xdom & distance, Xdom & ~distance];
        for i = 1:size(C,2)
            C(:,i) = C(:,i).*Dloc(:,kk);
        end
        C = logical(C);
        Call(:,:,kk) = C;
    end
    
    % pdom L, pdom H, xdom L, xdom H, 
    valid = logical(sum(sum(Call,2),3));
    outVal = out(valid==1,:,:);
    
    % re-ref data:
    TmB = out(:,:,1)-out(:,:,2); % T minus B
    Nperm = 1000; n = size(outVal,1); Bias = [];
    for iter = Nperm:-1:1
        Bperm = outVal(randsample(1:n,n),:,2);
        Bias(:,:,iter) = outVal(:,:,1)-Bperm;
    end
    Bias = nanmean(nanmean(Bias,3),1); % bias-free re-referenced data
    TmB_bf = bsxfun(@minus,out(:,:,1),Bias)-out(:,:,2);
    
    allBias(s,1) = Bias(1); % human
    
    Y_full = [TmB_bf,TmB];
    find_nan = isnan(sum(Y_full,2));
    Y_full(find_nan,:) = nan;

    for kk = 1:size(Dloc,2)
        C = Call(:,:,kk);
        tmp = [];
        for whichc = 1:size(C,2)
            rsp = nanmean(Y_full(C(:,whichc)==1,:),1);
            tmp = [tmp; rsp];
        end
        ContextEff(s,:,:,kk) = tmp;
    end
    
end % end subj
size(ContextEff)

Nm = size(ContextEff,3);

clc
N = sum(Call,1);
squeeze(sum(N(:,[1,3],:),2)) % closer to L
squeeze(sum(N(:,[2,4],:),2)) % closer to H

% accuracy:
cef1 = mean(ContextEff(:,[1,3],1:Nm/2,:),2); % closer to L
cef2 = mean(ContextEff(:,[2,4],1:Nm/2,:),2); % closer to H
cef = cat(2,cef1,cef2);
size(cef)

% plots
close all;clc
vw_color = [160,207,231; 198,133,201]/255;
figure('position',[273   268   782   246])
human = cef(:,:,1,:);
modelpred = cef(:,:,2:end,:);
nmodel = size(modelpred,3);
for whichm = 1:nmodel
    subplot(1,nmodel,whichm)
    thism = modelpred(:,:,whichm,:);
    w = bar_yc_model(thism,0,vw_color); hold on;
    bar_yc_human(human,0,vw_color*0+.7); hold on;
    offsetAxes
    ylim([-1,1]*0.1)
    axis tight
    if i==1
        legend({'superior{\it D}','inferior{\it D}'},...
        'location','s','orientation','v','box','off','AutoUpdate','off')
    end
    str = {'\color{blue}L','\color{red}H'};
    set(gca,'xticklabel',str,'xticklabelrotation',0)
    set(gca,'fontsize',14,'box','off','tickdir','out')
    set(gca,'linewidth',1)

    stat2 = squeeze(thism-human);
    [~,p,~,s] = ttest(stat2,0,'tail','both');
    m = squeeze(nanmean(human))'; m = m(:); ns = size(stat2,1);
    se = squeeze(nanstd(human))'./sqrt(ns); se = se(:);
    p = squeeze(p)'; praw = p; [a,b] = size(p); p = p(:); ps = p;
    p(isnan(ps)) = [];
    % bonf-holm correction of p values:
    adjp = Bonf_Holm_yc(p);
    ps(~isnan(ps)) = adjp;
    p = reshape(ps,[a,b]); p = praw;
    t = squeeze(s.tstat)';
    w = w/2; nn = size(human,2); x = [(1:nn)-w/2;(1:nn)+w/2]; x = x(:);
    
    stat3 = squeeze(thism+human)/2;
    m3 = squeeze(nanmean(stat3))'; m3 = m3(:);
    % sig. asterisk
    y = [x,p(:),m(:),se(:),t(:)];
    for k = 1:size(y,1)
        sigy = m3(k);
        if sign(y(k,3))>0, va = 'middle'; else, va = 'top'; end
            pval = y(k,2);
        if pval<.05
            text(y(k,1),sigy,'*','HorizontalAlignment','center',...
            'VerticalAlignment',va,'BackGroundColor','none',...
            'fontsize',20,'color',ones(1,3)*0.5);hold on;
        end
    end
    set(gca, 'color', 'none');
    array2table([t(:),p(:)],'VariableNames',{'T-val','p-val (bonf-holm adj.)'})
    offsetAxes
    ylim([-1,1]*0.1); set(gca,'ytick',[-0.1,0,0.1])
    off = 0.4; 
    xlim([1-off,2+off])
    
    grey = ones(1,3)*0.7;
    ylab = .08; xlab = x(1)+.1;
    if whichm==1
    plot(xlab,ylab,'o','linewidth',1.5,'color',grey*0.9,'markersize',8); hold on
    plot([1,1]*xlab,[-1,1]*0.013+ylab,'linewidth',1,'color',grey)
    end
    
    text(1,-0.09,'D closer to','fontsize',14)
    
end


% human
cef1 = mean(ContextEff(:,[1,3],[6,1],:),2);
cef2 = mean(ContextEff(:,[2,4],[6,1],:),2); 
cef = (cat(2,cef1,cef2)); % raw, bf
size(cef)
% subj,struc,raw/bf,ddom
size(cef)

vw_color = [160,207,231; 198,133,201]/255;
figure('position',[0   0   782   246])

modelpred = cef;
nmodel = size(modelpred,3);
panel = [1,3];
for whichm = 1:2
    subplot(1,4,panel(whichm))
    thism = squeeze(modelpred(:,:,whichm,:));
    w = bar_yc_model(thism,0,vw_color); hold on;
    offsetAxes
    ylim([-1,1]*0.1)
    
    axis tight

    if whichm==1
        legend({'superior{\it D}','inferior{\it D}'},...
        'location','nw','orientation','v','box','off','AutoUpdate','off')
    end
    str = {'\color{blue}L','\color{red}H'};
    
    set(gca,'xticklabel',str,'xticklabelrotation',0)
    set(gca,'fontsize',14,'box','off','tickdir','out')
    set(gca,'linewidth',1)
    
    stat2 = squeeze(thism);
    [~,p,~,s] = ttest(stat2,0,'tail','both'); 
    m = squeeze(nanmean(stat2))'; m = m(:); ns = size(stat2,1);
    se = squeeze(nanstd(stat2))'./sqrt(ns); se = se(:);
    p = squeeze(p)'; praw = p; [a,b] = size(p); p = p(:); ps = p;
    p(isnan(ps)) = [];
    % bonf-holm correction of p values:
    adjp = Bonf_Holm_yc(p);
    ps(~isnan(ps)) = adjp;
    p = reshape(ps,[a,b]); 
    t = squeeze(s.tstat)';
    w = w/2; nn = size(thism,2); x = [(1:nn)-w/2;(1:nn)+w/2]; x = x(:);
    % sig. asterisk
    
    if whichm==2
        y = [x,p(:),m(:),se(:),t(:)];
    for k = 1:size(y,1)
        sigy = y(k,3) + sign(y(k,3))*(y(k,4));
        if sign(y(k,3))>0, va = 'middle'; else, va = 'top'; end
        pval = y(k,2);
        if pval<.05
            text(y(k,1),sigy,'*','HorizontalAlignment','center',...
            'VerticalAlignment',va,'BackGroundColor','none',...
            'fontsize',16);hold on;
        end
    end
    
    end
    set(gca, 'color', 'none');
    array2table([t(:),p(:)],'VariableNames',{'T-val','p-val (bonf-holm adj.)'})
    offsetAxes
    ylim([-1,1]*0.1); set(gca,'ytick',[-0.1,0,0.1])
    off = 0.4; 
    xlim([1-off,2+off])
    text(1,-0.09,'D closer to','fontsize',14)

    
end

figure('position',[273   0   78   246])
offsetAxes
m = mean(allBias);
b=bar(m,'facecolor',ones(1,3)*.75);
b.EdgeColor='none';
se = std(allBias)/sqrt(n_subj);
hold on
plot([1,1],[m-se,m+se],'linewidth',1.5,'color',ones(1,3)*0.1*.9)
set(gca, 'color', 'none');
set(gca,'fontsize',14,'box','off','tickdir','out')
set(gca,'linewidth',1,'xticklabel','')
ylim([-1,1]*0.1)
ylim([-1,1]*0.1); set(gca,'ytick',[-0.1,0,0.1])
ax = gca; % ax is the handle to the axes
ax.XColor = 'none';
