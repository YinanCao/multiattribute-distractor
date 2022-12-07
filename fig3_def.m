clear;clc;close all;
addpath(genpath('./modelfits/'))
addpath(genpath('./functions/'))
load('Binary_models_[2022_5_27_16_9_53].mat') % dynamic

model = {EV_CI,AU_CI,EV_DR}; % 3 models

for ID = 1:3

datadir = './datasets/';
datafile = {
       'behav_fmri.mat'
       'gluth_exp4.mat'
       'gluth_exp1.mat'
       'gluth_exp3.mat'
       'gluth_exp2_HP.mat'
       };
% aggregating data
accuracy = [];trial_type = [];probs = [];rews = [];RT = [];
for whichf = 1:length(datafile)
    D1 = load([datadir,datafile{whichf}]);
    accuracy = cat(2,accuracy,D1.behavior.accuracy);
    trial_type = cat(2,trial_type,D1.behavior.trial_type);
    probs = cat(2,probs,D1.behavior.probs);
    rews = cat(2,rews,D1.behavior.rews);
    RT = cat(2,RT,D1.behavior.RT); % in ms
end
n_subj = length(rews);

models = [];
for whichmodel = numel(model):-1:1
    for s = (1:n_subj)
        y = model{whichmodel}.outputFull_B{s};
        models(s,whichmodel,:) = y.relacc;
    end
end

out = [];
for s = n_subj:-1:1
    disp(['subj: ',num2str(s)])
    tt = trial_type{s}; %trial type (2 = distractor trial)
    
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
    data_acc = accuracy{s}; % p(HV over LV)
    data_rt = RT{s}/1000;
    data_rt(data_rt<0.1) = nan;
    miss =  isnan(data_acc) | isnan(data_rt);
    data_acc(miss) = nan;
    % model 
    pred = squeeze(models(s,ID,:,1));
    
    Y = [pred,data_acc(tt==1)];
    
    attribute = [Pnorm(tt==1,1:2),Xnorm(tt==1,1:2)]; % attribute matrix
    
    % geometric structure type: P dom, X dom, congruent (double dom)
    structp = [
    (Pnorm(:,1)>Pnorm(:,2) & Xnorm(:,1)<=Xnorm(:,2))+...
    (Xnorm(:,1)>Xnorm(:,2) & Pnorm(:,1)<=Pnorm(:,2)),...
    Xnorm(:,1)>Xnorm(:,2) & Pnorm(:,1)>Pnorm(:,2)];

    [~,struct_cat] = max(structp,[],2);
    
    EV = Pnorm(tt==1,1:2).*Xnorm(tt==1,1:2);
    dEV = EV(:,1)-EV(:,2);
    dEV = round(dEV,4);
    y = [dEV,Y,struct_cat(tt==1,:)];
    y = sortrows(y,1);
    
    EVlevel = unique(y(:,1));
    tmp = [];
    for i = 1:length(EVlevel)
        for j = 1:2
        id = y(:,1)==EVlevel(i)&y(:,end)==j;
        tmp(i,j,:) = nanmean(y(id,2:3),1);
        end
    end
    out(s,:,:,:) = tmp;
end

figure('position',[877   800-(ID-1)*312   312   205])
bar_yc(out(:,:,:,2),0); hold on;
markersize = 8;
err_yc(out(:,:,:,1),markersize)
tick = round(EVlevel,2);
set(gca,'xtick',1:length(EVlevel),'xticklabel',tick,...
    'fontsize',15,'xticklabelrotation',45)
set(gca,'ytick',[0.5,0.75,1],'tickdir','out','box','off','linewidth',1.2)
xlabel('\Delta EV (HV - LV)')
ylabel('p(H over L)')
set(gca, 'color', 'none');
ylim([0.45,1])

end



% plot functions
function bar_yc(data,flag)
    vw_color = [160,207,231; 198,133,201; 198,133,201;]./255;
    model_series = squeeze(nanmean(data));
    nsub = size(data,1);
    model_error  = squeeze(nanstd(data))./sqrt(nsub);
    if flag
        model_error  = squeeze(nanstd(data));
    end
    b = bar(model_series,'grouped');

    [ngroups, nbars] = size(model_series);

    for k = 1:nbars
        b(k).EdgeColor = vw_color(k,:);
        b(k).FaceColor = vw_color(k,:);
    end
    hold on;

    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        hold on;
        y = model_series(:,i);
        se = model_error(:,i);
        err = [y+se,y-se]';
        for k = 1:size(err,2)
            hold on
            plot(ones(1,2)*(x(k)),err(:,k),'linewidth',1.5,...
                'color',vw_color(i,:)*.9)
        end
        hold on;
    end
    hold off
end


function err_yc(data,markersize)
    model_series = squeeze(nanmean(data));
    nsub = size(data,1);
    model_error = squeeze(nanstd(data))./sqrt(nsub);
    hold on;
    [ngroups, nbars] = size(model_series);
    groupwidth = min(0.8, nbars/(nbars + 1.5));

    vw_color = [160,207,231; 198,133,201; 198,133,201;]./255;
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        off = 0.05;
        y = model_series(:,i);
        se = model_error(:,i);
        err = [y+se,y-se]';
        for k = 1:size(err,2)
            hold on
            plot(ones(1,2)*(x(k)-off),err(:,k),'linewidth',2,...
                'color',ones(1,3)*0.5)
        end
        hold on
        plot(x-off, model_series(:,i),'ko',...
            'markerfacecolor',vw_color(i,:),'markersize',markersize);
    end
end

