clear;clc;close all;
addpath(genpath('./modelfits/'))
addpath(genpath('./functions/'))
load('Binary_models_[2022_5_27_16_9_53].mat') % dynamic

datadir = './datasets/';
datafile = {
       'behav_fmri.mat'
       'gluth_exp4.mat'
       'gluth_exp1.mat'
       'gluth_exp3.mat'
       'gluth_exp2_HP.mat'
       };
AU_dyna_linear = AU_CI;
EV_dyna_linear = EV_CI;
EVDN_dyna_linear = EV_DN;
EVDR_dyna_linear = EV_DR;

modelofinterest = {AU_dyna_linear,EV_dyna_linear,EVDN_dyna_linear,EVDR_dyna_linear};
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

for s = 1:n_subj
    disp(['subj: ',num2str(s)])
    %get data
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
    data_rt(miss) = nan;
    
    modelpredB = [];
    Nmodel = numel(modelofinterest);
    for mk = 1:Nmodel
        tmp = modelofinterest{mk};
        mu = tmp.outputFull_B{s}.mu;
        pB = tmp.outputFull_B{s}.relacc;
        rtB = tmp.outputFull_B{s}.rtout;
        double_neg = mu(:,1)<0 & mu(:,2)<0;
        pB(double_neg) = nan;
        rtB(double_neg) = nan;
        modelpredB = [modelpredB,pB,rtB];
    end
    Y_b = [data_acc(tt==1),data_rt(tt==1),modelpredB]; % binary
    Y_t = [data_acc(tt==2),data_rt(tt==2)]; % ternary
    
    % group repeated binary trials into unique conditions based on
    % P and X of H and L:
    x = [Pnorm(tt==1,1:2),Xnorm(tt==1,1:2),Y_b];
    [t1,t2,t3] = unique(x(:,1:4),'rows');
    binary_data = [];
    for k = unique(t3)'
        binary_data = [binary_data; nanmean(x(t3==k,:),1)];
    end
    % For each ternary trial, find the matched binary trial:
    x = [Pnorm(tt==2,:),Xnorm(tt==2,:),Y_t];
    out = [];
    for trl = size(x,1):-1:1 % loop over every ternary trial
        % find the binary condition with matching hv and lv:
        ter_hvlv = x(trl,[1,2,4,5]); % prob and mag of hv and lv of ternay trial
        bi_id = find(sum(abs(bsxfun(@minus,binary_data(:,1:4),ter_hvlv)),2)<1e-6);
        bi = binary_data(bi_id,5:end); % binary condition data
        out(trl,:,1) = bi;
    end
    
    attribute = [Pnorm(tt==2,:),Xnorm(tt==2,:)];
    hv = attribute(:,1).*attribute(:,4); % expected value
    lv = attribute(:,2).*attribute(:,5);
    d =  attribute(:,3).*attribute(:,6);
    
    % data for map
    Y_complete = out;
    y_param = hv - lv;
    x_param = d - hv;
    % Binning to construct map (using the exact method as in Chau et al. 2020 eLife)
    avg_window = 0.3;
    bin_x = 0:.01:(1-avg_window);
    bin_y = 0:.01:(1-avg_window);
    bin_x = [bin_x; bin_x + avg_window];
    bin_y = [bin_y; bin_y + avg_window];
    if s == 1 % pre-allocation
        binned_avg = NaN(size(bin_y,2), size(bin_x,2), size(Y_complete,2), n_subj);
    end
    for bin_y_count = 1:size(bin_y,2)
        for bin_x_count = 1:size(bin_x,2)
            ind = y_param>=quantile(y_param,bin_y(1,bin_y_count)) & ...
                  y_param<=quantile(y_param,bin_y(2,bin_y_count)) & ...
                  x_param>=quantile(x_param,bin_x(1,bin_x_count)) & ...
                  x_param<=quantile(x_param,bin_x(2,bin_x_count));
            binned_avg(bin_y_count,bin_x_count,:,s) = nanmean(Y_complete(ind,:),1);
        end
    end
end

binned_avg = nanmean(binned_avg,4);

saveFile = 'Fig3_maps.mat';
if ~exist(saveFile,'file')
    save(saveFile,'binned_avg')
end