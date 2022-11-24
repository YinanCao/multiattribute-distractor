clear;clc;close all;

addpath(genpath([pwd,'/functions/']))
datadir = [pwd,'/datasets/'];

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
    trial_type = cat(2,trial_type,D1.behavior.trial_type);
    probs = cat(2,probs,D1.behavior.probs);
    rews = cat(2,rews,D1.behavior.rews);
    accuracy = cat(2,accuracy,D1.behavior.accuracy);
    RT = cat(2,RT,D1.behavior.RT); % in ms
end
n_subj = length(rews);

Nfit = 10;
for s = 1:n_subj
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
    data_rt(miss) = nan;

    Y = [data_acc,1-data_acc,data_rt]; % data matrix
    Y = Y(tt==1,:); % b trials
    ntrl = size(Y,1);
    attribute = [Pnorm(tt==1,1:2),Xnorm(tt==1,1:2)]; % attribute matrix
    minRT = nanmin(Y(:,3));
    
    I0_flag = 1; % default
    Tnd_flag = 1; % default
    ffi_flag = 2; % free c (default)
    
    nfold = 5;
    CV_trl = reshape(1:ntrl,ntrl/nfold,nfold);
    
    modelspec = {[0,0],[0,1],[1,0],[1,1]}; % free coefficients of P and X functions
    % Xlin,Plin (EV):   [0,0]
    % Xpower,Plin (EU): [0,1]
    % Xlin,Ppt:         [1,0]
    % Xpower,Ppt (PT):  [1,1]

    % fit model to all trials and get best-fitting parameters
    for model = 1:length(modelspec)
        fpar = modelspec{model};
        EVnlS{model}.outputFull_B{s} = fitFunc_Binary_static_EV_nl(attribute,Y(:,1:2),Nfit,[],fpar);
        EVnlD{model}.outputFull_B{s} = fitFunc_Binary_EV_dyna_FFI_nl(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
                    [],fpar,I0_flag,Tnd_flag,ffi_flag);
    end
    
    % cross-validation:
    for fold = 1:nfold
        trltest  = CV_trl(:,fold); trltrain = setdiff(1:ntrl,trltest);
        for model = 1:length(modelspec)

            fpar = modelspec{model};

            % static
            output = fitFunc_Binary_static_EV_nl(attribute(trltrain,:),Y(trltrain,1:2),Nfit,[],fpar);
            parmEst = output.Xfit; % get best parameters
            EVnlS{model}.testLL_B(s,1,fold) = fitFunc_Binary_static_EV_nl(attribute(trltest,:),Y(trltest,1:2),Nfit,parmEst,fpar);

            % dynamic
            output = fitFunc_Binary_EV_dyna_FFI_nl(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
                        [],fpar,I0_flag,Tnd_flag,ffi_flag);
            parmEst = output.Xfit; % get best parameters
            EVnlD{model}.testLL_B(s,1,fold) = fitFunc_Binary_EV_dyna_FFI_nl(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
                        parmEst,fpar,I0_flag,Tnd_flag,ffi_flag);
        end
    end
     
end

toSave = 1;
time_str = strrep(mat2str(fix(clock)),' ','_');
if toSave
save(['Binary_prospect_',time_str,'.mat'],...
    'datafile','EVnlD','EVnlS')
end
disp('done')

