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

    % fit ternary:
    Y = [data_acc,1-data_acc,data_rt]; % data matrix
    Y = Y(tt==2,:); % ternary trials
    ntrl = size(Y,1);
    attribute = [Pnorm(tt==2,:),Xnorm(tt==2,:)]; % attribute matrix
    minRT = nanmin(Y(:,3));
    
    distort_flag = 0; % linear (default)
    FFI_type = 2; % default
    I0_flag = 1; % default
    Tnd_flag = 1; % default
    ffi_flag = 2; % free c (default)
    
    nfold = 5;
    CV_trl = reshape(1:ntrl,ntrl/nfold,nfold);
    
    % selective integration:
    rankSI.outputFull_T{s} = fitFunc_Ternary_AU_dyna_rankSI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    % adaptive gain:
    adapGain.outputFull_T{s} = fitFunc_Ternary_dyna_AdapGain_FFI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    % dual-route (EV)
    EV_DR.outputFull_T{s} = fitFunc_Ternary_EV_dyna_DualRoute(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,[],1,I0_flag,Tnd_flag);
    % dual-route (AU)
    AU_DR.outputFull_T{s} = fitFunc_Ternary_AU_dyna_DualRoute(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,[],1,I0_flag,Tnd_flag);
    % EV + Divisive Normalisation (DN)
    EV_DN.outputFull_T{s} = fitFunc_Ternay_EV_dyna_FFI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    % AU + DN
    AU_DN.outputFull_T{s} = fitFunc_Ternay_AUdivN_dyna_FFI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    % AU CI no D: as if fitting binary
    CIatt = attribute(:,[1,2,4,5]);
    AU_CI_noD.outputFull_T{s} = fitFunc_Binary_AU_dyna_rankSI_FFI(CIatt,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,0,[],I0_flag,Tnd_flag,ffi_flag);
    % EV
    EV_CI_noD.outputFull_T{s} = fitFunc_Binary_EV_dyna_FFI(CIatt,Y(:,1:2),Y(:,3),minRT,Nfit,...
         distort_flag,0,[],I0_flag,Tnd_flag,ffi_flag);
    
    
    for fold = 1:nfold
    trltest  = CV_trl(:,fold); trltrain = setdiff(1:ntrl,trltest);
    
    % SI:
    % fit training data:
    output = fitFunc_Ternary_AU_dyna_rankSI(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
             distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    parmEst = output.Xfit; % get best parameters
    % cross-fit test data:
    rankSI.testLL_T(s,1,fold) = fitFunc_Ternary_AU_dyna_rankSI(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
             distort_flag,1,parmEst,FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
         
    % adaptive gain:
    output = fitFunc_Ternary_dyna_AdapGain_FFI(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    parmEst = output.Xfit; % get best parameters
    adapGain.testLL_T(s,1,fold) = fitFunc_Ternary_dyna_AdapGain_FFI(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,parmEst,FFI_type,I0_flag,Tnd_flag,ffi_flag);
     
    % dual-route (EV)
    output = fitFunc_Ternary_EV_dyna_DualRoute(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,[],1,I0_flag,Tnd_flag);
    parmEst = output.Xfit; % get best parameters
    EV_DR.testLL_T(s,1,fold) = fitFunc_Ternary_EV_dyna_DualRoute(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,parmEst,1,I0_flag,Tnd_flag);
     
    % dual-route (AU)
    output = fitFunc_Ternary_AU_dyna_DualRoute(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,[],1,I0_flag,Tnd_flag);
    parmEst = output.Xfit; % get best parameters
    AU_DR.testLL_T(s,1,fold) = fitFunc_Ternary_AU_dyna_DualRoute(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,parmEst,1,I0_flag,Tnd_flag);

    % EV + Divisive Normalisation (DN)
    output = fitFunc_Ternay_EV_dyna_FFI(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    parmEst = output.Xfit; % get best parameters
    EV_DN.testLL_T(s,1,fold) = fitFunc_Ternay_EV_dyna_FFI(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,1,parmEst,FFI_type,I0_flag,Tnd_flag,ffi_flag);
     
    % AU + Divisive Normalisation (DN)
    output = fitFunc_Ternay_AUdivN_dyna_FFI(attribute(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    parmEst = output.Xfit; % get best parameters
    AU_DN.testLL_T(s,1,fold) = fitFunc_Ternay_AUdivN_dyna_FFI(attribute(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,1,parmEst,FFI_type,I0_flag,Tnd_flag,ffi_flag);
     
    % AU CI no D
    output = fitFunc_Binary_AU_dyna_rankSI_FFI(CIatt(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,0,[],I0_flag,Tnd_flag,ffi_flag);
    parmEst = output.Xfit; % get best parameters
    AU_CI_noD.testLL_T(s,1,fold) = fitFunc_Binary_AU_dyna_rankSI_FFI(CIatt(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,0,parmEst,I0_flag,Tnd_flag,ffi_flag);
    
    % EV CI no D
    output = fitFunc_Binary_EV_dyna_FFI(CIatt(trltrain,:),Y(trltrain,1:2),Y(trltrain,3),minRT,Nfit,...
         distort_flag,0,[],I0_flag,Tnd_flag,ffi_flag);
    parmEst = output.Xfit; % get best parameters
    EV_CI_noD.testLL_T(s,1,fold) = fitFunc_Binary_EV_dyna_FFI(CIatt(trltest,:),Y(trltest,1:2),Y(trltest,3),minRT,Nfit,...
         distort_flag,0,parmEst,I0_flag,Tnd_flag,ffi_flag);

    end
     
end % subj

toSave = 1;
time_str = strrep(mat2str(fix(clock)),' ','_');
if toSave
save(['Ternary_models_',time_str,'.mat'],'datafile',...
'rankSI',...
'adapGain',...
'EV_DR',...
'AU_DR',...
'EV_CI_noD',...
'AU_CI_noD',...
'EV_DN',...
'AU_DN')
end
disp('done')


