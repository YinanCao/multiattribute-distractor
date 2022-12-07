clear;clc;close all;
addpath(genpath('./modelfits/'))
addpath(genpath('./functions/'))
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

crossFit = cell(0);
Nfit = 10; % random starting points
for s = 1:n_subj
    disp(['subj: ',num2str(s)])
    tt = trial_type{s}; % trial type
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
    
    distort_flag = 0; % linear function, no distortion
    I0_flag = 1;
    Tnd_flag = 1;
    ffi_flag = 2; % free c parameter for FFI
    FFI_type = 2; % FFI inhibition mean of all other traces
    MI_flag = 1; % dual-route model mutual inhibition strength (free parameter)
    
    % model recovery:
    proto_model = 1; % SI
    % fit human data with a model of interest:
    proto = fitFunc_Ternary_AU_dyna_rankSI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
             distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    % get simulated model predictions using the best parameter estimates:
    p = [proto.relacc,1-proto.relacc]; rt = proto.rtout;
    % cross-fit all models to the simulated data:
    crossFit{s,proto_model,1} = fitFunc_Ternary_AU_dyna_rankSI(attribute,p,rt,minRT,Nfit,...
        distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    crossFit{s,proto_model,2} = fitFunc_Ternary_dyna_AdapGain_FFI(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    crossFit{s,proto_model,3} = fitFunc_Ternary_EV_dyna_DualRoute(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],MI_flag,I0_flag,Tnd_flag);
    

    proto_model = 2; % adaptive gain
    proto = fitFunc_Ternary_dyna_AdapGain_FFI(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
        distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    p = [proto.relacc,1-proto.relacc]; rt = proto.rtout;
    crossFit{s,proto_model,1} = fitFunc_Ternary_AU_dyna_rankSI(attribute,p,rt,minRT,Nfit,...
        distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    crossFit{s,proto_model,2} = fitFunc_Ternary_dyna_AdapGain_FFI(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    crossFit{s,proto_model,3} = fitFunc_Ternary_EV_dyna_DualRoute(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],MI_flag,I0_flag,Tnd_flag);

    
    proto_model = 3; % dual route
    proto = fitFunc_Ternary_EV_dyna_DualRoute(attribute,Y(:,1:2),Y(:,3),minRT,Nfit,...
        distort_flag,[],MI_flag,I0_flag,Tnd_flag);
    p = [proto.relacc,1-proto.relacc]; rt = proto.rtout;
    crossFit{s,proto_model,1} = fitFunc_Ternary_AU_dyna_rankSI(attribute,p,rt,minRT,Nfit,...
        distort_flag,1,[],FFI_type,I0_flag,Tnd_flag,ffi_flag,1);
    crossFit{s,proto_model,2} = fitFunc_Ternary_dyna_AdapGain_FFI(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],FFI_type,I0_flag,Tnd_flag,ffi_flag);
    crossFit{s,proto_model,3} = fitFunc_Ternary_EV_dyna_DualRoute(attribute,p,rt,minRT,Nfit,...
        distort_flag,[],MI_flag,I0_flag,Tnd_flag);

end

time_str = strrep(mat2str(fix(clock)),' ','_');
save(['modelrecovery_CDmodels_',time_str,'.mat'],'crossFit')
disp('done')