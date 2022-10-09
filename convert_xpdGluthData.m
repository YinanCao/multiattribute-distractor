% Behavioural data provided by the studies of Chau et al. (2020;
% 2014):
% https://datadryad.org/stash/dataset/doi:10.5061/dryad.040h9t7/

% Behavioural data provided by Gluth, S., Spektor, M. S., &
% Rieskamp, J. (2018):
% https://osf.io/8r4fh/

% For convenience, raw datafiles can be found in: ./datasets_raw/

%% Gluth Exp 4: N = 44
% extracts data from xpd files;
% save variables using the same conventions as in Chau et al. 
% and their dataset
clear; clc; close all;
datadir = './datasets_raw/GluthExp4/data/';
fList = dir([datadir,'vis*.xpd']); % get a list of files
v = {'trial_type' % variables of interest
    'hv_prob'
    'lv_prob'
    'd_prob'
    'hv_mag'      
    'lv_mag'      
    'd_mag'
    'response'
    'RT'
    'match_rsp'
    'match_rt'
    'block' % 0:training, 1:experiment
    'subject_id'};

for sub = 1:length(fList) % extract data for each subject
    
    disp(['Loading Gluth Exp 4 subject: ', num2str(sub)])
    f = [datadir,fList(sub).name];
    A = importdata(f);
    var = A.colheaders';
    var{8} = 'response';
    var{9} = 'RT';
    
    pos = A.data(:,16:18); % stimuli positions

    var_k = [];
    for k = length(v):-1:1
        var_k(k) = find(contains(var,v{k}));
    end
    D = A.data(:,var_k);
    
    pos = pos(D(:,end-1)==1,:); % remove training trials
    behavior.pos{sub} = pos; % column order: H -> L -> D, stim. position

    D = D(D(:,end-1)==1,:); % remove training trials

    rsp = D(:,8); % 0=HV, 1=LV, 2=D, >2:miss or empty quadrant
    rsp(rsp>1) = nan; % set invalid responses (including choosing D) as NaN
    behavior.accuracy{sub} = 1-rsp;
    behavior.trial_type{sub} = D(:,1); % 1=binary trial, 2=ternary trial
    behavior.probs{sub} = D(:,2:4); % column order: HV -> LV -> D, D=-99 in binary trials
    behavior.rews{sub} = D(:,5:7);  % column order: HV -> LV -> D 
    
    % Chau fMRI datafile contains the following variables, so we create them...:
    behavior.response_invalid_distractor{sub} = D(:,8)==2;
    behavior.response_invalid{sub} = isnan(rsp);

    behavior.RT{sub} = D(:,9); % in ms

    behavior.HV{sub} = behavior.probs{sub}(:,1) .* behavior.rews{sub}(:,1);
    behavior.LV{sub} = behavior.probs{sub}(:,2) .* behavior.rews{sub}(:,2);
    behavior.D{sub} = behavior.probs{sub}(:,3) .* behavior.rews{sub}(:,3);
    
end

disp('done')
savedir = './datasets/';
saveFile = [savedir,'gluth_exp4.mat'];
if ~exist(saveFile,'file')
    save(saveFile,'behavior');
else
    prompt = 'Datafile already exists. Do you want to overwrite gluth_exp4.mat? Y/N [N]: ';
    str = input(prompt,'s');
    if isempty(str)
        str = 'N';
    end
    if strcmp(str,'Y')
        save(saveFile,'behavior');
    end
end

behavior

%%
%% Gluth Exp 2: 
clear; clc; close all;
datadir = './datasets_raw/GluthExp2/data/';
fList = dir([datadir,'vis*.xpd']); % get a list of files
v = {'trial_type'
    'hv_prob'
    'lv_prob'
    'd_prob'
    'hv_mag'      
    'lv_mag'      
    'd_mag'
    'response'
    'RTtime'
    'match_rsp'
    'match_rt'
    'block'
    'subject_id'}; % rsp is 8

% high pressure, 25 subjects, even, group=0
% low pressure, 24 subjects, odd, group=1
group_name = {'HP','LP'};
for group = [0,1]
    s_con = 1;
    behavior = [];
    for sub = 1:length(fList)

        f = [datadir,fList(sub).name];
        A = importdata(f);
        var = A.colheaders';
        var{8} = 'response';
        var{9} = 'RTtime';
        
        pos = A.data(:,16:18); % stimuli positions
        
        var_k = [];
        for k = 1:length(v)
            var_k = [var_k,find(contains(var,v{k}))];
        end
        D = A.data(:,var_k);
        
        
        pos = pos(D(:,end-1)==1,:); % remove training trials
    
        D = D(D(:,end-1)==1,:); % remove training trials
        sub_id = D(1,end); % subject id

        if mod(sub_id,2)==group
            rsp = D(:,8);
            rsp(rsp>1) = nan; % 0=HV, 1=LV, not accuracy yet
            behavior.accuracy{s_con} = 1-rsp;
            behavior.trial_type{s_con} = D(:,1); % including novel trials
            behavior.probs{s_con} = D(:,2:4);
            behavior.rews{s_con} = D(:,5:7);
            behavior.response_invalid_distractor{s_con} = D(:,8)==2;
            behavior.response_invalid{s_con} = isnan(rsp);
            
            behavior.pos{s_con} = pos; % column order: H -> L -> D, stim. position

            behavior.HV{s_con} = behavior.probs{s_con}(:,1) .* behavior.rews{s_con}(:,1);
            behavior.LV{s_con} = behavior.probs{s_con}(:,2) .* behavior.rews{s_con}(:,2);
            behavior.D{s_con}  = behavior.probs{s_con}(:,3) .* behavior.rews{s_con}(:,3);

            behavior.RT{s_con} = D(:,9); % in ms
            s_con = s_con + 1;
        end
    end

    disp('done')
    savedir = './datasets/';
    save([savedir,'gluth_exp2_',group_name{group+1},'.mat'],'behavior');
    length(behavior.rews)
end



%%
%% Exp 1 Gluth
clear; clc; close all;
datadir = './datasets_raw/GluthExp1/data/';
fList = dir([datadir,'exp3*.xpd']); % get a list of files
v = {'trial_type' % variables of interest
    'hv_prob'
    'lv_prob'
    'd_prob'
    'hv_mag'      
    'lv_mag'      
    'd_mag'
    'response'
    'RT'
    'match_rsp'
    'match_rt'
    'block'
    'subject_id'};

for sub = 1:length(fList) % extract data for each subject
    
    disp(['Loading Gluth Exp 1 subject: ', num2str(sub)])
    f = [datadir,fList(sub).name];
    A = importdata(f);
    var = A.colheaders';
    var{8} = 'response';
    var{9} = 'RT';
    
    pos = A.data(:,16:18); % stimuli positions

    var_k = [];
    for k = length(v):-1:1
        var_k(k) = find(contains(var,v{k}));
    end
    D = A.data(:,var_k);
    pos = pos(D(:,end-1)==1,:); % remove training trials
    behavior.pos{sub} = pos; % column order: H -> L -> D, stim. position
    D = D(D(:,end-1)==1,:); % remove training trials

    % 412 trials in total, involving novel trials
    
    rsp = D(:,8); % 0=HV, 1=LV, 2=D
    rsp(rsp>1) = nan; % set invalid responses (including choosing D) as NaN
    behavior.accuracy{sub} = 1-rsp;
    behavior.trial_type{sub} = D(:,1); % 1=binary trial, 2=ternary trial
    behavior.probs{sub} = D(:,2:4); % column order: HV -> LV -> D, D=-99 in binary trials
    behavior.rews{sub} = D(:,5:7);  % column order: HV -> LV -> D 
    
    % Chau fMRI datafile contains the following variables, so we create them...:
    behavior.response_invalid_distractor{sub} = D(:,8)==2;
    behavior.response_invalid{sub} = isnan(rsp);

    behavior.RT{sub} = D(:,9); % in ms

    behavior.HV{sub} = behavior.probs{sub}(:,1) .* behavior.rews{sub}(:,1);
    behavior.LV{sub} = behavior.probs{sub}(:,2) .* behavior.rews{sub}(:,2);
    behavior.D{sub} = behavior.probs{sub}(:,3) .* behavior.rews{sub}(:,3);
    
end

% disp('done')
savedir = './datasets/';
saveFile = [savedir,'gluth_exp1.mat'];
save(saveFile,'behavior');
behavior

%%
%% Gluth exp 3

clear; clc; close all;
datadir = './datasets_raw/GluthExp3/data/';
fList = dir([datadir,'visuelle*.xpd']); % get a list of files
v = {'trial_type' % variables of interest
    'hv_prob'
    'lv_prob'
    'd_prob'
    'hv_mag'      
    'lv_mag'      
    'd_mag'
    'response'
    'RT'
    'match_rsp'
    'match_rt'
    'block'
    'subject_id'};

for sub = 1:length(fList) % extract data for each subject
    
    disp(['Loading Gluth Exp 3 subject: ', num2str(sub)])
    f = [datadir,fList(sub).name];
    A = importdata(f);
    var = A.colheaders';
    var{8} = 'response';
    var{9} = 'RT';

    pos = A.data(:,16:18); % stimuli positions
    
    var_k = [];
    for k = length(v):-1:1
        var_k(k) = find(contains(var,v{k}));
    end
    D = A.data(:,var_k);
    
    pos = pos(D(:,end-1)==1,:); % remove training trials
    behavior.pos{sub} = pos; % column order: H -> L -> D, stim. position
    
    D = D(D(:,end-1)==1,:); % remove training trials

    rsp = D(:,8); % 0=HV, 1=LV, 2=D
    rsp(rsp>1) = nan; % set invalid responses (including choosing D) as NaN
    behavior.accuracy{sub} = 1-rsp;
    behavior.trial_type{sub} = D(:,1); % 1=binary trial, 2=ternary trial
    behavior.probs{sub} = D(:,2:4); % column order: HV -> LV -> D, D=-99 in binary trials
    behavior.rews{sub} = D(:,5:7);  % column order: HV -> LV -> D 
    
    % Chau fMRI datafile contains the following variables, so we create them...:
    behavior.response_invalid_distractor{sub} = D(:,8)==2;
    behavior.response_invalid{sub} = isnan(rsp);

    behavior.RT{sub} = D(:,9); % in ms

    behavior.HV{sub} = behavior.probs{sub}(:,1) .* behavior.rews{sub}(:,1);
    behavior.LV{sub} = behavior.probs{sub}(:,2) .* behavior.rews{sub}(:,2);
    behavior.D{sub} = behavior.probs{sub}(:,3) .* behavior.rews{sub}(:,3);
    
end

disp('done')
savedir = './datasets/';
saveFile = [savedir,'gluth_exp3.mat'];
save(saveFile,'behavior');

behavior