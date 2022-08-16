clear;clc;close all;
currdir = pwd;
parentdir = fileparts(pwd);
addpath(genpath([parentdir,'/modelfits/']))
load Fig3_maps.mat

% panel a:
map = cat(3,binned_avg(:,:,1:2:end),binned_avg(:,:,2:2:end));
nmodel = size(map,3);
figure('position',[534   595   661   243])
tlo = tiledlayout(2,nmodel/2,'TileSpacing','compact','Padding','compact');
for panel = 1:nmodel
    x = map(:,:,panel); hold on;
    L = nexttile; imagesc(x); set(gca,'YDir','normal','xtick',[],'ytick',[]);
    if panel<=nmodel/2
        colormap(L,hot);caxis([0.5,0.9]); hold on;
        contour(x,[0.6,0.72,0.82],'k','linewidth',1,'ShowText','off');
    else
        colormap(L,gray);caxis([0.83,0.9]); hold on;
        contour(x,ones(1,2)*0.875,'k','linewidth',1);
    end
    if panel==1 || panel==nmodel/2+1
        cb = colorbar; t = get(cb,'Limits'); tick = linspace(t(1),t(2),2); 
        tickstr = cell(0);
        for k = 1:2
            textStrings = strtrim(cellstr(num2str(tick(k))));
            tickstr(k) = strrep(cellstr(textStrings), '0.', '.');
        end
        set(cb,'YAxisLocation','left','location','westoutside')
        set(cb,'Ticks',tick,'ticklabels',tickstr,'fontsize',16)
    end
end


% panel b/c:
% load models
parentdir = './modelfits/';
linD = load([parentdir,'Binary_models_[2022_5_27_16_9_53].mat']);
nlD = load([parentdir,'Binary_models_NL_[2022_5_27_13_47_43].mat']);

linS = load([parentdir,'StaticBinaryModel_Linear_[2022_2_26_17_3_16].mat']);
nlS = load([parentdir,'StaticBinaryModel_NL_[2022_2_26_19_2_24].mat']);

model = {
{linD.AU_CI,linD.EV_CI,linD.EV_DN,linD.EV_DR}
{nlD.AU_CI,nlD.EV_CI,nlD.EV_DN,nlD.EV_DR}};
modelNames = {
    {'AU (Linear)','EV (Linear)','EV + DN (Linear)','Dual-route (Linear)'}
    {'AU (NL)','EV (NL)','EV + DN (NL)','Dual-route (NL)'}
    };
opt.DisplayWin = 0;
figure('position',[1079,333,502,182])
for k = 1:2
    subplot(1,2,k)
    thism = model{k};
    xx = [];
    for j = 1:numel(thism)
        xx = cat(2,xx,thism{j}.testLL_B);
        xx(xx<-1e5) = nan;
        xx(isinf(xx)) = nan;
    end
    xx = nanmean(xx,3);
    xx = xx(1:144,:);
    
    [~,out1] = VBA_groupBMC(xx',opt);
    out = {out1};
    allname = modelNames{k};
    for i = 1:length(out)
        tmp = out{i};
        N = length(tmp.pxp);
        tmp.pxp
        bar(tmp.Ef,'facecolor',ones(1,3)*0.75,'barwidth',0.6,'edgecolor',[1,1,1]*0.5,'linewidth',1.5)
        hold on;
        errorbar(tmp.Ef,sqrt(tmp.Vf),'.','linewidth',1,'color',[1,1,1]*0.5)
        ylim([0,1.1])
        set(gca,'xtick',1:length(tmp.Ef),'xticklabel',allname,'xticklabelrotation',30)
        set(gca,'fontsize',15,'box','off','tickdir','out','linewidth',1)
        set(gca,'ytick',[0,1])
        if k==1
            ylabel({'Posterior';'model freq.'})
        end
        
        % plot Pexc
        hold on;
        [~,best] = max(tmp.Ef);
        pb = tmp.Ef(best);
        pb_sem = sqrt(tmp.Vf(best));
        L = 0.06; off = pb_sem*2.5;
        w = 1.2;
        plot(ones(1,2)*best,[pb,pb+L]+off,'k','linewidth',w)
        hold on;
        plot([1,4],ones(1,2)*(pb+L+off),'k','linewidth',w)
    end
    set(gca, 'color', 'none');
end

% static models
model = {
{linS.AU,linS.EV,linS.EVDN}
{nlS.AU,nlS.EV,nlS.EVDN}};
modelNames = {
    {'AU (Linear)','EV (Linear)','EV + DN (Linear)'}
    {'AU (NL)','EV (NL)','EV + DN (NL)'}
    };
figure('position',[641   333   437   175])
for k = 1:2
    subplot(1,2,k)
    thism = model{k};
    xx = [];
    for j = 1:numel(thism)
        xx = cat(2,xx,thism{j}.testLL_B);
    end
    xx = mean(xx,3);
    xx = xx(1:144,:);
    [~,out1] = VBA_groupBMC(xx',opt);
    out = {out1};
    allname = modelNames{k};
    for i = 1:length(out)
        tmp = out{i}; N = length(tmp.pxp);
        tmp.pxp
        bar(tmp.Ef,'facecolor',ones(1,3)*0.75,'barwidth',0.6,'edgecolor',[1,1,1]*0.5,'linewidth',1.5)
        hold on;
        errorbar(tmp.Ef,sqrt(tmp.Vf),'.','linewidth',1,'color',[1,1,1]*0.5)
        ylim([0,1.1])
        set(gca,'xtick',1:length(tmp.Ef),'xticklabel',allname,'xticklabelrotation',30)
        set(gca,'fontsize',15,'box','off','tickdir','out','linewidth',1,'ytick',[0,1])
        % plot Pexc
        hold on;
        [~,best] = max(tmp.Ef); pb = tmp.Ef(best); pb_sem = sqrt(tmp.Vf(best));
        L = 0.06; off = pb_sem*5.5; w = 1.2;
        plot(ones(1,2)*best,[pb,pb+L]+off,'k','linewidth',w); hold on;
        plot([1,3],ones(1,2)*(pb+L+off),'k','linewidth',w)
    end
    set(gca, 'color', 'none');
    if k==1
        ylabel({'Posterior';'model freq.'})
    end
end
