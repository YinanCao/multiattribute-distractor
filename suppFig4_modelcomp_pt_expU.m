clear; clc; close all;

parentdir = './modelfits/';
linD = load([parentdir,'Binary_models_[2022_5_27_16_9_53].mat']);
linS = load([parentdir,'StaticBinaryModel_Linear_[2022_2_26_17_3_16].mat']);
load([parentdir,'Binary_prospect_[2022_11_20_22_42_45].mat'])

opt = [];
opt.DisplayWin = 0;

model = {
{EVnlD{1}, EVnlD{2}, EVnlD{4}, linD.AU_CI}
{EVnlS{1}, EVnlS{2}, EVnlS{4}, linS.AU}
};
modelNames = {'EV','Expected Utility','Prospect Theory','AU'};
figure('position',[840   506   507   239])
ns = 144;
for k = 1:2
    subplot(1,2,k)
    thism = model{k};
    xx = [];
    for j = 1:numel(thism)
        xx = cat(2,xx,thism{j}.testLL_B(1:ns,:,:));
        for s = 1:ns
            fitg(s,1,j,k) = thism{j}.outputFull_B{s}.BIC;
        end
    end
    xx = mean(xx,3); % average across cv folds
    fitg(:,2,:,k) = xx;

    [~,out1] = VBA_groupBMC(xx',opt);
    out = {out1};
    allname = modelNames;
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
        L = 0.05; off = pb_sem*3.; w = 1.2;
        plot(ones(1,2)*best,[pb,pb+L]+off,'k','linewidth',w); hold on;
        plot([1,4],ones(1,2)*(pb+L+off),'k','linewidth',w)
    end
    set(gca, 'color', 'none');
    if k==1
        ylabel({'Posterior';'model freq.'})
    end
end

ff = squeeze(sum(fitg,1));

delta = squeeze(ff-ff(:,end,:));

for v = 1:2 % dyna, static
    array2table(delta(:,:,v))
end








