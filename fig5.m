clear;clc;close all;
modeldir = './modelfits/';

Tmodel = load([modeldir,'Ternary_models_[2022_5_27_17_38_25].mat']);
Bmodel = load([modeldir,'Binary_models_[2022_5_27_16_9_53].mat']);

figure('position',[770   527   266   420])
T = []; B = [];
for s = 1:144
    T = [T; Tmodel.rankSI.outputFull_T{s}.Xfit];
    B = [B; Bmodel.rankSI.outputFull_B{s}.Xfit];
end
pB = B(:,2:4); % 1-SIw,k,theta
pT = [mean(T(:,2:3),2),T(:,4:5)]; % 1-SIw,k,theta
y = {
    {[1-pB(:,1),pB(:,2)],[1-pB(:,1),pB(:,3)]};
    {[1-pT(:,1),pT(:,2)],[1-pT(:,1),pT(:,3)]};
    };
name = {
        {'B SI{\it w}','B drift k','B bound \theta'},...
        {'T SI{\it w}','T drift k','T bound \theta'}
        };

fs = 16;
for i = 1:2
    subplot(2,1,i)
    input = y{i};
    lab = name{i};
    range1 = [0,10];
    range2 = [0,8];
    yc_scatter_dualYaxis(input,lab,range1,range2)
    set(gca,'fontsize',fs)
    set(gca,'color','none');
end


load([modeldir,'Fig6_staticSI_[2022_5_5_21_45_28].mat'])

B = param{1}(1:144,:); % binary
T = param{3}(1:144,:); % ternary (no monotonic constraint for w)
y = {
    [1-B(:,3),1./B(:,2)];
    [1-mean(T(:,3:4),2),1./T(:,2)]
    };
name = {{'B SI{\it w}','B decision noise (1/\beta)'},{'T SI{\it w}','T decision noise (1/\beta)'}};
n = length(y);
fs = 16;

figure('position',[503   527   266   420])
for i = 1:n
    subplot(2,1,i)
    input = y{i};
    lab = name{i};
    range = [0,.6];
    yc_scatter(input,lab,range)
    set(gca,'fontsize',fs)
    set(gca,'color','none');
end


%%
function yc_scatter_dualYaxis(input,lab,range1,range2)
% left:
scatter(input{1}(:,1),input{1}(:,2),50,ones(1,3)*0.2,'filled')
b = robustfit(input{1}(:,1),input{1}(:,2));
x = 0:1;
y = b(2)*x + b(1);
hold on;
plot(x,y,'r-','linewidth',1.5)
xlabel(lab{1}); ylabel(lab{2})
axis square; axis tight
ylim(range1)
[r,pval] = corr(input{1}(:,1),input{1}(:,2),'type','spearman');
pval = pval*2;

if pval>.05,pstar = '(n.s.)';
end
if pval<.05,pstar = '*';
end
if pval<.01,pstar = '**';
end
if pval<.001,pstar = '***';
end
XL = get(gca,'XLim');
YL = get(gca,'YLim');
str = ['r = ',num2str(round(r,3)),pstar];
text(XL(1)+(XL(2)-XL(1))*0.05,YL(2)-(YL(2)-YL(1))*0.05,str,'fontsize',16,'color','r')
set(gca,'tickdir','out','linewidth',1)
yyaxis right
grayc = ones(1,3)*0.7;
scatter(input{2}(:,1),input{2}(:,2),50,grayc)

b = robustfit(input{2}(:,1),input{2}(:,2));
x = 0:1;
y = b(2)*x + b(1);
hold on;
plot(x,y,'-','linewidth',1.5,'color',grayc)


ylabel(lab{3})
axis square; axis tight
offsetAxes
ylim(range2)
[r,pval] = corr(input{2}(:,1),input{2}(:,2),'type','spearman');
pval = pval*2;
if pval>.05,pstar = '(n.s.)';
end
if pval<.05,pstar = '*';
end
if pval<.01,pstar = '**';
end
if pval<.0001,pstar = '***';
end
XL = get(gca,'XLim');
YL = get(gca,'YLim');
str = ['r = ',num2str(round(r,3)),pstar];
disp(str)
text(XL(1)+(XL(2)-XL(1))*0.05,YL(1)+(YL(2)-YL(1))*0.05,str,'fontsize',16,'color',grayc)
set(gca,'tickdir','out','linewidth',1)
ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = grayc;

end

function yc_scatter(input,lab,range)
scatter(input(:,1),input(:,2),50,ones(1,3)*0.5,'filled')
b = robustfit(input(:,1),input(:,2));
x = 0:1;
y = b(2)*x + b(1);
hold on;
plot(x,y,'r-','linewidth',1.5)

xlabel(lab{1}); ylabel(lab{2})
axis square; axis tight
offsetAxes
ylim(range)
[r,pval] = corr(input(:,1),input(:,2),'type','spearman');
if pval>.05,pstar = '(n.s.)';
end
if pval<.05,pstar = '*';
end
if pval<.01,pstar = '**';
end
if pval<.0001,pstar = '***';
end
XL = get(gca,'XLim');
YL = get(gca,'YLim');
str = ['r = ',num2str(round(r,3)),pstar];
text(XL(1)+(XL(2)-XL(1))*0.05,YL(2)-(YL(2)-YL(1))*0.05,str,'fontsize',16,'color','b')
set(gca,'tickdir','out','linewidth',1)
set(gca,'ytick',linspace(range(1),range(2),3))
end