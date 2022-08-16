function groupwidth = bar_yc_model(data,flag,vw_color)

data = squeeze(data);
model_series = squeeze(nanmean(data));
nsub = size(data,1);
model_error  = squeeze(nanstd(data))./sqrt(nsub);
if flag
    model_error = squeeze(nanstd(data));
end
b = bar(model_series,'grouped');
for k = 1:size(data,3)
    b(k).EdgeColor = vw_color(k,:);
    b(k).FaceColor = vw_color(k,:);
end
hold on;
[ngroups,nbars] = size(model_series);
groupwidth = min(0.8,nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups)-groupwidth/2+(2*i-1)*groupwidth/(2*nbars);
    y = model_series(:,i);
    se = model_error(:,i);
    err = [y+se,y-se]';
    for k = 1:size(err,2)
        hold on
        plot(ones(1,2)*x(k),err(:,k),'linewidth',1.5,'color',vw_color(i,:)*.9)
        hold on
    end
    hold on
end
hold off
end