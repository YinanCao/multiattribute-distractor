function bar_yc_human(data,flag,vw_color)
data = squeeze(data);
model_series = squeeze(nanmean(data));
nsub = size(data,1);
model_error  = squeeze(nanstd(data))./sqrt(nsub);
if flag
    model_error  = squeeze(nanstd(data));
end
hold on;
[ngroups, nbars] = size(model_series);
groupwidth = min(0.8, nbars/(nbars + 1.5));
of = [-1,1]*0.06;
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    y = model_series(:,i);
    se = model_error(:,i);
    err = [y+se,y-se]';
    for k = 1:size(err,2)
        hold on
        plot(ones(1,2)*x(k)+of(i),y(k),'o','linewidth',1.5,'color',vw_color(i,:)*0.9);
        hold on
        plot(ones(1,2)*x(k)+of(i),err(:,k),'linewidth',1,'color',vw_color(i,:))
        hold on
    end
    hold on
end
hold off
end
