function l = err_yc(input,style,color)
    ns = size(input,1);
    M = nanmean(input);
    SE = nanstd(input)/sqrt(ns);
    x = 1:length(M);
    y = M';
    se = SE';
    err = [y+se,y-se]';
    l = plot(x,M,style,'color',color,'linewidth',1.5,...
        'markersize',8,'markerfacecolor',color);
    for k = 1:size(err,2)
        hold on
        plot(ones(1,2)*(x(k)),err(:,k),'linewidth',1.5,'color',color)
    end
    hold on
end