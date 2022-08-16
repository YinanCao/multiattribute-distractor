function [T,p,M,SEM] = Fast_Ttest(dat,nu,tail)

% subjects on first dimension of dat
if nargin < 3
    tail = 'both';
end

if nargin < 2
    nu = size(dat,1)-1;
end

Datmea = squeeze(nanmean(dat,1));
Datstd = squeeze(nanstd(dat,1,1));
if nu > 0
    Datden = sqrt((Datstd.^2)./nu);
    T = Datmea./Datden;
    if nargout > 1
        switch tail
            case 'right'
                % [h p ci stats]=ttest(x,0,'Tail','right');
                p = (1-tcdf(T,nu)); %right tailed mean>0
            case 'both'
                % [h p ci stats]=ttest(x,0,'Tail','both');
                p = tcdf(-abs(T),nu)*2; %two tailed mean ~=0
        end

        M = Datmea;
        SEM = squeeze(std(dat,0,1))./sqrt(nu+1);
    end
else
    T = dat;
    M = dat;
    p = 1;
    SEM = 0;

end

