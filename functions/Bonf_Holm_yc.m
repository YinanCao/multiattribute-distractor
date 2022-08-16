function [p_corr,h] = Bonf_Holm_yc(p,alpha)
    % Holm's seq Bonferroni correction
    % Yinan Cao
    % oxford 2019
    if sum(isnan(p))>0, error('NaN found in p list!'); end
    if size(p,1)>size(p,2), p = p'; end
    if nargin < 2
        alpha = .05;
    end
    Np = length(p);
    c_alpha = alpha./(Np:-1:1);
    [p_sorted, idx] = sort(p,'ascend'); % sort p vals
    i = find(p_sorted>c_alpha,1); h = ones(size(p));
    if ~isempty(i)
        h(idx(i:end)) = 0;
    end
    for i = Np:-1:1
        p_corr(idx(i)) = max(min([p_sorted(1:i).*(Np+1-(1:i));ones(1,i)],[],1));
    end
end