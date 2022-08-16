function V = getVIF(X)

npred = size(X,2);
V = nan(npred,1);
if npred > 2
    for i = 1:npred
        other = setdiff(1:npred,i);
        Rsq = OLS(X(:,i),X(:,other));
        V(i) = 1/(1-Rsq);
    end
else
    Rsq = OLS(X(:,1),X(:,2));
    V(1) = 1/(1-Rsq);
    V(2) = V(1);
end

function Rsq = OLS(Y,X)

n = size(X,1);
X = [ones(n,1), X];
y = X*pinv(X'*X)*X'*Y;
res = Y - y;
SSres = sum(res.^2);
SStot = var(Y,1)*n;
Rsq = 1 - (SSres/SStot);