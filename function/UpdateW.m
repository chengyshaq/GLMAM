function [W,WXXW,WXGXGWPool] = UpdateW(W,X,Y,new_Y,C,XX,XGXGPool,MgPool,betav,param)
[d,l] = size(W);

manifold = elliptopefactory(d,l);
problem.M = manifold;

% Define the problem cost function and its gradient.
problem.cost = @(x) Wcost(x,X,Y,new_Y,C,XX,XGXGPool,MgPool,betav,param);
problem.grad = @(x) Wgrad(x,X,Y,new_Y,C,XX,XGXGPool,MgPool,betav,param);

options = param.tooloptions;
[x xcost info] = steepestdescent(problem,W,options);
W = x;
WXXW = C'*W'*XX*W*C;
WXGXGWPool = cell(size(MgPool));
for i=1:length(MgPool)
    WXGXGWPool{i} = C'*W'*XGXGPool{i}*W*C;
end
end

function cost = Wcost(W,X,Y,new_Y,C,XX,XGXGPool,MgPool,betav,param)

lambda3 = param.lambda3;
lambda4 = param.lambda4;
lambda5 = param.lambda5;

cost = 0.5*norm((X*W-new_Y*C),'fro')^2 + lambda3*norm(W,'fro');
CWXXWC = C'*W'*XX*W*C;
for i=1:length(MgPool)
    Mg = MgPool{i};
    CWXGXGWC = C'*W'*XGXGPool{i}*W*C;
    cost = cost + lambda4*betav(i)*trace(CWXXWC*Mg) ...
        + lambda5*trace(CWXGXGWC*Mg);
end
end
function grad = Wgrad(W,X,Y,new_Y,C,XX,XGXGPool,MgPool,betav,param)
lambda3 = param.lambda3;
lambda4 = param.lambda4;
lambda5 = param.lambda5;

grad = X'*(X*W-new_Y*C)+2*lambda3*W;
for i=1:length(MgPool)
    Mg = MgPool{i};
    grad = grad + (lambda4*betav(i)*XX + lambda5*XGXGPool{i})*W*C*Mg*C';
end
end