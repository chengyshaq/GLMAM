function [ L,Z] = UpdateZ(L,betag,WXXW,WXGXGW,param)
[l,k] = size(L);

manifold = elliptopefactory(l,k);
problem.M = manifold;

% Define the problem cost function and its gradient.
problem.cost = @(x) LCost(x, betag, WXXW,WXGXGW,param);
problem.grad = @(x) LGrad(x, betag, WXXW,WXGXGW,param);

options = param.tooloptions;
[x xcost info] = steepestdescent(problem,L,options);
L = x;
Z = L*L';
end

function cost = LCost(L,betag, WXXW,WXGXGW,param)
lambda5=param.lambda5;
lambda4=param.lambda4;
cost = 0.5*lambda4*betag*trace(WXXW*L*L')+0.5*lambda5*trace(WXGXGW*L*L');
end
function grad = LGrad(L,betag, WXXW,WXGXGW,param)
lambda5=param.lambda5;
lambda4=param.lambda4;
grad = lambda4*betag*WXXW*L+lambda5*WXGXGW*L;
end