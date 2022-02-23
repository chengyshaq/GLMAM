function [M,WXXW,WXGXGWPool] = UpdateM(M,X,Y,new_Y,W,XX,XGXGPool,MgPool,betav,param)
[l,k] = size(M);

manifold = elliptopefactory(l,k);
problem.M = manifold;

% Define the problem cost function and its gradient.
problem.cost = @(x) MCost(x,X,Y,new_Y,W,XX,XGXGPool,MgPool,betav,param);
problem.grad = @(x) MGrad(x,X,Y,new_Y,W,XX,XGXGPool,MgPool,betav,param);

options = param.tooloptions;
[x xcost info] = steepestdescent(problem,M,options);
M = x;
WXXW = M'*W'*XX*W*M;
WXGXGWPool = cell(size(MgPool));
for i=1:length(MgPool)
    WXGXGWPool{i} = M'*W'*XGXGPool{i}*W*M;
end
end

function cost = MCost(M,X,Y,new_Y,W,XX,XGXGPool,MgPool,betav,param)
lambda1=param.lambda1;
lambda2=param.lambda2;
lambda4=param.lambda4;
lambda5=param.lambda5;

cost = 0.5*norm((X*W-new_Y*M),'fro')^2 + 0.5*lambda1*norm((new_Y*M-Y),'fro')^2 + lambda2*norm(M,'fro')^2;
MWXXWM = M'*W'*XX*W*M;
for i=1:length(MgPool)
    Mg = MgPool{i};
    WXGXGW = M'*W'*XGXGPool{i}*W*M;
    part_local = lambda4*betav(i)*trace(MWXXWM*MgPool{i}) + lambda5*trace(WXGXGW*Mg);
    cost = cost + part_local;
end
end
function grad = MGrad(M,X,Y,new_Y,W,XX,XGXGPool,MgPool,betav,param)
lambda1=param.lambda1;
lambda2=param.lambda2;
lambda4=param.lambda4;
lambda5=param.lambda5;

part1 = -new_Y'*(X*W-new_Y*M);
part2 = lambda1*new_Y'*(new_Y*M-Y);
grad = part1 + part2 + 2*lambda2*M;
for i=1:length(MgPool)
    part_local = (lambda4*betav(i)*W'*XX*W + lambda5*W'*XGXGPool{i}*W)*M*MgPool{i};
    grad = grad + part_local;
end
end