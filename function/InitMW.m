function [ M,W ] = InitMW(M,W,X,Y,param)
     [l,~] = size(M);
     [d,~] = size(W);
     obj_old = [];
     last = 0;
     
     for i=1:param.maxIter
        M = upm(M,X,Y,W,param,l) ;
        W = upw(W,X,Y,M,param,d,l);
        
        obj = 0.5*norm((X*W-Y*M),'fro')^2 + 0.5*param.lambda1*norm((Y*M-Y),'fro')^2;
        disp(obj);
        last = last + 1;
        obj_old = [obj_old;obj];
         if last < 5
             continue;
         end 
         stopnow = 1;
         for ii=1:3
             stopnow = stopnow & (abs(obj-obj_old(last-1-ii)) < 1e-3);
         end
         if stopnow
             break;
         end
     end
end
function W =upw(W,X,Y,M,param,d,l)
          manifold = euclideanfactory(d, l);
          problem.M = manifold;
          problem.cost = @(W) Wcost(W,X,Y,M,param);
          problem.grad = @(W) Wgrad(W,X,Y,M,param);
          options = param.tooloptions;
          [x xcost info] = steepestdescent(problem,W,options);
          W = x; 
end

function M= upm(M,X,Y,W,param,l) 
        manifold = euclideanfactory(l, l);
        problem.M = manifold;
        problem.cost = @(M) Mcost(M,X,Y,W,param);
        problem.grad = @(M) Mgrad(M,X,Y,W,param);
        options = param.tooloptions;     
        [x xcost info] = steepestdescent(problem,M,options);
        M = x;
end


function cost = Mcost(M,X,Y,W,param)
     lambdam = param.lambda1;
     lambda2 = param.lambda2;
     cost = 0.5*norm((X*W-Y*M),'fro')^2 + 0.5*lambdam*norm((Y*M-Y),'fro')^2 + lambda2*norm(M,'fro')^2;
end

function grad = Mgrad(M,X,Y,W,param)
     lambda1 = param.lambda1;
     lambda2 = param.lambda2;
     part1 = -Y'*(X*W-Y*M);
    part2 = lambda1*Y'*(Y*M-Y);
    grad = part1 + part2 + 2*lambda2*M;
end

function cost = Wcost(W,X,Y,M,param)
     lambdaw = param.lambda3;
     cost = 0.5*norm((X*W-Y*M),'fro')^2 + lambdaw*norm(W,'fro')^2;
end

function grad = Wgrad(W,X,Y,M,param)
     lambdaw = param.lambda3;
     grad = X'*(X*W-Y*M) + 2*lambdaw*W;
end