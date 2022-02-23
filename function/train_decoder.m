function [W,M,MgPool,obj_old] = train_decoder(X,Ymis,new_Y,param)
[n,d] = size(X);
[l,~] = size(Ymis);
param.maxIter = 50;
param.tooloptions.maxiter = 60;
param.tooloptions.gradnorm = 1e-5;
W = rand(d,l);
M = zeros(l,l);
g = param.g;

[T, ~] = kmeans(X,param.g,'emptyaction','drop');
tic;
[betav, XGXGPool, XX, param ] = InitGroup( Ymis, X',T, param );
MgPool = cell(g,1);
LgPool = cell(g,1);
for i = 1:g
    LgPool{i} = rand(l,d);
end
[M,W] = InitMW(M,W,X,Ymis',param);
obj_old = [];
last = 0;

WXXW = M'*W'*XX*W*M;
WXGXGWPool = cell(g,1);
for i = 1:g
    WXGXGWPool{i} = M'*W'*XGXGPool{i}*W*M;
end

for i=1:param.maxIter
    for gr=1:g
        WXGXGW = WXGXGWPool{gr};
        [Lg] = UpdateZ(LgPool{gr},betav(gr),WXXW,WXGXGW,param);
        LgPool{gr} = Lg;
        MgPool{gr} = Lg*Lg';
    end
    [ M ] = UpdateM(M,X,Ymis',new_Y,W,XX,XGXGPool,MgPool,betav,param);
    [ W,WXXW,WXGXGWPool ] = UpdateW(W,X,Ymis',new_Y,M,XX,XGXGPool,MgPool,betav,param);
    
    obj = 0.5*norm((X*W-Ymis'*M),'fro')^2 + 0.5*param.lambda1*norm((new_Y*M-Ymis'),'fro')^2;
    disp(obj);
    last = last + 1;
    obj_old = [obj_old;obj];
    if last < 5
        continue;
    end
    stopnow = 1;
    for ii=1:3
        stopnow = stopnow & (abs(obj-obj_old(last-1-ii)) < 1e-6);
    end
    if stopnow
        break;
    end
end
end