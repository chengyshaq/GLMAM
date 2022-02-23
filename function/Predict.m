function [TY,resluts] = Predict(W,Xt,Yt)
TY = Xt*W;
resluts =  evalt(TY',Yt, (max(TY(:))-min(TY(:)))/2);
end