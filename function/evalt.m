function [ Result ] = evalt(Fpred,Ygnd,thr)
 Ypred = sign(Fpred-thr);

 %% Average Precision
AvgPrec = Average_precision(Fpred,Ygnd);
Result.AveragePrecision = AvgPrec;

%% Coverage
Cvg = coverage(Fpred,Ygnd);
Result.Coverage = Cvg;

%% One Error
OE = One_error(Fpred,Ygnd);
Result.OneError = OE;

%% Ranking Loss
RkL = Ranking_loss(Fpred,Ygnd);
Result.RankingLoss = RkL;

Result.AvgAuc = avgauc(Fpred,Ygnd);