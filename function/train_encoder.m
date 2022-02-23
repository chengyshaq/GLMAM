function [new_Y] = train_encoder( X,Y,Num)
[num_train,~] = size(X);
[~,num_label] = size(Y);
dist_max = diag(realmax*ones(1,num_train));
temp_dist = pdist(X);
dist = squareform(temp_dist)+dist_max;

new_Y = zeros(num_train,num_label);
temp = Num:-1:1;
my = ((repmat(temp',1,num_label))-0.5)./Num;
for i = 1:num_train
    [~,index] = sort(dist(i,:));
    label_neighbor_index = index(1:Num);
    label_neighbor = Y(label_neighbor_index,:).*my;
    new_Y(i,:) = sum(label_neighbor);
end
end