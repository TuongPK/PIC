tic;

clear all;
close all;

load fisheriris.mat;

nFeature = size(meas, 2);
nTarget = size(unique(species), 1);

net = MPC(2, nFeature, nTarget);

% Normalize
for i = 1 : nFeature
    col = meas(:,i);
    meas(:,i) = meas(:,i) / max(col);
end

% Unify data
termTrain = [meas(1 : 30, :) ; meas(51 : 80, :); meas(101 : 130, :)] ;
target = [repmat([1 0 0], 50, 1); repmat([0 1 0], 50, 1); repmat([0 0 1], 50, 1)];
termTarget = [target(1 : 30, :) ; target(51 : 80, :); target(101 : 130, :)] ;

% Randomize input flow
index = randperm(90);
trainData = zeros(90,4);
trainTarget = zeros(90,3);

for i = 1 : 90
    trainData(i,:) = termTrain(index(i),:);
    trainTarget(i,:) = termTarget(index(i),:);
end

% Train network
net = net.train(30, trainData, trainTarget); 

% Test network
testData = [meas(31 : 50, :) ; meas(81 : 100, :); meas(131 : 150, :)];
testTarget = [target(31 : 50, :) ; target(81 : 100, :); target(131 : 150, :)] ;
net.test(testData, testTarget);

toc;