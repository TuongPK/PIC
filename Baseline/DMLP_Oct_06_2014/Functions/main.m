tic;

close all;
clear all;

% Network initilization
netStruct = [50 50 50];
net = DMLP(netStruct, 0.01, 0.4);

% Load dataset
load fisheriris.mat;

% Normalize
nFeature = size(meas, 2);

for i = 1 : nFeature
    col = meas(:,i);
    meas(:,i) = meas(:,i) / max(col);
end


% Training

% Select first 30 samples for training
termTrain = [meas(1 : 30, :) ; meas(51 : 80, :); meas(101 : 130, :)] ;

% Create one-hot class
termTarget = zeros(150,3);

% Assign target class
indexTarget = zeros(150, 1);
indexTarget(1:50) = indexTarget(1:50) + 1;
indexTarget(51:100) = indexTarget(51:100) + 2;
indexTarget(101:150) = indexTarget(101:150) + 3;

for i = 1 : 150
    termTarget(i, indexTarget(i)) = 1;
end

% Target for training
finalTarget = [termTarget(1 : 30, :) ; termTarget(51 : 80, :); termTarget(101 : 130, :)] ;

% Shuffle
index = randperm(90);

% Create training set
trainData = zeros(90,4);
trainTarget = zeros(90,3);

for i = 1 : 90
    trainData(i,:) = termTrain(index(i),:);
    trainTarget(i,:) = finalTarget(index(i),:);
end

net = train(net, 1000, trainData, trainTarget); 

% Testing
testData = [meas(31 : 50, :) ; meas(81 : 100, :); meas(131 : 150, :)];
testTarget = [termTarget(31 : 50, :) ; termTarget(81 : 100, :); termTarget(131 : 150, :)] ;

test(net, testData, testTarget);

toc;