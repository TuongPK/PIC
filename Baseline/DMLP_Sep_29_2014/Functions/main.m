% XOR dataset
data = [[0 0 0]; 
        [1 0 1]; 
        [0 1 1]; 
        [1 1 0]];
    
% Network initilization
netStruct = [10 10];
net = DMLP(netStruct, 0.01, 0.4);

% Training
train(net, 50, data); 