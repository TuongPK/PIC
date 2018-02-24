classdef MPC
    
    properties
        PC
        nPC
        output
    end
    
    methods
        
        % Constructor
        function net = MPC(nPC, nFeature, nTarget)
            if (nargin > 2)
                net.nPC = nPC;
                netStruct = [nFeature 100 100 100 nTarget];
                net.output = zeros(1, nTarget);
                
                for iPC = 1 : net.nPC
                    net.PC{iPC} = DMLP(netStruct, 0.01, 0.4);
                end
            end 
        end
        
        % Train network
        function out = train(net, nEpoch, trainData, trainTarget)
            % Grouping parameters
            trainParam = struct;
            trainParam.trainData = trainData; % train data
            trainParam.trainTarget = trainTarget; % train target
            trainParam.nSample = size(trainData, 1); % number of training sample
            trainParam.nTarget = size(trainTarget, 2); % number of category
            
            sosError = []; % total mean squared error

            % Training
            conMatrix = zeros(trainParam.nTarget);
            
            for iE = 1 : nEpoch
                epochError = 0;
                
                % Sequentiallu input samples
                for iS = 1 : trainParam.nSample
                    data = trainParam.trainData(iS, :);
                    target = trainParam.trainTarget(iS, :);

                    % Forward step
                    for iPC = 1 : net.nPC
                        net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(0.1);
                        
                        [net.PC{iPC}, out] = net.PC{iPC}.forward(data);
                        net.output = net.output + out;
                    end
                    
                    % Calculate ouput and error
                    net.output = tanh(net.output);
                    [vTarget, iTarget] = max(target);
                    [vOutput, iOutput] = max(net.output);
                                                        
                    conMatrix(iTarget, iOutput) = conMatrix(iTarget, iOutput) + 1;
                    epochError = epochError + sum((target - net.output) .^ 2) / 2;
                    
                    % Backpropagation
                    for iPc = 1 : net.nPC
                        net.PC{iPC} = net.PC{iPC}.backProp(net.output, target);
                    end
                end
                
                sosError = [sosError epochError];
            end
            
            plot(sosError);
            disp(conMatrix);
            out = net;
        end
        
        % Test network
        function test(net, testData, testTarget)
            nSample = size(testData, 1);
            nTarget = size(testTarget, 2);
                        
            conMatrix = zeros(nTarget);
            
            for iS = 1 : nSample
                input = testData(iS, :);
                target = testTarget(iS, :);
                sampleOutput = 0;
                % Forward step
                for iPC = 1 : net.nPC
                    net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(-1); % Use all
                    [net.PC{iPC}, out] = net.PC{iPC}.forward(input);
                    sampleOutput = sampleOutput + out;
                end

                % Calculate ouput and error
                net.output = tanh(sampleOutput);
                [vTarget, iTarget] = max(target);
                [vOutput, iOutput] = max(sampleOutput);
                
                conMatrix(iTarget, iOutput) = conMatrix(iTarget, iOutput) + 1;
            end
            
            disp(conMatrix);
        end
    end
end