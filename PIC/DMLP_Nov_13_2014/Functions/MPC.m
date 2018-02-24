classdef MPC
    
    properties
        PC
        nPC
        output
        maxDepth
    end
    
    methods
        
        % Constructor
        function net = MPC(nPC, nFeature, nTarget)
            if (nargin > 1)
                net.nPC = nPC;
                net.maxDepth = 0;
                
                netStruct = [nFeature 50 50 50 nTarget];
                
                if size(netStruct, 2) > net.maxDepth
                    net.maxDepth = size(netStruct,2);
                end

                for iPC = 1 : net.nPC
                    net.PC{iPC} = DMLP(netStruct, 0.01, 0.4);
                end
            end 
        end
        
        % Train network
        function out = train(net, nEpoch, trainData, trainTarget)
            % Grouping data
            trainParam = struct;
            trainParam.trainData = trainData; % train data
            trainParam.trainTarget = trainTarget; % train target
            trainParam.nSample = size(trainData, 1); % number of training sample
            trainParam.nTarget = size(trainTarget, 2); % number of category
            
            sosError = []; % total mean squared error
            
            % Training
            conMatrix = zeros(trainParam.nTarget);
            
            for iE = 1 : nEpoch
                disp(iE);
                epochError = 0;
                
                % Sequently input each sample
                for iS = 1 : trainParam.nSample
                    data = trainParam.trainData(iS, :);
                    target = trainParam.trainTarget(iS, :);
                    net.output = zeros(1, size(target,2));
                    
                    % Forward step
                    for iPC = 1 : net.nPC
                        % Generate drop map
                        net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(0.2);
                        
                        % Input layer
                        net.PC{iPC}.acVal{1} = data;
                    end
                    
                    % Layer by layer
                    for iL = 2 : net.maxDepth 
                        for iPC = 1 : net.nPC % PC by PC
                            if iL <= net.PC{iPC}.nLayer % Check if has reach the top
                                % Forward
                                [net.PC{iPC}, out] = net.PC{iPC}.forward(iL);

                                if iL == net.PC{iPC}.nLayer % Last layer before output
                                    net.output = net.output + out;
                                end
                            end
                        end
                    end
                    
                    % Calculate ouput and error
                    net.output = tanh(net.output);
                    [vTarget, iTarget] = max(target);
                    [vOutput, iOutput] = max(net.output);
                                                        
                    conMatrix(iTarget, iOutput) = conMatrix(iTarget, iOutput) + 1;
                    epochError = epochError + sum((target - net.output) .^ 2) / 2;
                    
                    % Backpropagation
                    for iPC = 1 : net.nPC
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
            
            % Sequently input each sample
            for iS = 1 : nSample
                data = testData(iS, :);
                target = testTarget(iS, :);
                net.output = zeros(1, nTarget);
                
                % Forward step
                for iPC = 1 : net.nPC
                    % Generate drop map
                    net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(-1);

                    % Input layer
                    net.PC{iPC}.acVal{1} = data;
                end

                % Layer by layer
                for iL = 2 : net.maxDepth 
                    for iPC = 1 : net.nPC % PC by PC
                        if iL <= net.PC{iPC}.nLayer % Check if has reach the top
                            % Forward
                            [net.PC{iPC}, out] = net.PC{iPC}.forward(iL);

                            if iL == net.PC{iPC}.nLayer % Last layer before output
                                net.output = net.output + out;
                            end
                        end
                    end
                end

                % Calculate ouput and error
                net.output = tanh(net.output);
                [vTarget, iTarget] = max(target);
                [vOutput, iOutput] = max(net.output);

                conMatrix(iTarget, iOutput) = conMatrix(iTarget, iOutput) + 1;
            end
            
            disp(conMatrix);
        end
    end
end