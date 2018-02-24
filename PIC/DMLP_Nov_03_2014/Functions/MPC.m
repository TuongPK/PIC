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
                net.output = zeros(1, nTarget);
                
                netStruct = [nFeature 50 50 nTarget];
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
                epochError = 0;
                
                % Sequently input each sample
                for iS = 1 : trainParam.nSample
                    data = trainParam.trainData(iS, :);
                    target = trainParam.trainTarget(iS, :);
                    
                    % Generate crossover map 
                    for iPC = 1 : net.nPC
                        net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(0.2); 
                    end
                    
                    % Forward step
                    for iL = 1 : net.maxDepth % Layer by layer
                        for iPC = 1 : net.nPC % PC by PC
                        
                            if iL == 1 % Input layer
                                net.PC{iPC}.acVal{iL} = data;
                                net.PC{iPC}.cLayer = 0;
                            elseif iL <= net.PC{iPC}.nLayer % Check if has reach the top
                                
                                % Crossover forward
                                dropMap = net.PC{iPC}.dropMap;
                                netStruct = net.PC{iPC}.netStruct;

                                seedIndex = [];
                                seed = zeros(1, netStruct(net.PC{iPC}.cLayer)); % Seed to add
                                                               
                                for iN = 1 : netStruct(net.PC{iPC}.cLayer)
                                    if dropMap{net.PC{iPC}.cLayer}(iN) == 0
                                        seedPC = randi(net.nPC, 1);
                                        seedNode = net.PC{iPC}.getSeed();
                                        
                                        % To ensure node from other PC is
                                        % swapped
                                        while (seedPC == iPC)
                                            seedPC = randi(net.nPC, 1);
                                            
                                            % To ensure new node is swapped
                                            if size(seedIndex,1) > 0
                                                while (seedIndex(sum(ismember([seedPC seedNode], seedIndex, 'rows')) > 0))
                                                    seedNode = net.PC{iPC}.getSeed();
                                                end
                                            end
                                        end

                                        seedIndex = [seedIndex; [seedPC seedNode]];
                                        seed(1,iN) = net.PC{iPC}.getSeed(seedNode) * (-0.5 + rand(1));
                                    end
                                end

                                [net.PC{iPC}, out] = net.PC{iPC}.forward(iL, seed);

                                if iL == net.PC{iPC}.nLayer % Last layer before output
                                    net.output = net.output + out;
                                end
                            end
                        end
                        
                        for iPC = 1 : net.nPC
                            if net.PC{iPC}.cLayer < net.PC{iPC}.nLayer
                                net.PC{iPC}.cLayer = net.PC{iPC}.cLayer + 1;
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

                % Forward step
                for iL = 1 : net.maxDepth % Layer by layer
                    for iPC = 1 : net.nPC % PC by PC
                        if iL == 1 % Input layer
                            net.PC{iPC}.acVal{iL} = data;
                        elseif iL <= net.PC{iPC}.nLayer % Check if has reach the top
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