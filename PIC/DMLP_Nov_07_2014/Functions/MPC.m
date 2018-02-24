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
            if (nargin > 2)
                net.nPC = nPC;
                net.maxDepth = 0;
                net.output = zeros(1, nTarget);

                netStruct = [nFeature 50 50 nTarget];
                
                if size(netStruct, 2) > net.maxDepth
                    net.maxDepth = size(netStruct,2);
                end

                for iPC = 1 : net.nPC
                    net.PC{iPC} = DMLP(netStruct, 0.01, 0.4, 0.1);
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
            
            % Generate crossover seed
            for iPC = 1 : net.nPC
                map = net.PC{iPC}.dropMap;
                netStruct = net.PC{iPC}.netStruct;
                
                for iL = 2 : net.PC{iPC}.nLayer - 1
                    seedIndex = [];
                    for iN = 1 : netStruct(iL)
                        
                        if map{iL}(iN) == 0 % at each cross over point
                            seed = struct;
                            
                            % Select seed PC
                            seedPC = randi(net.nPC);
                            while seedPC == iPC
                                seedPC = randi(net.nPC);
                            end
                            
                            % For shallow PCs, pick from the top layer
                            if net.PC{seedPC}.nLayer - 1 < iL
                                seedLayer = net.PC{seedPC}.nLayer - 1;
                                val = net.PC{seedPC}.netStruct(seedLayer);
                            else
                                seedLayer = iL;
                                val = net.PC{seedPC}.netStruct(seedLayer);
                            end
                            
                            % Select seed node
                            seedNode = randi(val);
                            if size(seedIndex,1) > 0
                                while sum(ismember(seedIndex,[seedPC seedNode], 'rows')) > 0
                                    seedNode = randi(val);
                                end
                            end
                            
                            seedIndex = [seedIndex ; [seedPC seedNode]];
                             
                            seed.PC = seedPC;
                            seed.layer = seedLayer;
                            seed.node = seedNode;
                            seed.weight = zeros(1, netStruct(iL + 1));
                            
                            for i = 1 : netStruct(iL + 1)
                                seed.weight(1, i) = -0.5 + rand;
                            end
                            
                            net.PC{iPC}.seedMat{iL, iN} = seed;
                        end
                    end
                end
            end
                
            for iE = 1 : nEpoch
                epochError = 0;
                
                % Sequently input each sample
                for iS = 1 : trainParam.nSample
                    data = trainParam.trainData(iS, :);
                    target = trainParam.trainTarget(iS, :);
                    
                    % Forward step
                    for iPC = 1 : net.nPC
                        net.PC{iPC}.acVal{1} = data;
                    end
                    
                    % Layer by layer
                    for iL = 2 : net.maxDepth 
                        for iPC = 1 : net.nPC % PC by PC
                            if iL < net.PC{iPC}.nLayer % Check if has reach the top
                                % Forward
                                dropMap = net.PC{iPC}.dropMap{iL};
                                netStruct = net.PC{iPC}.netStruct(iL);
                                seed = [];
                                
                                for iN = 1 : netStruct
                                    seed(iN) = 0;
                                    if dropMap(iN) == 0
                                        seedPC = net.PC{iPC}.seedMat{iL, iN}.PC;
                                        seedLayer = net.PC{iPC}.seedMat{iL, iN}.layer;
                                        seedNode = net.PC{iPC}.seedMat{iL, iN}.node;
                                        seedWeight = net.PC{iPC}.seedMat{iL, iN}.weight;
                                        
                                        disp(net.PC{seedPC}.acVal{seedLayer});
                                        seed = net.PC{seedPC}.acVal{seedLayer}(seedNode) .* seedWeight;
                                    end
                                end
                                
                                [net.PC{iPC}, out] = net.PC{iPC}.forward(iL, seed);

                                if iL == (net.PC{iPC}.nLayer - 1) % Last layer before output
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
                
                
                % Forward step
                for iPC = 1 : net.nPC
                    % Generate drop map
                    net.PC{iPC}.dropMap = net.PC{iPC}.generateMap(-1);

                    % Input layer
                    [net.PC{iPC}, out] = net.PC{iPC}.forward(1, data);
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