classdef DMLP
    
    properties
        weight
        bias
        netStruct
        lRate
        momentum
    end
    
    methods
        
        % Constructor
        function net = DMLP(netStruct, lRate, momentum)
            if (nargin > 2)
                net.netStruct = netStruct;
                net.lRate = lRate;
                net.momentum = momentum;
            end 
        end
 
        % Train network
        function out = train(net, nEpoch, trainData)
            
            % Pre-training
            function out = preTrain(net, trainParam, opParam)
                input = trainParam.data(:, 1 : (trainParam.nFeature - 1));

                for iL = 1 : trainParam.nLayer
                    if (iL < trainParam.nLayer)
                        net.weight{iL} = -0.5 + rand(trainParam.nNode(iL), trainParam.nNode(iL + 1));
                    end
                    net.bias{iL} = -0.5 + rand(1, trainParam.nNode(iL));
                end

                for iL = 1 : (trainParam.nLayer - 1)
                    errSum = 0;
                    wChange = zeros(trainParam.nNode(iL), trainParam.nNode(iL + 1));
                    pBChange = zeros(1, trainParam.nNode(iL));
                    cBChange = zeros(1, trainParam.nNode(iL + 1));

                    for ep = 1 : opParam.nEpoch

                        for iS = 1 : trainParam.nSample

                            % Positive phase
                            posInp = input(iS, :);

                            posHidProb = sigmf(posInp * net.weight{iL} + net.bias{iL + 1}, [1 0]);
                            posProduct = posInp' * posHidProb;
                            posHidAct = sum(posHidProb);
                            posVisAct = sum(posInp);
                            posHidState = posHidProb > rand(1, trainParam.nNode(iL + 1));

                            % Negative phase
                            negInp = sigmf(posHidState * net.weight{iL}' + net.bias{iL}, [1 0]);
                            negHidProb = sigmf(negInp * net.weight{iL} + net.bias{iL + 1}, [1 0]);
                            negProduct = negInp' * negHidProb;
                            negHidAct = sum(negHidProb);
                            negVisAct = sum(negInp);

                            err = sum(sum((posInp - negInp).^2));
                            errSum = errSum + err;

                            wChange = opParam.momentum * wChange + opParam.lRate * (posProduct - negProduct);
                            pBChange = opParam.momentum * pBChange + opParam.lRate * (posVisAct - negVisAct);
                            cBChange = opParam.momentum * cBChange + opParam.lRate * (posHidAct - negHidAct);

                            net.weight{iL} = net.weight{iL} + wChange;
                            net.bias{iL} = net.bias{iL} + pBChange;
                            net.bias{iL + 1} = net.bias{iL + 1} + cBChange;
                        end
                    end
                    input = sigmf(input * net.weight{iL} + repmat(net.bias{iL + 1}, trainParam.nSample, 1), [1 0]);
                end          

                net.bias(1) = [];
                out = net;
            end
            
            % Grouping parameters
            trainParam = struct;
            trainParam.data = trainData;
            trainParam.nSample = size(trainParam.data, 1);
            trainParam.nFeature = size(trainParam.data, 2);
            trainParam.nNode = [(trainParam.nFeature - 1) net.netStruct 1];
            trainParam.nLayer = size(trainParam.nNode,2);
            error = [];
            
            % RBM-like pre-training
            opParam = struct;
            opParam.lRate = 0.01;
            opParam.momentum = 0.2;
            opParam.nEpoch = 5;
            
            net = preTrain(net, trainParam, opParam);

            preDWeight = [];
            preDBias = [];
            
            % Iterate through epochs
            for i = 1 : nEpoch
                
                % Sequentially input samples
                for iS = 1 : trainParam.nSample
                    data = trainParam.data(iS, 1 : (trainParam.nFeature - 1));
                    target = trainParam.data(iS, trainParam.nFeature);
                    
                    acVal{1} = data;
                                        
                    % Forward step
                    for iL = 1 : (trainParam.nLayer - 1)
                        pLayer = iL;
                        cLayer = iL + 1;
                        acVal{cLayer} = acVal{pLayer} * net.weight{pLayer} + net.bias{pLayer};
                        
                        if (cLayer ~= trainParam.nLayer)    % Hidden layer
                            acVal{cLayer} = sigmf(acVal{cLayer}, [1 0]);
                        else    % Output layer
                            acVal{cLayer} = tanh(acVal{cLayer});
                        end
                    end
                    
                    % Backward step
                    for iL = (trainParam.nLayer : -1 : 1)
                        neLayer = iL + 1;
                        cLayer = iL;
                        lOut = acVal{cLayer};
                        
                        if (cLayer == trainParam.nLayer) % Output layer
                            error = [error ((target - lOut) .^2)/2];
                            grad{cLayer} = (target - lOut) .* (1 - lOut) .* (1 + lOut);
                        else    % Hidden layer
                            grad{cLayer} = lOut .* (1 - lOut) .* (grad{neLayer} * net.weight{cLayer}');
                        end
                    end
                    
                    for iL = 1 : (trainParam.nLayer - 1)
                        pLayer = iL;
                        cLayer = iL + 1;
                        
                        dWeight{pLayer} = net.lRate .* (acVal{pLayer}' * grad{cLayer});
                        dBias{pLayer} = net.lRate .* grad{cLayer};
                    
                        net.weight{pLayer} = net.weight{pLayer} + dWeight{pLayer};
                        net.bias{pLayer} = net.bias{pLayer} + dBias{pLayer};
                        
                        if (i * iS ~= 1) % Not first input
                            net.weight{pLayer} = net.weight{pLayer} + net.momentum .* preDWeight{pLayer};
                            net.bias{pLayer} = net.bias{pLayer} + net.momentum .* preDBias{pLayer};
                        end
                    end
                    
                    preDWeight = dWeight;
                    preDBias = dBias;
                    
                end
            end
            plot(error);
            
            out = net;
        end
        
        % Test network
        function test(net)
        end    
    end
end