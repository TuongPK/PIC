classdef DMLP
    
    properties
        weight
        bias
        netStruct
        nLayer
        lRate
        momentum
        acVal
        preDWeight
        preDBias
        dropMap
    end
    
    methods
        
        % Constructor
        function net = DMLP(netStruct, lRate, momentum)
            if (nargin > 1)
                net.netStruct = netStruct;
                net.lRate = lRate;
                net.momentum = momentum;
                net.nLayer = size(netStruct,2);
                
                for iL = 1 : net.nLayer
                    if (iL < net.nLayer)
                        net.weight{iL} = -0.5 + rand(net.netStruct(iL), net.netStruct(iL + 1));
                        net.preDWeight{iL} = zeros(net.netStruct(iL), net.netStruct(iL + 1));
                    end
                    net.bias{iL} = -0.5 + rand(1, net.netStruct(iL));
                    net.preDBias{iL} = zeros(1, net.netStruct(iL));
                end
            end 
        end
        
        % Generate dropout map
        function map = generateMap(net, prob)
            for iL = 1 : (net.nLayer - 1)
                map{iL} = rand(1, net.netStruct(iL)) > prob;
            end
        end
               
        function [outNet, outVal] = forward(net, data)
            net.acVal{1} = data;
                                     
            % Forward step
            for iL = 1 : (net.nLayer - 1)
                pLayer = iL;
                cLayer = iL + 1;
                net.acVal{cLayer} = (net.acVal{pLayer} .* net.dropMap{pLayer}) * net.weight{pLayer} + net.bias{cLayer};
                        
                if (cLayer ~= net.nLayer) 
                    net.acVal{cLayer} = sigmf(net.acVal{cLayer}, [1 0]);
                end
            end
            
            outVal = net.acVal{net.nLayer};
            outNet = net;
        end
        
        function out = backProp(net, output, target)
            net.acVal{net.nLayer} = output;
            
            for iL = (net.nLayer : -1 : 1)
                neLayer = iL + 1;
                cLayer = iL;
                lOut = net.acVal{cLayer};

                if (cLayer == net.nLayer) % output layer
                    grad{cLayer} = (target - lOut) .* (1 - lOut) .* (1 + lOut);
                else
                    grad{cLayer} = lOut .* (1 - lOut) .* (grad{neLayer} * net.weight{cLayer}');
                end
            end

            for iL = 1 : (net.nLayer - 1)
                pLayer = iL;
                cLayer = iL + 1;

                dWeight{pLayer} = net.lRate .* ((net.acVal{pLayer} .* net.dropMap{pLayer})' * grad{cLayer});
                dBias{cLayer} = net.lRate .* grad{cLayer};

                net.weight{pLayer} = net.weight{pLayer} + dWeight{pLayer} + net.momentum .* net.preDWeight{pLayer};
                net.bias{cLayer} = net.bias{cLayer} + dBias{cLayer} + net.momentum .* net.preDBias{cLayer};
            end

            net.preDWeight = dWeight;
            net.preDBias = dBias;
            
            out = net;
        end
        
        function [outNet, outVal] = pretrain(net, input, layer, trainParam, opParam)
            wChange = zeros(net.netStruct(layer), net.netStruct(layer + 1));
            pBChange = zeros(1, net.netStruct(layer));
            cBChange = zeros(1, net.netStruct(layer + 1));

            for ep = 1 : opParam.nEpoch
                for iS = 1 : trainParam.nSample
                    % Positive phase
                    posInp = input(iS, :);

                    posHidProb = sigmf(posInp * net.weight{layer} + net.bias{layer + 1}, [1 0]);
                    posProduct = posInp' * posHidProb;
                    posHidAct = sum(posHidProb);
                    posVisAct = sum(posInp);
                    posHidState = posHidProb > rand(1, net.netStruct(layer + 1));

                    % Negative phase
                    negInp = sigmf(posHidState * net.weight{layer}' + net.bias{layer}, [1 0]);
                    negHidProb = sigmf(negInp * net.weight{layer} + net.bias{layer + 1}, [1 0]);
                    negProduct = negInp' * negHidProb;
                    negHidAct = sum(negHidProb);
                    negVisAct = sum(negInp);

                    wChange = opParam.momentum * wChange + opParam.lRate * (posProduct - negProduct);
                    pBChange = opParam.momentum * pBChange + opParam.lRate * (posVisAct - negVisAct);
                    cBChange = opParam.momentum * cBChange + opParam.lRate * (posHidAct - negHidAct);

                    net.weight{layer} = net.weight{layer} + wChange;
                    net.bias{layer} = net.bias{layer} + pBChange;
                    net.bias{layer + 1} = net.bias{layer + 1} + cBChange;
                end
            end
            
            outVal = sigmf(input * net.weight{layer} + repmat(net.bias{layer + 1}, trainParam.nSample, 1), [1 0]);
            outNet = net;
        end
    end
end