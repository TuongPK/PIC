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
        seedMat
    end
    
    methods
        
        % Constructor
        function net = DMLP(netStruct, lRate, momentum, prob)
            if (nargin > 3)
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
                
                net.dropMap = net.generateMap(prob);
            end 
        end
        
        % Generate dropout map
        function map = generateMap(net, prob)
            for iL = 1 : net.nLayer
                if iL > 1
                    map{iL} = rand(1, net.netStruct(iL)) > prob;
                else
                    map{iL} = zeros(1, net.netStruct(iL)) + 1;
                end
            end
        end
   
        function [outNet, outVal] = forward(net, layer, seed)
            pLayer = layer - 1;
            cLayer = layer;
            
            net.acVal{cLayer} = (net.acVal{pLayer} .* net.dropMap{pLayer}) * net.weight{pLayer} + net.bias{cLayer} + seed;
            
            if (cLayer ~= net.nLayer) 
                net.acVal{cLayer} = sigmf(net.acVal{cLayer}, [1 0]);
            end

            outVal = net.acVal{cLayer};
            outNet = net;
        end
        
        function out = backProp(net, output, target)
            net.acVal{net.nLayer} = output;
            weight = net.weight;
            tempWeight = net.weight;
            
            for iL = 1 : (net.nLayer - 1)
                for iN = 1 : net.netStruct(iL)
                    if net.dropMap{iL}(iN) == 0
                        for i = 1 : net.netStruct(iL + 1) 
                            weight{iL}(iN, i) = net.seedMat{iL, iN}.weight(1, i);
                        end
                    end
                end
            end
            
            for iL = (net.nLayer : -1 : 1)
                neLayer = iL + 1;
                cLayer = iL;
                lOut = net.acVal{cLayer};

                if (cLayer == net.nLayer) % output layer
                    grad{cLayer} = (target - lOut) .* (1 - lOut) .* (1 + lOut);
                else
                    grad{cLayer} = lOut .* (1 - lOut) .* (grad{neLayer} * weight{cLayer}');
                end
            end

            for iL = 1 : (net.nLayer - 1)
                pLayer = iL;
                cLayer = iL + 1;

                dWeight{pLayer} = net.lRate .* ((net.acVal{pLayer} .* net.dropMap{pLayer})' * grad{cLayer});
                dBias{cLayer} = net.lRate .* grad{cLayer} .* net.dropMap{cLayer};

                weight{pLayer} = weight{pLayer} + dWeight{pLayer} + net.momentum .* net.preDWeight{pLayer};
                net.bias{cLayer} = net.bias{cLayer} + dBias{cLayer} + net.momentum .* net.preDBias{cLayer};
            end

            net.preDWeight = dWeight;
            net.preDBias = dBias;
            
            for iL = 1 : (net.nLayer - 1)
                for iN = 1 : net.netStruct(iL)
                    if net.dropMap{iL}(iN) == 0
                        net.seedMat{iL, iN}.weight = weight{iL}(iN, :);
                    
                        for i = 1 : net.netStruct(iL + 1)
                            net.weight{iL}(iN, i) = tempWeight{iL}(iN, i);
                        end
                    end
                end
            end
            
            out = net;
        end
    end
end