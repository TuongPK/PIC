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
        cLayer
        seed
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
            for iL = 1 : net.nLayer
                map{iL} = rand(1, net.netStruct(iL)) > prob;
            end
        end
   
        function [outNet, outVal] = forward(net, layer, data)
            pLayer = layer - 1;
            cLayer = layer;
            
            net.acVal{cLayer} = (net.acVal{pLayer} .* net.dropMap{pLayer}) * net.weight{pLayer} + net.bias{cLayer};
            
            if (cLayer ~= net.nLayer) 
                net.acVal{cLayer} = sigmf(net.acVal{cLayer}, [1 0]);
            end

            outVal = net.acVal{cLayer};
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
    end
end