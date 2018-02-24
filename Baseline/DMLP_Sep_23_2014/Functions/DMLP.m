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
            nSample = size(trainData, 1); 
            nFeature = size(trainData, 2);  
            
            nNode = [(nFeature - 1) net.netStruct 1];   
            
            % Initialize parameters
            for iL = 1 : (size(nNode,2) - 1)
               net.weight{iL} = -0.5 + rand(nNode(iL), nNode(iL + 1));
               net.bias{iL} = -0.5 + rand(1, nNode(iL + 1));
            end
            
            nLayer = size(nNode,2);
            error = [];
            
            preDWeight = [];
            preDBias = [];
            
            % Iterate through epochs
            for i = 1 : nEpoch
                
                % Sequentially input samples
                for iS = 1 : nSample
                    data = trainData(iS, 1 : (nFeature - 1));
                    target = trainData(iS, nFeature);
                    
                    acVal{1} = data;
                                        
                    % Forward step
                    for iL = 1 : (nLayer - 1)
                        pLayer = iL;     
                        cLayer = iL + 1;  
                         
                        acVal{cLayer} = acVal{pLayer} * net.weight{pLayer} + net.bias{pLayer};
                        
                        if (cLayer ~= nLayer)   % Hidden layer
                            acVal{cLayer} = sigmf(acVal{cLayer}, [1 0]);
                        else    % Output layer
                            acVal{cLayer} = tanh(acVal{cLayer});
                        end
                    end
                    
                    % Backward step
                    for iL = (nLayer : -1 : 1)
                        neLayer = iL + 1;   
                        cLayer = iL;        
                        lOut = acVal{cLayer};
                        
                        if (cLayer == nLayer)   % Output layer
                            error = [error ((target - lOut) .^2)/2];
                            grad{cLayer} = (target - lOut) .* (1 - lOut) .* (1 + lOut);
                        else    % Hidden layer
                            grad{cLayer} = lOut .* (1 - lOut) .* (grad{neLayer} * net.weight{cLayer}');
                        end
                    end
                    
                    for iL = 1 : (nLayer - 1)
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