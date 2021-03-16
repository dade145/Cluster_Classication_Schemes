classdef NCC
    
    properties
        mu; %2D-Matrix, one column for each class
        n_classes;
        correction;
    end
    
    methods
        function obj = NCC(n_classes, correction)
            %n_classes: number of classes into which classisy the data
            %correction: label correction, 1 for MNIST dataset 0 for ORL
            obj.mu = [];
            obj.n_classes = n_classes;
            obj.correction = correction;
        end
        
        function [obj,time] = fit(obj,X, Y)
            %X: training samples
            %Y: training labels
            tic;
            m = zeros(size(X,1), obj.n_classes);
            for i=1:obj.n_classes
                idx = Y == i-obj.correction;
                m(:,i)= mean(X(:,idx),2);
            end
            obj.mu = m;
            time=toc;
        end
        
        function [acc,time] = score(obj, X, Y)
            %X: test samples
            %Y: test labels
            tic;            
            dist = zeros(obj.n_classes, size(X,2));
            for k=1:obj.n_classes
                %Matrix computation
                dist(k,:) = vecnorm(X - obj.mu(:,k)).^2;
            end
            [~,res] = min(dist,[],1);
            res = res-obj.correction;
            
            %Compute accuracy
            %If diff is zero, the labels are equal and so correct
            diff = find(Y-res' == 0);
            acc = size(diff,1)/size(Y,1);
            time=toc;
        end
        
    end
end

