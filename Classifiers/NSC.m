classdef NSC
    
    properties
        mu; %3D Matrix, one page per each class, one column per subclass
        n_classes;
        n_subclasses;
        correction;
    end
    
    methods
        function obj = NSC(n_classes, k, correction)
            %n_classes: number of classes into which classisy the data
            %k: number of subcalsses
            %correction: label correction, 1 for MNIST dataset 0 for ORL
            obj.mu = [];
            obj.n_classes = n_classes;
            obj.correction = correction;
            obj.n_subclasses = k;
        end
        
        function [obj,time] = fit(obj,X, Y)
            %X: training samples
            %Y: training labels
            tic;
            m = zeros(size(X,1), obj.n_subclasses);
            for i=1:obj.n_classes
                idx = Y == i-obj.correction;
                m(:, :, i) = kmeans(X(:,idx), obj.n_subclasses);
            end
            obj.mu = m;
            time=toc;
        end
    
        function [acc,time] = score(obj, X, Y)
            %X: test samples
            %Y: test labels
            tic; 
            d_sub = zeros(obj.n_subclasses,size(X,2));
            d = zeros(obj.n_classes,size(X,2));
            for k =1:obj.n_classes
                for j = 1:obj.n_subclasses
                    %Matrix computation
                    d_sub(j,:) = vecnorm(X - obj.mu(:,j,k)).^2; %Distance from each point to one subclass
                end
                d(k,:) =  min(d_sub, [], 1); %Minimun distance for each point (vector)
            end
            [~,c] = min(d, [], 1);
            res = c - obj.correction;
            
            %Compute accuracy
            %If diff is zero, the labels are equal and so correct
            diff = find(Y-res' == 0);
            acc = size(diff,1)/size(Y,1);
            time=toc;
        end
        
    end
end

