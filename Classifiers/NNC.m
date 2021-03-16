classdef NNC
    
    properties
        data_train;
        labels_train;
    end
    
    methods
        function obj = NSC()
        end
        
        function [obj,time] = fit(obj,X, Y)
            %X: training samples
            %Y: training labels
            tic;
            obj.data_train = X;
            obj.labels_train = Y;
            time=toc;
        end
        
        function [acc,time] = score(obj, X, Y)
            %X: test samples
            %Y: test labels
            tic;
            res = zeros(size(X,2),1);
            for i=1:size(X,2)
                d = vecnorm(obj.data_train - X(:,i)).^2;
                [~,idx] = min(d);
                res(i) = obj.labels_train(idx);
            end
            
%             acc = 0;
%             for i=1:size(Y,1)
%                 if Y(i)==res(i)
%                     acc = acc+1;
%                 end
%             end
%             acc = acc/size(Y,1);
            
            %Compute accuracy
            %If diff is zero, the labels are equal and so correct
            diff = find(Y-res == 0);
            acc = size(diff,1)/size(Y,1);
            time=toc;
        end
        
    end
end