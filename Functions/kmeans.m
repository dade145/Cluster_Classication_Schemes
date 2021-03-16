function M = kmeans(data,K,max,eta)
    %data: data to cluster
    %K: number of cluster to find
    %max: maximum number of iteration
    %eta: miminum distance between two update of the centroids matrix
    M = data(:,randi(size(data,2),1,K)); %mean matrix
    last_M = ones(size(M)); 
    if nargin<3
        max= 150; %default value of max iteration
    end
    if nargin<4
        eta = 1e-5; %default stop condition
    end
    
    for z=1:max
       if norm(M - last_M) < eta
           break;
       end
       last_M = M;
       dist = zeros(K, size(data,2));
       for k=1:K
           dist(k,:) = vecnorm(data - M(:,k)).^2;
       end
       [~, labels] = min(dist, [], 1);
       %Update mean matrix
       for k=1:K
           M(:,k) = mean(data(:, labels==k),2);
       end 
    end
end

