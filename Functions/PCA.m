function [W,time] = PCA(X, n_dimension)
    %X: data to transform
    %n_dimension: number of dimensions to keep
    tic;
    X = X-mean(X,2);
    St = X*X';
    [V,d] = eig(St,'vector');
    [~, ind] = sort(d, 'descend');
    V = V(:, ind);
    W = V(:,1:n_dimension);
    time = toc;
end