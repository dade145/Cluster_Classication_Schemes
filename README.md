# A Comparison Between Classification Schemes

_This repository has been created as part of the "Optimization and Data Analytics" course at Aarhus University._  

Image Classification consists of categorizing and labelling groups of pixels into different classes. To perform this task, many different algorithms have been proposed. The aim of the project was to implement and compare a subclass of them which makes use of clusters: Nearest Class Centroid classifier, Nearest Sub-Class Centroid classifier and Nearest Neighbor classifier.  
The analysis has been conducted using two different standard datasets, MNIST and ORL, studying how the same algorithm performed on different database and how the number of training datapoints and features can impact the success rate and execution time. For better examinate the influence of the number of features, various Principal Component Analysis (PCA) have been applied, reducing the number of dimensions while keeping more information as possible.  
Results show that the classification accuracy increases with the number of centroids, but with a downside in terms of execution time. 
The whole study has been performed in a MATLAB environment due to its native approach to matrixes.

In `Report.pdf`, the conclusions of the analysis can be read.
