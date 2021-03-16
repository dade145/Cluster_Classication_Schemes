close all;

%% Add path
disp('Adding path...');
addpath(genpath('MNIST/'));
addpath(genpath('ORL/'));
addpath(genpath('plots/'));
addpath(genpath('Functions/'));
addpath(genpath('Classifiers/'));
 
%% Load Data
disp('Loading data...');
[MNIST_train, MNIST_test,MNIST_train_labels,MNIST_test_labels] = load_MNIST();
load('orl_data.mat');
ORL_data = data; clear data;
load('orl_lbls.mat');
ORL_labels = lbls; clear lbls;

% Shuffle ORL dataset
whole      = [ORL_labels, ORL_data'];            %Concanete data and labels
random_all = whole(randperm(size(whole, 1)), :); %Shuffle
ORL_labels = random_all(:,1);                    %Re-split data from labels
ORL_data   = random_all(:, 2:size(whole,2))'; 
% Split ORL dataset in train and test
ORL_datasize     = size(ORL_data,2);
train_number     = 0.7*ORL_datasize;
ORL_train        = ORL_data(:,1:train_number);
ORL_train_labels = ORL_labels(1:train_number);
ORL_test         = ORL_data(:,train_number+1:ORL_datasize);
ORL_test_labels  = ORL_labels(train_number+1:ORL_datasize);

clear ORL_data; clear ORL_labels; clear whole; clear random_all;
%% PCA
disp('Appling PCA...');
num_PCA         = 4;
MNIST_train_PCA = cell(4,1);
MNIST_test_PCA  = cell(4,1);
ORL_train_PCA   = cell(4,1);
ORL_test_PCA    = cell(4,1);
m=2; o=2;
%Apply PCA 4 time, with:
% 2 dimensions, 1% of the original dimesions, 2% and 3%
%Return the transformed data
for i=1:num_PCA
    W                  = PCA(MNIST_train, m);
    MNIST_train_PCA{i} = W'*MNIST_train;
    MNIST_test_PCA{i}  = W'*MNIST_test;
    W                  = PCA(ORL_train, o);
    ORL_train_PCA{i}   = W'*ORL_train;
    ORL_test_PCA{i}    = W'*ORL_test;
    m                  = cast(m-1+size(MNIST_train,1)*0.01*i, 'int32');
    o                  = cast(o-1+size(ORL_train,1)*0.01*i, 'int32');
end

clear W; clear m; clear o;

%%
label = {}; %Used for plotting
%Accuracy and execution time are saved in two matrix for each dataset (MNIST and ORL)
%Matrix are organized as follow:
%Each row is a different algorithm, each columns is a different set (NO PCA, PCA with 2 dimensions, PCA 1%...)
%Time matrxi saves both fit time and score time
%% NCC
disp('NCC MNIST...');
label{1} = 'NCC';
MNIST_ncc = NCC(10, 1);
[MNIST_ncc, MNIST_time(1,1,1)] = MNIST_ncc.fit(MNIST_train, MNIST_train_labels);
[MNIST_acc(1,1), MNIST_time(1,1,2)] = MNIST_ncc.score(MNIST_test, MNIST_test_labels);
for i=1:num_PCA
    [MNIST_ncc, MNIST_time(1,1+i,1)] = MNIST_ncc.fit(MNIST_train_PCA{i}, MNIST_train_labels);
    [MNIST_acc(1,1+i), MNIST_time(1,1+i,2)] = MNIST_ncc.score(MNIST_test_PCA{i}, MNIST_test_labels);
end

disp('NCC ORL...');
ORL_ncc = NCC(40,0);
[ORL_ncc, ORL_time(1,1,1) ] = ORL_ncc.fit(ORL_train, ORL_train_labels);
[ORL_acc(1,1), ORL_time(1,1,2)] = ORL_ncc.score(ORL_test, ORL_test_labels);
for i=1:num_PCA
    [ORL_ncc, ORL_time(1,1+i,1)] = ORL_ncc.fit(ORL_train_PCA{i}, ORL_train_labels);
    [ORL_acc(1,1+i), ORL_time(1,1+i,2)] = ORL_ncc.score(ORL_test_PCA{i}, ORL_test_labels);
end

clear MNIST_ncc; clear ORL_ncc;
%% NSC
label{2} = 'NSC2';
label{3} = 'NSC3';
label{4} = 'NSC5';

subkluster = [2 3 5];
cc_k=2;

disp('NSC MNIST...');
for k=subkluster
    MNIST_nsc = NSC(10, k, 1);
    [MNIST_nsc, MNIST_time(cc_k,1,1)] = MNIST_nsc.fit(MNIST_train, MNIST_train_labels);
    [MNIST_acc(cc_k,1), MNIST_time(cc_k,1,2)] = MNIST_nsc.score(MNIST_test, MNIST_test_labels);
   
    
    for i=1:num_PCA
        [MNIST_nsc, MNIST_time(cc_k,1+i,1)] = MNIST_nsc.fit(MNIST_train_PCA{i}, MNIST_train_labels);
        [MNIST_acc(cc_k,1+i), MNIST_time(cc_k,1+i,2)] = MNIST_nsc.score(MNIST_test_PCA{i}, MNIST_test_labels);

    end
    cc_k = cc_k+1;
end

disp('NSC ORL...');
cc_k = 2;
for k=subkluster
    ORL_nsc = NSC(40, k, 0);
    [ORL_nsc, ORL_time(cc_k,1,1) ] = ORL_nsc.fit(ORL_train, ORL_train_labels);
    [ORL_acc(cc_k,1), ORL_time(cc_k,1,2)] = ORL_nsc.score(ORL_test, ORL_test_labels);
    for i=1:num_PCA
        [ORL_nsc, ORL_time(cc_k,1+i,1)]                = ORL_nsc.fit(ORL_train_PCA{i}, ORL_train_labels);
        [ORL_acc(cc_k,1+i), ORL_time(cc_k,1+i,2)] = ORL_nsc.score(ORL_test_PCA{i}, ORL_test_labels);
    end
    cc_k=cc_k+1;
end
    
clear MNIST_nsc; clear ORL_nsc;
%% NN
label{5} = 'NN';

disp('NN MNIST...');
MNIST_nnc = NNC();
[MNIST_nnc, MNIST_time(5,1,1)] = MNIST_nnc.fit(MNIST_train, MNIST_train_labels);
[MNIST_acc(5,1), MNIST_time(5,1,2)] = MNIST_nnc.score(MNIST_test, MNIST_test_labels);

for i=1:num_PCA
    [MNIST_nnc, MNIST_time(5,1+i,1)] = MNIST_nnc.fit(MNIST_train_PCA{i}, MNIST_train_labels);
    [MNIST_acc(5,1+i), MNIST_time(5,1+i,2)] = MNIST_nnc.score(MNIST_test_PCA{i}, MNIST_test_labels);
end

disp('NN ORL...');
ORL_nnc = NNC();
[ORL_nnc,  ORL_time(5,1,1)] = ORL_nnc.fit(ORL_train, ORL_train_labels);
[ORL_acc(5,1),  ORL_time(5,1,2)] = ORL_nnc.score(ORL_test, ORL_test_labels);

for i=1:num_PCA
    [ORL_nnc, ORL_time(5,1+i,1)] = ORL_nnc.fit(ORL_train_PCA{i}, ORL_train_labels);
    [ORL_acc(5,1+i), ORL_time(5,1+i,2)] = ORL_nnc.score(ORL_test_PCA{i}, ORL_test_labels);
end

clear MNIST_nnc; clear ORL_nnc;
%% Plot
close all;
disp('Plotting...');
FontSize = 20;
FontSizeValues = 12;

fig_MNIST = figure('name','MNIST Acc','units','normalized','outerposition',[0 0 1 1]);
b = bar(round(MNIST_acc*100,2));
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels3 = string(b(3).YData);
text(xtips3,ytips3,labels3,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;
labels4 = string(b(4).YData);
text(xtips4,ytips4,labels4,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips5 = b(5).XEndPoints;
ytips5 = b(5).YEndPoints;
labels5 = string(b(5).YData);
text(xtips5,ytips5,labels5,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
ylim([0 100]);
set(gca,'xticklabel',label);
set(gca,'FontSize',FontSize);
ylabel('Accuracy [%]','FontSize',FontSize);
legend({'NO PCA', 'PCA 2', 'PCA 1%', 'PCA 2%', 'PCA 3%'},...
        'Location','northwest','Orientation','horizontal')
grid on

fig_MNIST_time = plotBarStackGroups(round(MNIST_time*1000,0),label);

fig_ORL = figure('name','ORL Acc','units','normalized','outerposition',[0 0 1 1]);
b=bar(round(ORL_acc*100,2));
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips2 = b(2).XEndPoints;
ytips2 = b(2).YEndPoints;
labels2 = string(b(2).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips3 = b(3).XEndPoints;
ytips3 = b(3).YEndPoints;
labels3 = string(b(3).YData);
text(xtips3,ytips3,labels3,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips4 = b(4).XEndPoints;
ytips4 = b(4).YEndPoints;
labels4 = string(b(4).YData);
text(xtips4,ytips4,labels4,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
xtips5 = b(5).XEndPoints;
ytips5 = b(5).YEndPoints;
labels5 = string(b(5).YData);
text(xtips5,ytips5,labels5,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom','FontSize', FontSizeValues)
ylim([0 100]);
set(gca,'xticklabel',label);
set(gca,'FontSize',FontSize);
ylabel('Accuracy [%]','FontSize',FontSize);
legend({'NO PCA', 'PCA 2', 'PCA 1%', 'PCA 2%', 'PCA 3%'},...
        'Location','southwest','Orientation','horizontal')
grid on

fig_ORL_time = plotBarStackGroups(round(ORL_time*1000,0),label);

fig_PCA_M = figure('name', 'PCA MNIST','units','normalized','outerposition',[0 0 1 1]);
x = MNIST_train_PCA{1}';
scatter(x(:,1), x(:,2), 36, MNIST_train_labels);

fig_PCA_O = figure('name', 'PCA ORL','units','normalized','outerposition',[0 0 1 1]);
x = ORL_train_PCA{1}';
scatter(x(:,1), x(:,2), 120, ORL_train_labels, 'filled');

time_m = zeros(size(MNIST_time,1),size(MNIST_time,2));
time_o = zeros(size(MNIST_time,1),size(MNIST_time,2));
for i=1:size(MNIST_time, 1)
    for j=1:size(MNIST_time, 2)
        time_m(i,j) = MNIST_time(i,j,1)+MNIST_time(i,j,2);
        time_o(i,j) = ORL_time(i,j,1)+ORL_time(i,j,2);
    end
end

fig_all = figure('units','normalized','outerposition',[0 0 1 1]);
x_MNIST = reshape(MNIST_acc(:,1)*100.',1,[]);
x_MNIST = [x_MNIST reshape(MNIST_acc(:,5)*100.',1,[])];
y_MNIST = reshape(time_m(:,1).',1,[]);
y_MNIST = [y_MNIST reshape(time_m(:,5).',1,[])];
x_ORL   = reshape(ORL_acc(:,1)*100.',1,[]);
x_ORL   = [x_ORL reshape(ORL_acc(:,5)*100.',1,[])];
time_o = sum(ORL_time,3);
y_ORL   = reshape(time_o(:,1).',1,[]);
y_ORL   = [y_ORL reshape(time_o(:,5).',1,[])];
h=semilogy(x_MNIST,y_MNIST,'x',x_ORL,y_ORL,'o');
lgd = legend('MNIST','ORL','Location','northwest');
lgd.FontSize = FontSize;
xlim([80,100]);
l = {'NCC', 'NSC2','NSC3','NSC5','NN', 'NCC PCA 3%', 'NSC2 PCA 3%','NSC3 PCA 3%','NSC5 PCA 3%','NN PCA 3%'};
labelpoints(x_MNIST,y_MNIST,l,'N','FontSize',FontSizeValues+4);
labelpoints(x_ORL,y_ORL,l,'N','FontSize',FontSizeValues+4);
set(h(1),'MarkerSize',12,'Linewidth',3);
set(h(2),'MarkerSize',12,'Linewidth',3);
set(gca,'FontSize',FontSize);
ylabel('Execution Time [s]','FontSize',FontSize)
xlabel('Accuracy [%]','Fontsize',FontSize)
grid on

clear x; clear x_MNIST; clear x_ORL; clear y_MNIST; clear y_ORL;
clear xtips1; clear xtips2; clear xtips3; clear xtips4; clear xtips5;
clear ytips1; clear ytips2; clear ytips3; clear ytips4; clear ytips5;
%% Save Figures
disp('Saving plot...');
print(fig_MNIST, 'plots/mnist_acc','-depsc');
print(fig_MNIST_time, 'plots/mnist_time','-depsc');
print(fig_ORL, 'plots/orl_acc','-depsc');

print(fig_ORL_time, 'plots/orl_time','-depsc');

print(fig_PCA_M, 'plots/mnist_pca','-depsc');

print(fig_PCA_O, 'plots/orl_pca','-depsc');

print(fig_all, 'plots/comparision','-depsc');

clear fig_all; clear fig_MNIST; clear fig_ORL; clear fig_MNIST_time;
clear fig_ORL_time; clear fig_PCA_M; clear fig_PCA_O;

disp('DONE!');