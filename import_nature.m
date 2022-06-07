% 1. import data (natural scenes) from local directory
% dataset: CIFAR-100
% source: https://www.cs.toronto.edu/~kriz/cifar.html
% reference work: "Learning Multiple Layers of Features from Tiny Images,Alex Krizhevsky, 2009"

% 2. create "mini-batches"
% equivalent to the concatenation of the methods "converter.m" and "make_batches.m" from the original code

% converting images of CIFI-100 dataset to desired shape defined by n_shape
% call following line if cifi_64bit.mat unexistent


%cifi_data;

load cifi_data d_train d_val;
% load cifi_64bit.mat d64_train d64_val; 
% d_train = d64_train; d_val = d64val;

%%% for computation with subset
% randaccess = randperm(30000);
% d64_train_sub = zeros(30000,4096);
% for i=1:length(randaccess)
%     x = d_train(randaccess(i),:);
%     d64_train_sub(i,:) = x;
% end
% d_train = d64_train_sub;


% create batches for training set
n_totnum=size(d_train,1);
fprintf(1, 'Size of the natual scene training,  dataset= %5d \n', n_totnum);
rand('state',0); %so we know the permutation of the training data / ≈rng object
n_randomorder=randperm(n_totnum);
n_numbatches=n_totnum/n_batchsize;
n_numdims  =  size(d_train,2);
% training
n_batchdata = zeros(n_batchsize, n_numdims, n_numbatches);
% validation

for b=1:n_numbatches
  n_batchdata(:,:,b) = d_train(n_randomorder(1+(b-1)*n_batchsize:b*n_batchsize),:);
end
% following hinton: create representative subset of training data
% if you plot the histogram, you'll see that its close to uniform distr.:
n_train = double(d_train(1:2500,:));


% save information for rapid testing
save import_n n_batchdata n_train




clear d_train;





