% convert cifi-100 32x32 pixels into 64x64
load cifi/c_100_train;
dim = double(n_shape);
dim_scale = dim ./ 32.;
tr_labels = fine_labels;
% convert to greyscale
red = data(:,1:1024);green = data(:,1025:2048);blue = data(:,2049:3072);
data = red*0.2989 + green*0.5870 + blue*0.1140;
% show example: ex = reshape(data(1,:),[32 32 1]);imshow(ex');
% upscale to 64x64
data = reshape(data,[50000 32 32 1]);
% imshow(reshape(data(1,:,:),[32 32 1])');
data_dim = [];
parfor i=1:50000
    x = reshape(data(i,:,:,:),[32 32 1]);
    image = imresize(x,dim_scale,"bicubic");
    image = reshape(image,[1 dim dim]);
    data_dim = [data_dim;image];
end
data_dim = single(reshape(data_dim,[50000 (dim*dim)]));
% normalize
d_train = data_dim./255;

%%% now create test and validation sets
load c_100_test;
% convert to greyscale
red = data(:,1:1024);green = data(:,1025:2048);blue = data(:,2049:3072);
data = red*0.2989 + green*0.5870 + blue*0.1140;
data = reshape(data,[10000 32 32 1]);
data_dim = [];
parfor i=1:10000
    x = reshape(data(i,:,:,:),[32 32 1]);
    image = imresize(x,dim_scale,"bicubic");
    image = reshape(image,[1 dim dim]);
    data_dim = [data_dim;image];
end
% reshape and normalize
data_dim = single((reshape(data_dim,[10000 (dim*dim)])))./255;
% d_test = data_dim(1:5000,:);
% d_val = data_dim(5001:10000,:);
d_val=data_dim;
save cifi_data d_train d_val; %d64_test


% in case I want to apply the whitening alg.:

% imgsize = 40;
% % apply whitening -- code from Burghausen 2001
% f = -imgsize/2:imgsize/2-1;
% % create filter:
% [fx fy] = meshgrid(f);
% [~, rho] = cart2pol(fx,fy);  % polar coordinates (rho is the filter)
% % multiply by a circular, Gaussian filter:
% filtf = rho.*exp(-0.5*(rho/(0.7*imgsize/2)).^2);
% % apply filter on image
% filtf = reshape(filtf,[1 1600]);
% img_w = shapedata(1,:).*filtf;
% % contrast normalization
% neigh = 20; % divide the output of a neuron by some measure of the total activity in the neighborhood
% [x y] = meshgrid(-neigh/2:neigh/2-1);
% G = exp(-0.5*((x.^2+y.^2)/(neigh/2)^2));
% G = G/sum(G(:));


