
w=load('y_filter.mat');
%size of w 64x20x11x11
w=w.x;
%sixe of w now = 11x11x64x20
w = permute(w,[3,4,2,1]);

%w_cb=rand(11,11,10,64);

w5 = w(:,:,:,1:6);
%size of 11x11x10x5
%j=2
%for i=1:10
%   w_cb(:,:,:,i)=w(:,:,:,j);
%   j=j+2;
    
%end

%for i=1:64 
hold on
%temp_w = w(:,:,:,:);
temp_w=reshape(w5,11,11,1,[]);
%temp_w = permute(temp_w,[1,2,4,3]);

% Get the network weights for the second convolutional layer
%w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
temp_w = mat2gray(temp_w);
temp_w = imresize(temp_w,3);

%temp_w=((temp_w)/max(temp_w(:)-min(temp_w(:))))*255;
%w1=temp_w(:,:,1,:);
%w2=w(:,:,3,:);
%subplot_num=size(w,3);
% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.

%for i=1:640
%    temp_w(:,:,1,i)=histeq(temp_w(:,:,1,i),2048);
%end


montage(temp_w,'Size',[6 10],'DisplayRange',[ ])
%title('First convolutional layer weights')

%figure 
%montage(w2)
hold off
%end

%{
w=load('y_filter.mat');
w=w.x;
w = permute(w,[3,4,2,1]);

size(w)

% Get the network weights for the second convolutional layer
%w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w = mat2gray(w);
w = imresize(w,5);

w1=w(:,:,1,:);
%w2=w(:,:,3,:);
%subplot_num=size(w,3);
% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.


figure
montage(w1)
hold on
title('First convolutional layer weights')
figure 
montage(w2)

%}





