% http://daniel-e.github.io/2017-10-20-loading-mnist-handwritten-digits-with-octave-or-matlab/
%d.trainX is a (60000,784) matrix which contains the pixel data for training
%d.trainY is a (1,60000) matrix which contains the labels for the training data
%d.testX is a (10000,784) matrix which contains the pixel data for testing
%d.testY is a (1,10000) matrix which contains the labels for the test set

d = load('mnist.mat');

X = d.trainX;
Y = d.trainY;
i = reshape(X(1,:), 28, 28)';
label = Y(1)
image(i);