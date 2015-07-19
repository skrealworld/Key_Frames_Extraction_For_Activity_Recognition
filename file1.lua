--require 'itorch'
require 'nn'
require 'image'
require 'mattorch'
file_train = mattorch.load('Xdata_train20.mat');
file_train_labels =mattorch.load('Ydata_train20.mat');
file_test= mattorch.load('Xdata_testing20.mat');
file_test_labels= mattorch.load('Ydata_testing20.mat');
trainset = {};
trainset.data =   file_train.Xdata_train; 
trainset.label =   file_test_labels.Ydata_train;
testset = {};
testset.data = file_test.Xdata_testing20;
testset.label = file_test_labels.Ydata_testing20;                  --torch.load('/Users/SK_Mac/cifar10-test.t7')
---testset = torch.load('/Users/SK_Mac/cifar10-test.t7')
classes = {'Archery','Basketball'  ,  'BasketballDunk' ,   'Biking',    'Bowling' ,  'BoxingPunchingBag' ,'BoxingSpeedBag' ,   'BreastStroke' , 'CricketBowling' ,   'CricketShot' ,  'Diving'   , 'FrontCrawl'  ,  'HighJump'  ,'HorseRace' ,'HorseRiding' ,  'LongJump'  ,'SkyDiving' ,'SoccerPenalty' ,'Surfing'   ,'TableTennisShot'}
---print(trainset)
---print(#trainset.data)
--setmetatable(trainset, 
--    {__index = function(t, i) 
--                    return {t.data[i], t.label[i]} 
--                end}
--);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

--function trainset:size() 
--    return self.data:size(1) 
--end
--print(trainset:size()) -- just to test
--print(trainset[33]) -- load sample number 33.
--image.display(trainset[33][1])
--redChannel = trainset.data[{ {}, {1}, {}, {}  }]
--print(#redChannel)
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,10 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--design neural network

net = nn.Sequential()
net:add(nn.SpatialConvolution(10, 6, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.


trainer:train(trainset)
--print(classes[testset.label[100]])
--image.display(testset.data[100])

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,10 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--just for fun 
--horse = testset.data[100]
--print(horse:mean(), horse:std())

--print(classes[testset.label[100]])
--image.display(testset.data[100])
predicted = net:forward(testset.data[100])
-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

correct = 0
for i=1,2564 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print('correct =', 100*correct/2564 .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,2654 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

--for i=1,#classes do
--    print(classes[i], 100*class_performance[i]/1000 .. ' %')
--end


