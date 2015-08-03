--require 'itorch'


------ import required modules------
require 'nn'
require 'image'
require 'mattorch'
require 'cudnn'
require 'cunn'
require 'cutorch'
---------------------------dataset setup--------------------
file_train = mattorch.load('~/VideoClassification/dataSet/Subdatabase/Xdata_train20.mat');
file_train_labels =mattorch.load('~/VideoClassification/dataSet/Subdatabase/Ydata_train20.mat');
file_test= mattorch.load('~/VideoClassification/dataSet/Subdatabase/Xdata_testing20.mat');
file_test_labels= mattorch.load('~/VideoClassification/dataSet/Subdatabase/Ydata_testing20.mat');
trainset = {};
trainset.data = file_train.Xdata_train:double(); 
trainset.label = file_train_labels.Ydata_train[{{},1}];
print(trainset)
print(trainset.data:dim())
testset = {};
testset.data = file_test.Xdata_testing20:double();
testset.label = file_test_labels.Ydata_testing20[{{},1}];
print(testset)                  --torch.load('/Users/SK_Mac/cifar10-test.t7')
---testset = torch.load('/Users/SK_Mac/cifar10-test.t7')
classes = {'Archery','Basketball'  ,  'BasketballDunk' ,   'Biking',    'Bowling' ,  'BoxingPunchingBag' ,'BoxingSpeedBag' ,   'BreastStroke' , 'CricketBowling' ,   'CricketShot' ,  'Diving'   , 'FrontCrawl'  ,  'HighJump'  ,'HorseRace' ,'HorseRiding' ,  'LongJump'  ,'SkyDiving' ,'SoccerPenalty' ,'Surfing'   ,'TableTennisShot'}
---print(trainset)
---print(#trainset.data)
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
--trainset.data = trainset.data:cuda()
--trainset.data = trainset.data:cuda() -- convert the data from a ByteTensor to a DoubleTensor.
trainset.data = trainset.data:cuda();
trainset.label = trainset.label:cuda();
--classes = classes:cuda();

function trainset:size() 
    return self.data:size(1) 
end
--print(trainset:size()) -- just to test
--print(trainset[33]) -- load sample number 33.
--image.display(trainset[33][1])
--redChannel = trainset.data[{ {}, {1}, {}, {}  }]
--print(#redChannel)


------------------Data Normalization------------------------------------------------------


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




-----------------------------------------Alex net------------------------------------
--require 'cudnn'
--require 'cunn'
SpatialConvolution = cudnn.SpatialConvolution--lib[1]
SpatialMaxPooling = cudnn.SpatialMaxPooling--lib[2]

-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
features = nn.Sequential()
features:add(SpatialConvolution(10,64,7,7,2,2,2,2))       -- 224 -> 55
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(64,nil,nil,false))
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(192,nil,nil,false))
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(384,nil,nil,false))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256,nil,nil,false))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256,nil,nil,false))

classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096,nil,nil,false))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.BatchNormalization(4096,nil,nil,false))
classifier:add(nn.Linear(4096, 20)) -- Changed 1000 class into 20 classes
classifier:add(nn.LogSoftMax())

model = nn.Sequential()

function fillBias(m)
for i=1, #m.modules do
    if m:get(i).bias then
        m:get(i).bias:fill(0.1)
    end
end
end

--fillBias(features)
--fillBias(classifier)
model:add(features):add(classifier)
model=model:cuda()


-----------------------------------------Alex net end--------------------------------


-----------------Simple two layred CNN-----------------------------------------------

--[[
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

--]]
-------------------------------------------------------------------------------------

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.


trainer:train(trainset)
--print(classes[testset.label[100]])
--image.display(testset.data[100])

testset.data = testset.data:cuda()   -- convert from Byte tensor to Double tensor
--testset.data = testset.data:cuda()
for i=1,10 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

--just for fun 
--horse = testset.data[100]
--print(horse:mean(), horse:std())

--print(classes[testset.label[100]])
--image.display(testset.data[100])
predicted = model:forward(testset.data[100])
-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
print(predicted:exp())
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

correct = 0
for i=1,2564 do
    local groundtruth = testset.label[i]
    local prediction = model:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print('correct =', 100*correct/2564 .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,2654 do
    local groundtruth = testset.label[i]
    local prediction = model:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

--for i=1,#classes do
--    print(classes[i], 100*class_performance[i]/1000 .. ' %')
--end
