-require 'itorch'


------ import required modules------
require 'nn'
require 'image'
require 'mattorch'
require 'cudnn'
require 'cunn'

---------------------------dataset setup--------------------
file_train = mattorch.load('Xdata_train20.mat');
file_train_labels =mattorch.load('Ydata_train20.mat');
file_test= mattorch.load('Xdata_testing20.mat');
file_test_labels= mattorch.load('Ydata_testing20.mat');
trainset = {};
trainset.data = file_train.Xdata_train;
trainset.label = file_train_labels.Ydata_train[{{},1}];
print(trainset)
testset = {};
testset.data = file_test.Xdata_testing20;
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
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end
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

SpatialConvolution = cudnn.SpatialConvolution
SpatialMaxPooling = cudnn.SpatialMaxPooling


features = nn.Sequential()
features:add(SpatialConvolution(10,64,7,7,2,2,2,2))       
features:add(SpatialMaxPooling(3,3,2,2))                   
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(64,nil,nil,false))
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       
features:add(SpatialMaxPooling(3,3,2,2))                   
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(192,nil,nil,false))
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(384,nil,nil,false))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      
features:add(cudnn.ReLU(true))
features:add(nn.SpatialBatchNormalization(256,nil,nil,false))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      
features:add(SpatialMaxPooling(3,3,2,2))                   
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
criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(model, criterion)
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




