package.path = package.path .. ';Models/?.lua'  --define path for Models

model = require 'Alexnet_eladhoffer'    -- Select the model      
print(model)

require 'mattorch'

-- load the luminance 

--local trainx = mattorch.load('trainX.mat')    --  training data samples
--local trainy = mattorch.load('trainY.mat')
--local testx  = mattorch.load('testX.mat')
----local testy  = mattorch.load('testY.mat')

trainset = {}
trainset.data = torch.rand(20000,15,64,64):cuda()
trainset.label = torch.rand(20000,1):cuda()
--trainset.data = trainx.train:cuda()
--trainset.label = trainy.trainY:cuda()
print(trainset)
testset = {}
testset.data = torch.rand(5000,15,64,64):cuda()
testset.label = torch.rand(5000,1):cuda()
--testset.data = testx.vid_data:cuda()
--testset.data = testy.Y:cuda()
print(testset)
criterion  = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(model,criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 2

trainer:train(trainset)

predicted = model:forward(testset.data[50])
print(predicted)
