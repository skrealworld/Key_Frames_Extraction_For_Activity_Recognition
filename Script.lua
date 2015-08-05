package.path = package.path .. ';Models/?.lua'  --define path for Models

model = require 'Alexnet_eladhoffer'    -- Select the model
model = model:cuda()
print(model)

require 'mattorch'

-- load the luminance 

--local trainx = mattorch.load('trainX.mat')    --  training data samples
--local trainy = mattorch.load('trainY.mat')
--local testx  = mattorch.load('testX.mat')
----local testy  = mattorch.load('testY.mat')


trainset = {}
trainset.data = torch.rand(200,15,64,64):cuda()
temp1 = torch.rand(200)
for i=1,200 do temp1[i] = math.random(1,101) end
trainset.label = torch.rand(200,1)
trainset.label[{{},1}] = temp1
trainset.label = trainset.label:cuda()
--trainset.data = trainx.train:cuda()
--trainset.label = trainy.trainY:cuda()
print(trainset)
testset = {}
testset.data = torch.rand(50,15,64,64):cuda()
temp2 = torch.rand(50)
for i=1,50 do temp1[i] = math.random(1,101) end
testset.label = torch.rand(50,1)
testset.label[{{},1}] = temp2
testset.label = testset.label:cuda()
--testset.data = testx.vid_data:cuda()
--testset.data = testy.Y:cuda()
print(testset)

setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

criterion  = nn.ClassNLLCriterion()
criterion = criterion:cuda()
function trainset:size()
        return self.data:size(1)
end
