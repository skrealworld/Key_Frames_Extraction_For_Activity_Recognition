require 'nn';


net = nn.Sequential()
net:add(SpatialConvolution(15,64,11,11,4,4,2,2)) -- 224 --> 55
net:add(nn.ReLU(true))

net:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
net:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
net:add(nn.ReLU(true))
net:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
net:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
net:add(nn.ReLU(true))
net:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
net:add(nn.ReLU(true))
net:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
net:add(nn.ReLU(true))
net:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, 1000))
classifier:add(nn.LogSoftMax())

model = nn.Sequential()

function fillBias(m)
for i=1, #m.modules do
    if m:get(i).bias then
        m:get(i).bias:fill(0.1)
    end
end
end

fillBias(net)
fillBias(classifier)
model:add(net):add(classifier)


