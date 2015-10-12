require 'nn'
require 'cunn'

function CNR(model,nIn,nOut)
  model:add(nn.SpatialConvolution(nIn,nOut,3,3,1,1,1,1))
  model:add(nn.SpatialBatchNormalization(nOut,1e-3))
  model:add(nn.ReLU())
end

local function drop(net,value)
  net:add(nn.Dropout(value))
  return net
end

local function pool(net)
  net:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
  return net
end


function ApplyModel()
  local model = nn.Sequential() ;

  --CNR 1-2 layers
  CNR(model,15,48)
  drop(model,0.4)
  CNR(model,48,48)
  pool(model)       --32*32 

  CNR(model,48,96)
  drop(model,0.4)
  CNR(model,96,96)
  pool(model)       --16*16

  CNR(model,96,192)
  drop(model,0.4)
  CNR(model,192,192)
  pool(model)       --8*8   

  CNR(model,192,256)
  drop(model,0.4)
  CNR(model,256,256)
  pool(model)       --4*4

  CNR(model,256,512)
  drop(model,0.4)
  CNR(model,512,512)
  pool(model)       --2*2

  CNR(model,512,1024)
  drop(model,0.4)
  CNR(model,1024,1024)
  pool(model)       --1*1

  model:add(nn.View(1024))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(1024,512))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(512,101))

  return model
end
