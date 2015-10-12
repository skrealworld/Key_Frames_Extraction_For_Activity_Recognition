----------------------------------------------------------------------
-- This script shows how to train predefined CNN models with batches.  
-- dataset has 15 channel and resolution is 64X64
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 101-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Sourabh Kulhare
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'package'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "vggStyle_01")   type of model tor train: convnet | mlp | linear | vggStyle_01
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 1)        learning rate, for SGD only
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0.9)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]

-- fix seed
torch.manualSeed(1)
print(opt.threads)

-- Give the batch size number, default batchsize is 10
opt.batchSize = 10
-- threads
torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

--print(opt.batchSize)
-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {}
for i=1,101 do classes[i] = ""..i.."" end

--classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {64,64}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then

	package.path = package.path .. ';Models/?.lua'  --define path for Models
  --print(package.path)
	model = require 'Alexnet_eladhoffer'    -- Select the model
	model = model:cuda()      
	print('loaded the model')

      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      --model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      --model:add(nn.Tanh())
      --model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      --model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      --model:add(nn.Tanh())
      --model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      --model:add(nn.Reshape(64*2*2))
      --model:add(nn.Linear(64*2*2, 200))
      --model:add(nn.Tanh())
      --model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'vggStyle_01' then
      require './Models/vggStyle_01'   --store in a seperate models folder
      model = ApplyModel()
      model = model:cuda()
	print('loaded the model')
   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()
parameters = parameters:cuda()
gradParameters = gradParameters:cuda()
-- verbose
--print('<mnist> using model:')
--print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
--model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

----------------------------------------------------------------------

--train data: 81231 X 15 X 64 X 64
--trainData = torch.load('./data/trainX_sh.t7')   
--trainLabel = torch.load('./data/trainY_sh.t7')
--print('loaded training data');
--test data: 41830 X 15 X 64 X 64
--testData  = torch.load('./data/testX_sh.t7')
--testLabel  = torch.load('./data/testY_sh.t7')
--print('loaded testing data');
--data-------------------------------------------- 

trainset = {}
trainset.data = trainData
trainset.label = trainLabel

testset = {}
testset.data = testData
testset.label = testLabel

--sizes of train and test-------------------------
function trainset:size()
        return self.data:size(1)
end

function testset:size()
        return self.data:size(1)
end

--return pairs------------------------------------

setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


-- preprocess-------------------------------------
-- train data


--kernel = image.gaussian1D(7)
--normalize = nn.SpatialContrastiveNormalization(1,kernel):float()
--[[
for i = 1,trainset:size() do  --#examples
    local yTrain = torch.rand(15,64,64)
    local frame = trainset.data[i]
   for j=1,15 do  -- #channels
     -- y channeled frames of a train samp
     -- normalize y locally:
     yTrain[{{j},{},{}}] = normalize(frame[{{j},{},{}}]:float())

   end

   trainset.data[i] = yTrain

end
torch.save('norm_trainset.t7',trainset)

print('Performed normalization with training data')
-- test data
for i = 1,testset:size() do  --#examples
    local yTest = torch.rand(15,64,64)
    local frame = testset.data[i]
   for j=1,15 do  -- #channels

     -- y channeled frames of a train sample
     -- normalize y locally:
     yTest[{{j},{},{}}] = normalize(frame[{{j},{},{}}]:float())
   end

   testset.data[i] = yTest

end

torch.save('norm_testset.t7',testset)
--]]
--print('Performed normalization with training data')
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   local shuffle1 = torch.randperm(dataset:size())
   
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size()-1,opt.batchSize do
      -- create mini batch
      local inputs = torch.FloatTensor(opt.batchSize,15,geometry[1],geometry[2])
	    inputs = inputs:cuda()
      local targets = torch.FloatTensor(opt.batchSize)
            targets = targets:cuda()
            
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         --local sample = dataset[shuffle1[i]]
         --Print(#sample[1])
         local input = dataset.data[shuffle1[i]]:clone()
         
         --local _,target = sample[2]:clone():max(1)
         
         local target = dataset.label[shuffle1[i]]:clone()
         
         target = target:squeeze()
         
         inputs[k] = input
         
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()
	--confusion:zeros()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         --reset gradients
         gradParameters:zero()
	 --inputs = inputs()
	 --targets = targets()
         --evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         --print(outputs)
	 
         local f = criterion:forward(outputs, targets)

         --estimate df/dW
         local df_do = criterion:backward(outputs, targets)
	 df_do = df_do:cuda()
         model:backward(inputs, df_do)

         --penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
	    confusion:add(outputs[i], targets[i])
	
         end
	--print('fbfbbb')
	--io.flush()
	--io.read()

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
   
   --confusion:updateValids()
   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.t7')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.FloatTensor(opt.batchSize,15,geometry[1],geometry[2])
            inputs = inputs:cuda()
      --print(inputs)
      local targets = torch.FloatTensor(opt.batchSize)
            targets = targets:cuda()
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         --local sample = dataset[i]
         local input = dataset.data[i]:clone()
         local target = dataset.label[i]:clone()
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)
      --print(preds);
      --print(targets);

      -- confusion:
      -- to handle the end condition for confusion matrrix of testdata.
      temp = opt.batchSize
      if t>41824 then temp = 6 end
      for i = 1,temp do
	 
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
for i=1,300 do
   -- train/test
   trainset = torch.load('norm_trainset.t7')
   train(trainset)
   trainset = nil
   testset = torch.load('norm_testset.t7')
   test(testset)
   testset = nil
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
