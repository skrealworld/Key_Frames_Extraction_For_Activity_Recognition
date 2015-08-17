----------------------------------------------------------------------
--
---- This script shows how to train different models on the MNIST 
--
---- dataset, using multiple optimization techniques (SGD, LBFGS)
--
----
--
---- This script demonstrates a classical example of training 
--
---- well-known models (convnet, MLP, logistic regression)
--
---- on a 10-class classification problem. 
--
----
--
---- It illustrates several points:
--
---- 1/ description of the model
--
---- 2/ choice of a loss function (criterion) to minimize
--
---- 3/ creation of a dataset as a simple Lua table
--
---- 4/ description of training and test procedures
--
----
--
---- Clement Farabet
--
------------------------------------------------------------------------
--
--
--
package.path = package.path..';Models/?.lua' --defines the paths for Models
require 'mattorch'
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
--require 'dataset-mnist'
require 'pl'
require 'paths'

------------------------------------------------------------------------
---- parse command-line options
local opt = lapp[[
	   -s,--save          (default "logs")      subdirectory to save logs
	   -n,--network       (default "")          reload pretrained network
	   -m,--model         (default "AlexNet")   type of model tor train: convnet | mlp | linear								
            -f,--full                                use the full dataset
            -p,--plot                                plot while training
            -o,--optimization  (default "SGD")       optimization: SGD | LBFGS
            -r,--learningRate  (default 0.05)        learning rate, for S
            -b,--batchSize     (default 10)          batch size
            -m,--momentum      (default 0)           momentum, for SGD only
            -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
            --coefL1           (default 0)           L1 penalty on the weights
            --coefL2           (default 0)           L2 penalty on the weights
            -t,--threads       (default 4)           number of threads
                  ]]
-- fix seed
-- torch.manualSeed(1)
torch.manualSeed(1)
-- threads
-- torch.setnumthreads(opt.threads)
--print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
torch.setdefaulttensortype('torch.FloatTensor')
end
-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end
----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem                            --
classes = {'1','2','3','4','5','6','7','8','9','10'}
-- geometry: width and height of input images
geometry = {64,64}
--opt.model='AlexNex'
if opt.network == '' then
-- define model to train
	model = nn.Sequential()
	 if opt.model == 'convnet' then    
------------------------------------------------------------
-- convolutional network 
------------------------------------------------------------
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
		 model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
                 model:add(nn.Tanh())
                 model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
                 model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
                 model:add(nn.Tanh())
                 model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--stage 3 : standard 2-layer MLP:
                 model:add(nn.Reshape(64*2*2))
                 model:add(nn.Linear(64*2*2, 200))
                 model:add(nn.Tanh())
                 model:add(nn.Linear(200, #classes))
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
	  elseif opt.model == 'AlexNet' then
	        print ('accessing Alexnet Sandeep')  
		model:add(require 'Alexnet_eladhoffer')	
------------------------------------------------------------
          else
                print('Unknown model type')
                cmd:text()
                error()
                                                                                                                                                                         end
------------------------------------------------------------
--   elseif opt.model == 'AlexNet' then
--	model=require 'Alexnet_eladhoffer'
------------------------------------------------------------
    else	
     print('<trainer> reloading previously trained network')
     model = torch.load(opt.network)
    end
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)
--[[
print('now loading Alexnet_eladhoffer')
modelAlex = require 'Alexnet_eladhoffer'
print ('AlexNet Model')
print (modelAlex)
]]--
----------------------------------------------------------------------
-- loss function: negative log-likelihood
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
----------------------------------------------------------------------
-- get/create dataset
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 500
   nbTestingPatches = 100
   print('<warning> only using 500 samples to train quickly (use flag -full to use 60000 samples)')
end
-- create training set and normalize
--trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainDataX=torch.load('./Data/trainX.t7')
trainDataY=torch.load('./Data/trainY.t7')
testDataX=torch.load('./Data/testX.t7')
testDataY=torch.load('./Data/testY.t7')
datasetX=trainDataX.train
datasetY=trainDataY.trainY
print('trinX')
print(trainDataX)
print(trainDataY)
print('testData')
print(testDataX)
print(testDataY)
trainDataX=nil
trainDataY=nil

--trainData:normalizeGlobal(mean, std)
----create test set and normalize
--testData = mnist.loadTestSet(nbTestingPatches, geometry)

--testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
-- training function

function train(datasetX,datasetY)
-- epoch tracker
	epoch = epoch or 1

        -- local vars
	local time = sys.clock()

	-- do one epoch
        print('<trainer> on training set:')
        print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,(datasetX:size())[1],opt.batchSize do
		-- create mini batch
		local inputs = torch.Tensor(opt.batchSize,15,geometry[1],geometry[2])
        	local targets = torch.Tensor(opt.batchSize)
        	local k = 1
		for i = t,math.min(t+opt.batchSize-1,(datasetX:size())[1]) do
	                -- load new sample
			local sampleX = datasetX[{{i},{},{},{}}]
			local sampleY = datasetY[{{i},{1}}]
			--print(datasetY:size())
                        local input = sampleX:clone()
                        --local _,target = sampleY[2]:clone():max(1)
                        --target = target:squeeze()
                        local target = sampleY:clone()
                        inputs[k] = input
                        targets[k] = target
                        k = k + 1
		end
		print(inputs:size())
		print(targets:size())
		
		
                -- create closure to evaluate f(X) and df/dX
                local feval = function(x)
			-- just in case:
                         collectgarbage()


		-- get new parameters
                if x ~= parameters then
			parameters:copy(x)
                end

                -- reset gradients
                gradParameters:zero()
               
		-- evaluate function for complete mini batch
                model:cuda()
		inputs=inputs:cuda()
		local outputs = model:forward(inputs)
		criterion:cuda()
		targets=targets:cuda()
		local f = criterion:forward(outputs, targets)
		print('Size of Outputs')
		print(outputs:size())
                -- estimate df/dW
                df_do = criterion:backward(outputs, targets)
                 model:backward(inputs, df_do)
		
		
                 -- penalties (L1 and L2):
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
        xlua.progress(t, datasetX:size())
	else
	        error('unknown optimization method') 
	end

	end

	-- time taken
--	time = sys.clock() - time
--	time = time / datasetX:size()
--	print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	confusion:zero()

	-- save/log current net
	local filename = paths.concat(opt.save, 'mnist.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	if paths.filep(filename) then
		os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
	end
	print('<trainer> saving network to '..filename)

	-- torch.save(filename, model)

	-- next epoch
	   epoch = epoch + 1
	
end
-------------------------------------------------
--Calling  trainingFunction :
train(datasetX,datasetY)
-------------------------------------------------
-- test function
--[[
function test(dataset)

-- local vars
local time = sys.clock()
-- test over given dataset

print('<trainer> on testing Set:')
for t = 1,dataset:size(),opt.batchSize do
	-- disp progress
	xlua.progress(t, dataset:size())
	-- create mini batch
	local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
                                                                                                                                                                                                                                                                                                          local targets = torch.Tensor(opt.batchSize)
]]--
