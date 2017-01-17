--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'

--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
   
    --learningRate = opt.LR
    learningRate = 0.0,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
   print ("calling paramforepoch***********")
   local regimes = {
	-- start, end, WD,
        {  1,     18,   5e-4, },
        { 19,     29,   5e-4  },
        { 30,     43,   0 },
        { 44,     52,   0 },
        { 53,    1e8,   0 },
   }
    	
    
    for _, row in ipairs(regimes) do
        if epoch > row[1] and epoch <= row[2] then
            return { weightDecay=row[3] }, epoch == row[1]
	elseif epoch==row[1] then
	   return {learningRate=1e-2,weightDecay=row[3]},epoch==row[1]
	end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch
local learningRate_global=1e-2

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train(var_flag)
   print ("Value of var_flag", var_flag)
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
                          
   --local params, newRegime = paramsForEpoch(epoch)
  --print (newRegime)
  --if newRegime then
     -- print ("Insidde newregime")
      if var_flag==0 then
          print ("***************Inside var_flag 0")
          optimState = {
             learningRate = learningRate_global,
             learningRateDecay = 0.0,
             momentum = opt.momentum,
             dampening = 0.0,
             --weightDecay = params.weightDecay
             weightDecay = 5e-4
          }
      elseif var_flag==1 then
	  print("changing learning rate...current learning rate is",learningRate_global)
          learningRate_global=learningRate_global*0.95
          print ("*******Changed learning Rate...update learning rate is %f",learningRate_global)
          optimState = {
             learningRate = learningRate_global,
             learningRateDecay = 0.0,
             momentum = opt.momentum,
             dampening = 0.0,
            -- weightDecay = params.weightDecay
             weightDecay = 5e-4
          }
    
      end
   --end
   batchNumber = 0
   cutorch.synchronize()
   -- set the dropouts to training mode
   model:training()
   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
  for i=1,opt.epochSize do
  --   for i = 1,10 do	
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
--            print('inputs in add job',inputs)
            return inputs, labels
         end,
            --print('Outside loader function')
         -- the end callback (runs in the main thread)
        trainBatch
      )
      end
	--[[[local_err=error_global
	if(local_err-prev_err<1e-4) then
	    count=count+1
	        if (count>10)then
	            local_lr = local_lr * 0.95
		    optimState[learningRate]=local_lr
	        end
        else
            prev_err=local_err
	end
	print ("*****LearningRate is **********%f",learningRate)
  	]] 
   
   donkeys:synchronize()
   cutorch.synchronize()
   --learningRate_global = local_lr

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   --[[local function sanitize(net)
      local list = net:listModules()
      for _,val in ipairs(list) do
            for name,field in pairs(val) do
               if torch.type(field) == 'cdata' then val[name] = nil end
               if (name == 'output' or name == 'gradInput') then
                  val[name] = field.new()
               end
            end
      end
   end
   sanitize(model)
   --]]
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
return loss_epoch
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   --inputs:resize(inputsCPU:size()):copy(inputsCPU)
   local inputs1=inputsCPU[1]:cuda()
--   print("size of inputs1",#inputs1)
   local inputs2=inputsCPU[2]:cuda()
--   print("size of inputs2",#inputs2)
   local inputs3=inputsCPU[3]:cuda()
   inputs={inputs1,inputs2,inputs3}
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
     --print(model)
      --print(#inputs)
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1 Acc -%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
   --setError(err)
end



