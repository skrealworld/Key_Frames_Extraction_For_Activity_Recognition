--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local top1_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   top1_center = 0
   loss = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, labels = testLoader:get(indexStart, indexEnd)
            return inputs, labels
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / nTest
   loss = loss / (nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
os.execute('rm t.txt')
function testBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   batchNumber = batchNumber + opt.batchSize

   local inputs1=inputsCPU[1]:cuda()
   local inputs2=inputsCPU[2]:cuda()
   local inputs3=inputsCPU[3]:cuda()
   inputs={inputs1,inputs2,inputs3}
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   local outputs = model:forward(inputs)
   local err = criterion:forward(outputs, labels)
   --cutorch.synchronize()
   local pred = outputs:float()

   loss = loss + err

   local _, pred_sorted = pred:sort(2, true)
   file1=io.open("t.txt","aw")
   for i=1,pred:size(1) do
      local g = labelsCPU[i]
      file1:write(g .. "******" .. pred_sorted[i][1])
      file1:write("\n")

      if pred_sorted[i][1] == g then top1_center = top1_center + 1 end
   end
   if batchNumber % 1024 == 0 then
      print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   end
end
