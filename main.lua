--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
--require 'mattorch'
torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')
paths.dofile('model.lua')
opt.imageSize = model.imageSize or opt.imageSize
opt.imageCrop = model.imageCrop or opt.imageCrop

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

epoch = opt.epochNumber
local prev_loss=0.0
local count=0
local var_flag = 0
local flag=true
local loss=0.0

for i=1,opt.nEpochs do
   --print ("previous loss= %f and current  ",prev_loss,loss) 
--[[
   if flag==true then
       loss=train(var_flag)
       flag=false
       
   else 
       loss = train(var_flag)
       var_flag=0
       if (math.abs(loss-prev_loss)<1e-4)then
           print "Change very less prev_loss and the current loss"
	   print ("previous loss is = %f and current Loss= %f",prev_loss,loss)
           count=count+1
           if (count>=4)then
               print("LEARING RATE WILL BE UPDATED!!!!!!!!!")
	       var_flag=1
	       count=0
           end
       else 
       --prev_loss=loss
       print ("previous= %f and current loss= %f",prev_loss,loss)
       prev_loss=loss
      -- print (loss,prev_loss)
       end
   end
]]--		
   test()
   break
   epoch = epoch + 1
end
