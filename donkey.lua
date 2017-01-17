--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
imageMod = require 'image'
--mattorch=require 'fb.mattorch'
--require 'mattorch'
paths.dofile('dataset.lua')
paths.dofile('util.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, 'testCache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')
local y_meanstd=torch.load('/home/sk1846/Activity_recog/multi_results/cacheDir/Y_meansstdCache.t7')
local Y_mean=y_meanstd.mean
local Y_std=y_meanstd.std

local cbcr_meanstd=torch.load('/home/sk1846/Activity_recog/multi_results/cacheDir/CbCr_meansstdCache.t7')
local CbCr_mean=cbcr_meanstd.mean 
local CbCr_std=cbcr_meanstd.std   

local of_meanstd=torch.load('/home/sk1846/Activity_recog/multi_results/cacheDir/OF_meansstdCache.t7')
local OF_mean=of_meanstd.mean
local OF_std=of_meanstd.std

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {10, opt.imageSize, opt.imageSize}
local loadSize1   = {10,opt.imageSize, opt.imageSize}
local loadSize2   = {20,opt.imageSize, opt.imageSize}
local sampleSize = {10, opt.cropSize, opt.cropSize}
local sampleSize1 = {10,opt.cropSize, opt.cropSize}
local sampleSize2 = {20,opt.cropSize,opt.cropSize}



local function loadImage(path)
   --print(path)
   --local path   
   local matio=require 'matio' 
   local input = matio.load(path)
   if input.Y_data then 
   input=input.Y_data
   input=input:transpose(1,2)
   input=input:transpose(1,3)
   input=input:float()
   end
   if input.CbCr_data  then
   input=input.CbCr_data
   input=input:transpose(1,2)
   input=input:transpose(1,3)
   input=input:float()
   end 
   if input.OF_data then 
   input=input.OF_data
   input=input:transpose(1,2)
   input=input:transpose(1,3)
   input=input:float()
   end
   return input
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local Y_mean,Y_std,CbCr_mean,CbCr_std,OF_mean,OF_std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   --print("path in trainHOOK",path)
   collectgarbage()
   local input = loadImage(path)
   --local iW = input:size(3)
   --local iH = input:size(2)

  -- require 'image'
   -- do random crop
   --local oW = sampleSize[3]
   --local oH = sampleSize[2]
   --local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   --local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   --local out = imageMod.crop(input, w1, h1, w1 + oW, h1 + oH)
   local out=input
   --assert(out:size(3) == oW)
   --assert(out:size(2) == oH)
   -- do hflip with probability 0.5
   --if torch.uniform() > 0.5 then out = imageMod.hflip(out) end
   -- mean/std
   --for i=1,3 do -- channels
   if out:size(1)==10 then 
   for i=1,10 do --channels
      if Y_mean then out[{{i},{},{}}]:add(-Y_mean[i]) end
      if Y_std then out[{{i},{},{}}]:div(Y_std[i]) end
   end
   end
   if out:size(1)==20 then
   for i=1,20 do 
     if CbCr_mean then out[{{i},{},{}}]:add(-CbCr_mean[i]) end
     if CbCr_std then out[{{i},{},{}}]:div(CbCr_std[i]) end
   end
   end
   if out:size(1)==3 then
   for i=1,3 do
     if OF_mean then out[{{i},{},{}}]:add(-OF_mean[i]) end
     if OF_std then out[{{i},{},{}}]:div(OF_std[i]) end
   end
   end
   return out
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   assert(trainLoader.paths[1] == paths.concat(opt.data, 'train_new'),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. trainCache .. ' and rerun the program')
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train_new')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   --print(class)
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
testHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)
   --local oH = sampleSize[2]
   --local oW = sampleSize[3]
   --local iW = input:size(3)
   --local iH = input:size(2)
   --local w1 = math.ceil((iW-oW)/2)
   --local h1 = math.ceil((iH-oH)/2)
   --local out = imageMod.crop(input, w1, h1, w1+oW, h1+oH) -- center patch
   local out=input
   -- mean/std
   --for i=1,3 do -- channels

   if out:size(1)==10 then 
   for i=1,10 do -- channels
      if Y_mean then out[{{i},{},{}}]:add(-Y_mean[i]) end
      if Y_std then out[{{i},{},{}}]:div(Y_std[i]) end
   end
   end
   if out:size(1)==20 then
   for i=1,20 do
      if CbCr_mean then out[{{i},{},{}}]:add(-CbCr_mean[i]) end
      if CbCr_std then out[{{i},{},{}}]:div(CbCr_std[i]) end
   end
   end
   if out:size(1)==3 then
   for i=1,3 do -- channels
      if OF_mean then out[{{i},{},{}}]:add(-OF_mean[i]) end
      if OF_std then out[{{i},{},{}}]:div(OF_std[i]) end
   end
   end
   return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache')
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
   assert(testLoader.paths[1] == paths.concat(opt.data, 'test_new'),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. testCache .. ' and rerun the program')
else
   print('Creating test metadata')
   testLoader = dataLoader{
      paths = {paths.concat(opt.data, 'test_new')},
      loadSize = loadSize,
      sampleSize = sampleSize,
      split = 0,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()

-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)

if paths.filep(meanstdCache) then
   local meanstd = torch.load(meanstdCache)
   mean = meanstd.mean
   std = meanstd.std
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = 10000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   --local meanEstimate = {0,0,0}
   local meanEstimate = {0,0,0,0,0,0,0,0,0,0}--,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
   for i=1,nSamples do
      --print(i)
      local img = trainLoader:sample(1)[1]
      --print(img[{{},{1},{1}}])
      --for j=1,3 do
      for j=1,10 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   --for j=1,3 do
   for j=1,10 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   --local st1dEstimate = {0,0,0}
   local stdEstimate = {0,0,0,0,0,0,0,0,0,0}--,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      --for j=1,3 do
      for j=1,10 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   --for j=1,3 do
   for j=1,10 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanstdCache, cache)
   print('Time to estimate:', tm:time().real)
end

