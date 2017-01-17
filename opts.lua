--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    --local defaultDir = '/home/sk1846/Activity_recog/results/Y'
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache','/home/sk1846/Activity_recog/multi_results/YCbCrOF/checkpoint',
               'subdirectory in which to save/log experiments')
    cmd:option('-data','/home/sk1846/Activity_recog/all_data/UCF-101-uint8',
               'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | ccn2 | cunn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        4, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-imageSize',         256,    'Smallest side of the resized image')
    cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',        101, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         50,    'Number of total epochs to run')
    cmd:option('-epochSize',       750, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       64,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'YCbCrOf_fusion_model', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet | parModel |YCbCrOf_fusion_model')
    cmd:option('-retrain',     '/home/sk1846/Activity_recog/multi_results/YCbCrOF/checkpoint/alexnet12/,SunApr1017:48:102016/model_26.t7', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache,
                            cmd:string('alexnet12', opt,
                                       {retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, ',' .. os.date():gsub(' ',''))
    return opt
end

return M
