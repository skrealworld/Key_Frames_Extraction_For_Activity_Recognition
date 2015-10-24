require 'mattorch'
require 'nn'
require 'image'
-- Import data file
--
--
--
trainset = {}

temp_data = mattorch.load('./data/CbCr_vid_data_train_64_corr.mat')

trainset.data = temp_data.train_data

-- Import label file

temp_label = mattorch.load('./data/CbCr_trainlabels_64.mat')

trainset.label = temp_label.train_label:float()

function trainset:size()
        return self.data:size(1)
end

print(trainset)

--channels = {'r','g','b'}
mean1 = {}  
std1 = {}  

print('Stated the normalization process for tranning.')
for j=1,trainset.data:size(2) do  -- #frames * #channels
  mean1[j] = torch.mean(trainset.data[{{},j,{},{}}]:float())
  std1[j] = torch.std(trainset.data[{{},j,{},{}}]:float())
  trainset.data[{{},j,{},{}}]:add(-mean1[j])
  trainset.data[{{},j,{},{}}]:div(std1[j])
end

torch.save('normCbCr_trainset.t7',trainset)

testset = {}

temp_data = mattorch.load('./data/CbCr_vid_data_test_64_corr.mat')

testset.data = temp_data.test_data

temp_label = mattorch.load('./data/CbCr_testlabels_64.mat');

testset.label = temp_label.test_label:float()

function testset:size()
        return self.data:size(1)
end

print(#testset)

--channels = {'r','g','b'}

for j=1,testset.data:size(2) do  -- #frames * #channels
  testset.data[{{},j,{},{}}]:add(-mean1[j])
  trainset.data[{{},j,{},{}}]:div(std1[j])
end

torch.save('normCbCr_testset.t7',testset)

--[[
--Create the table for data
Y_64_norm_testset = {}
Y_64_norm_testset.data = temp_data
Y_64_norm_testset.label = temp_label

--sizes testset-------------------------

function Y_64_norm_testset:size()
        return self.data:size(1)
end

--return pairs------------------------------------

setmetatable(Y_64_norm_testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


-- preprocess-------------------------------------
-- test data


  
  normalization = nn.SpatialContrastiveNormalization(1,image.gaussian1D(7)):float()
for i = 1,Y_64_norm_testset:size() do  --#examples
    local yTrain = torch.rand(15,64,64)
    local vid_frame = Y_64_norm_testset.data[i]
   for j=1,15 do  -- #channels
     -- y channeled frames of a train samp
     -- normalize y locally:
     yTrain[{{j},{},{}}] = normalize(vid_frame[{{j},{},{}}]):float()

   end

   trainset.data[i] = yTrain

end
torch.save('Y_64_norm_testset.t7',Y_64_norm_testset)

--]]





