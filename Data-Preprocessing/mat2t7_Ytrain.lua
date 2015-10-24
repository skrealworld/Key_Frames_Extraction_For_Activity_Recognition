--[[
This script imports many .mat files from a directory and then create batches
preprocess it, normalize the data and save data .t7 file.


By- Sourabh Kulhare
--]]

require 'mattorch'
require 'nn'
require 'image'
-- Import data file

--[[
--get the number of .mat files available in the current directory
local handle1 = io.popen('find ./UCF-11/train_Y -type f | wc -l')  
num_of_files = tonumber(handle1:read())
print('Number of files : - ', num_of_files)

--]]

--get the names of all .mat files available in the current directory.  
local handle = io.popen('find ./UCF-11/train_Y -name "*.mat"')
io.input(handle)

names = {}

for line in io.lines() do
	--remove spaces from line 
	line = line:gsub("%s+", "")
	table.insert(names,line)
end

print('Number of samples:-', #names)
rand_idx = torch.randperm(#names)
normalization = nn.SpatialContrastiveNormalization(1,image.gaussian1D(7)):float()
group=1

print('Processing group number '..group)
for k=1,#names,2000 do

trainset = {}
trainset.data = torch.FloatTensor(2000,10,224,224)
trainset.label = torch.ByteTensor(2000,1)


function trainset:size()
        return self.data:size(1)
end


setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
temp1 = 1
for i=k,math.min(k+2000-1,#names) do
local temp_name = names[rand_idx[i]]
temp_str = string.sub(temp_name,-6,-5)

--print(temp_str)

if string.sub(temp_str,1,1)=='C' then 
	temp_label = string.sub(temp_str,2,2)
else
	temp_label = temp_str
end 

local temp_data = mattorch.load(temp_name)
temp_data = temp_data.data

--Normalize the Y-data locally. 
for j=1,10 do  -- #channels
	-- y channeled frames of a train samp
	-- normalize y locally:
	temp_data[{{j},{},{}}] = normalization(temp_data[{{j},{},{}}]:float())
end
--print(temp_label)
trainset.data[{temp1}]= temp_data:float()
trainset.label[{temp1}] = temp_label
temp1 = temp1+1
end  
torch.save('./Y_data/train_files/Y_224_trainset_G' .. group ..'.t7',trainset)
print('completed group number ' .. group)
group = group+1
end





