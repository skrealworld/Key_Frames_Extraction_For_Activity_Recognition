
require 'mattorch'
--[[
print('Finding ".mat" files.............')
local handle=io.popen('find -name "*.mat"')
io.input(handle)

print('Saving file names in a list........')

local file_names={}

for line in io.lines() do

        --remove spaces from the names
        line=line:gsub("%s+","")
        table.insert(file_names,line)
end

print('number of samples :- ', #file_names/4)

torch.save('file_names.t7',file_names)
--]]
file_names = torch.load('file_names.t7')

for i=1,#file_names do
-- data labels in .mat file
--Y - Y_data
--CbCr - CbCr_data
--RGB - RGB_data
--OF - OF_data

local temp_data = mattorch.load(file_names[i])

if(string.find(file_names[i],"_Y_")) then
temp_data=temp_data.Y_data
elseif(string.find(file_names[i],"_CbCr_")) then
temp_data=temp_data.CbCr_data
elseif(string.find(file_names[i],"_RGB_")) then
temp_data=temp_data.RGB_data
elseif(string.find(file_names[i],"_OF_")) then
temp_data=temp_data.OF_data
end

local temp_name = string.gsub(file_names[i],".mat",".t7")
torch.save(temp_name,temp_data)
                                                                                                                                          1,0-1         Top
