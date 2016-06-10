require 'loadcaffe'
require 'cudnn'
require 'nn'
require 'cutorch'
require 'paths'
require 'image'
require 'mattorch'

-- Import trained model 
model=torch.load('GoogLeNet_v2.t7')
model=model:cuda()
--Deal with each directory. 

--Remove extra layers
model:remove(30)
model:remove(29)

--print(model)


--Data directories. 
data_path = "./image_folders/."
folder_names={}

function preprocessImage(img) 

    -- Extrqact the center crop 
    height = img:size(2)
    width = img:size(3)
    if height>width then 
        ret_img = image.crop(img,1,(height/2)-(width/2),width,(height/2)+(width/2)) 
    else       
        ret_img = image.crop(img,(width/2)-(height/2),1,(width/2)+(height/2),height) 
    end       

        ret_img = image.scale(ret_img,224,224)

        -- Substract mean. 
        ret_img = ret_img - 118.380948
        ret_img = ret_img / 61.896913
        return ret_img
end

g=1
for dir in paths.iterdirs(data_path) do
    print("Processing video : " , g)
    t1 = os.time()
    file_names = {}    
    for file in paths.iterfiles("./image_folders/" .. dir .. '/.') do
    table.insert(file_names,file)
    end
    
    num_files = #file_names
    table.sort(file_names)
    first_file = file_names[1]
    reverse_first_file = string.reverse(first_file)
    idx = string.find(reverse_first_file,"_")
    idx = string.len(first_file) - (idx-1)

    videoData = torch.rand(num_files,1024)
    for i=1,num_files do 
        temp_file = string.sub(first_file,1,idx) .. i .. ".jpg"
        temp_data = image.load("./image_folders/" .. dir .. "/" .. temp_file,3,'byte'):float()
        temp_data = preprocessImage(temp_data)
        imgdata=torch.rand(1,3,224,224)
        imgdata[1]=temp_data
        op = model:forward(imgdata:cuda())
        videoData[i] = op[1]:float()      
        
    end
    mattorch.save('./features/'.. dir .. ".mat",videoData)
    print ("Execution Time : ", os.difftime(os.time(),t1))
    g=g+1 
end


