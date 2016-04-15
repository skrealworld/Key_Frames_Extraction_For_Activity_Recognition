
local function getNumber(path)
--Function to get the numbers from a string
--get the end index, where _K ends_
_,end_idx=string.find(path,"_OF_K")

--this will a string like 5_55 or 44_64, so only numbers with underscore
temp_str=string.sub(path,end_idx+1,string.len(path)-4)

--IF current keys number is in two digits, it means that total number
-- of keys will also be two digits.

if string.find(temp_str,"_")==3 then
    current_key_num=string.sub(temp_str,1,2)
    total_keys=string.sub(temp_str,4,5)
-- If current key number is a single digit number
else
    if string.len(temp_str)==4 then
        current_key_num=string.sub(temp_str,1,1)
        total_keys=string.sub(temp_str,3,4)
    else
        current_key_num=string.sub(temp_str,1,1)
        total_keys=string.sub(temp_str,3,3)
    end
end

return {current_key_num,total_keys}
end


local function getPastPATH(path)
--A function to get the path of last clip

numbers=getNumber(path)

_,endidx=string.find(path,"_OF_K")
if string.len(numbers[2])==2 then

    if tonumber(numbers[1])~=1 then
        pastPATH = string.sub(path,1,endidx)..(numbers[1]-1)..string.sub(path,-7,-1)
    else
        pastPATH = string.sub(path,1,endidx)..(numbers[1]+2)..string.sub(path,-7,-1)
    end
else

    if tonumber(numbers[1])~=1 then
        pastPATH = string.sub(path,1,endidx)..(numbers[1]-1)..string.sub(path,-6,-1)
    else
        pastPATH = string.sub(path,1,endidx)..(numbers[1]+2)..string.sub(path,-6,-1)
    end
end

return pastPATH

end


local function getFuturePATH(path)
--A function to get the path of future clip

numbers=getNumber(path)

_,endidx=string.find(path,"_OF_K")
if string.len(numbers[2])==2 then

    if numbers[1]~=numbers[2] then
        futurePATH = string.sub(path,1,endidx)..(numbers[1]+1)..string.sub(path,-7,-1)
    else
        futurePATH = string.sub(path,1,endidx)..(numbers[1]-2)..string.sub(path,-7,-1)
    end
else

    if numbers[1]~=numbers[2] then
        futurePATH = string.sub(path,1,endidx)..(numbers[1]+1)..string.sub(path,-6,-1)
    else
        futurePATH = string.sub(path,1,endidx)..(numbers[1]-2)..string.sub(path,-6,-1)
    end
end

return futurePATH
end
