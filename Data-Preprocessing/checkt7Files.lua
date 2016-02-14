--To get the list of directories and number of files present inside current directory. 
--To get subdirectories do "ls -d .*/" 

local handle1 = io.popen('ls -d */')
io.input(handle1)

dir_names={}

for line in io.lines() do
        line=line:gsub("%s+","")
        table.insert(dir_names,line)
end
handle1:close()

t7_count = {}
for i=1,#dir_names do
        handle2=io.popen('ls -lR ./' .. dir_names[i] .. '*.t7|wc -l')
        result = handle2:read("*a")
        t7_count[dir_names[i]]=result
        handle2:close()
end
