clc;
clear all;
DBpath='C:\Users\sa5641\Desktop\videoClassification\SubDataUCF101\'; % DataBase path 
load('ClassLabels.mat')
count_smpl=1;
count_idx=0;
label=1;
noOfSamp1Vid=1;
for f1=1:10
    
    for f2=1:noOfSamp1Vid
        if(rem(f2,35)==0)
            label=label+1;
        end
        
        %vidobj=VideoReader(['/Users/sourabhkulhare/Documents/MATLAB/3DSIFT_CODE_v1/Small_test/v',num2str(f),'.avi']);
        % obj=VideoReader(['C:\Users\sk1846\Desktop\Matconvnet\Small UCF101_20\FrontCrawl\frontcrawl (',num2str(f),').avi']);
        a=strcat(DBpath,ClassLabels{f1},'\',ClassLabels{f1},' (',num2str(f2),').avi')
        obj=VideoReader(strcat(DBpath,ClassLabels{f1},'\',ClassLabels{f1},' (',num2str(f2),').avi'));
        
            nFrames=obj.NumberOfFrames;
        
            for n1=5:20:nFrames
                t_dim=0;
                for n2=n1-4:n1+5
                    t_dim=t_dim+1;
                    if(n2>=nFrames)
                        break;
                    end
                    idx=(720*count_idx)+1;
                    im=read(obj,n2);
                    frame(idx:idx+240-1,:,:)=im(:,:,1);
                    frame(idx+240:idx+480-1,:,:)=im(:,:,2);
                    frame(idx+480:idx+720-1,:,:)=im(:,:,3);
                    count_idx=count_idx+1;
                    vid_data(:,:,t_dim,count_smpl)=frame(idx:idx+720-1,:);
                end
                Y(count_smpl)=label;
                count_smpl=count_smpl+1;
            end
    end
end