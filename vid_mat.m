clc;
load('ClassNames.mat') % Load the labeled mat file for classNames.
%count_smpl=1;
%count_idx=0;
%f1=1;
for f1=26:50     % Define the number of classes.
    cd (strcat('~/skdata/UCF-101/',classNames{f1})); % DataBase path
    d =dir; 
    num_of_videos =((length(d)/2)-3)*2/3;

    for f2=1:num_of_videos   
        obj=VideoReader(strcat(classNames{f1},' (',num2str(f2),').ogv')); %Access the video.
            nFrames=obj.NumberOfFrames; % Get the number of frames. 
            n10=1; %????
            for n1=8:20:nFrames   
                t_dim=0;
                for n2=n1-7:n1+7
                    t_dim=t_dim+1;
                    if(n2>=nFrames)    
                        break
                    end
                    %idx=(180*count_idx)+1;
                    im=read(obj,n2);   % Read the selected frame. 
                    im=imresize(im,[64,64]); % Change the resolution from 240x320 to 64x84. 
                    im_temp= rgb2ycbcr(im);
                    im = im_temp(:,:,1);
                    %frame(idx:idx+60-1,:,:)=im(:,:,1);          % Stacking all three(R,G,B) components of image. 
                    %frame(idx+60:idx+120-1,:,:)=im(:,:,2);
                    %frame(idx+120:idx+180-1,:,:)=im(:,:,3);
                    %count_idx=count_idx+1;
                   vid_data(:,:,t_dim,count_smpl)=im;
                   
                end
                
                Y(count_smpl)=f1;
                count_smpl=count_smpl+1;
                n10=n10+1;
            end
            sprintf('Number of samples of %d th video of class %d is %d',f2,f1,n10)
    end
    %label=label+1;
end
%tt= vid_data(:,:,:,1:707);
%yy= 6+Y(1:707);
cd ('~/matlab_work');
save('vid_data_train26_50.mat','vid_data');
save('vid_data_trainlabels126_50.mat','Y');