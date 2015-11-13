clc;
load('ClassNames.mat') % Load the labeled mat file for classNames.
load('training_per_class.mat'); % Load the file to get number of videos per class for testing
count_smpl=1;
%count_idx=0;
label=1;
for f1=1:101     % Define the number of classes.
    cd (strcat('~/skdata/UCF-101/',ClassNames{f1})); % DataBase path
    num_of_videos = length(dir('*.ogv'));
    
    start_index = training_per_class(f1);
    for f2=start_index:num_of_videos
	   
        obj=VideoReader(strcat(ClassNames{f1},'(',num2str(f2),').ogv')); %Access the video.
            nFrames=obj.NumberOfFrames; % Get the number of frames. 
            n10=1; %????
            for n1=8:20:nFrames-7   
                t_dim=0;
                for n2=n1-7:n1+7
                    t_dim=t_dim+1;
                    if(n2>=nFrames)    
                        break
                    end
                    %idx=(180*count_idx)+1;
                    im=read(obj,n2);   % Read the selected frame. 
                    im=imresize(im,[64,64]); % Change the resolution from 240x320 to 64x64. 
                    im_temp= rgb2ycbcr(im);
                    im = im_temp(:,:,1);
                    %frame(idx:idx+60-1,:,:)=im(:,:,1);          % Stacking all three(R,G,B) components of image. 
                    %frame(idx+60:idx+120-1,:,:)=im(:,:,2);
                    %frame(idx+120:idx+180-1,:,:)=im(:,:,3);
                    %count_idx=count_idx+1;
                    vid_data(:,:,t_dim,count_smpl)=im;
                end
                Y(count_smpl)=label;
                count_smpl=count_smpl+1;
                n10=n10+1;
            end
            sprintf('Number of samples of %d th video of class %d is %d',f2,f1,n10)
    end
    label=label+1;
end
%tt= vid_data(:,:,:,1:707);
%yy= 6+Y(1:707);
cd ('~/matlab_work');
save('testX.mat','vid_data');
save('testY.mat','Y');
%save('train_num1_30_2.mat','train_num');
