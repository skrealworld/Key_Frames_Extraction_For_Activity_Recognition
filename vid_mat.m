clc;
DBpath='~/skdata/UCF-101/'; % DataBase path 
load('~/skdata/ClassLabels.mat') % Load the labeled mat file for classNames.
count_smpl=1;
count_idx=0;
label=1;
noOfSamp1Vid=5;    % Define the number if training videos from each class. 
for f1=1:2     % Define the number of classes.
    for f2=1:noOfSamp1Vid
        obj=VideoReader(strcat(DBpath,classNames{f1},'/',classNames{f1},' (',num2str(f2),').avi')); %Access the video.
            nFrames=obj.NumberOfFrames; % Get the number of frames. 
            n10=1; 
            for n1=5:20:nFrames   
                t_dim=0;
                for n2=n1-4:n1+5
                    t_dim=t_dim+1;
                    if(n2>=nFrames)    
                        break;
                    end
                    idx=(180*count_idx)+1;
                    im=read(obj,n2);   % Read the selected frame. 
                    im=imresize(im,[60,80]); % Change the resolution from 240x320 to 60x80. 
                    frame(idx:idx+60-1,:,:)=im(:,:,1);          % Stacking all three(R,G,B) components of image. 
                    frame(idx+60:idx+120-1,:,:)=im(:,:,2);
                    frame(idx+120:idx+180-1,:,:)=im(:,:,3);
                    count_idx=count_idx+1;
                    vid_data(:,:,t_dim,count_smpl)=frame(idx:idx+180-1,:);
                end
                Y(count_smpl)=label;
                count_smpl=count_smpl+1;
                n10=n10+1;
            end
            sprintf('Number of samples of %d th video of class %d is %d',f2,f1,n10)
    end
    lable=label+1;
end
save('vid_data_training.mat','vid_data');
save('vid_data_training_labels.mat','Y');