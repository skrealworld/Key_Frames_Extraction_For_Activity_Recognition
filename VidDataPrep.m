clc;
load('ClassLabels.mat') % Load the labeled mat file for classNames.
%count_smpl=1;
%count_idx=0;
%f1=1;
VidDataClass=[];
VidDataLabel=[];
parfor f1=1:5     % Define the number of classes.
    cd (strcat('~/VideoClassification/dataSet/UCF-101/',classNames{f1})); % DataBase path
    d =dir; 
    num_of_videos =(length(d)/2)-1;
    vid_data =[];
    Y=[];
    
    for f2=1:num_of_videos/10   
        
        obj=VideoReader(strcat(classNames{f1},' (',num2str(f2),').ogv')); %Access the video.
        nFrames=obj.NumberOfFrames; % Get the number of frames.
        n10=1; %????
        loop1=(nFrames/20);
        for n1=1:loop1 % totoal number of samples 
            
            if (n1<5)
            %t_dim=0;
            
            
            for n2=1:15
                %t_dim=t_dim+1;
                workFrameNo=( (n1-1) *20)+ n2; % calculates the working frame 
                
                %if(n2>=nFrames)
                %    break
                %end
                %idx=(180*count_idx)+1;
                im=read(obj,workFrameNo);   % Read the selected frame.
                im=imresize(im,[64,64]); % Change the resolution from 240x320 to 64x84.
                im_temp= rgb2ycbcr(im);
                im = im_temp(:,:,1);
                %frame(idx:idx+60-1,:,:)=im(:,:,1);          % Stacking all three(R,G,B) components of image.
                %frame(idx+60:idx+120-1,:,:)=im(:,:,2);
                %frame(idx+120:idx+180-1,:,:)=im(:,:,3);
                %count_idx=count_idx+1;
                vid_data(:,:,n2,n1)=im;
                
            end
            
            Y(n1)=f1;
            %count_smpl=count_smpl+1;
            %n10=n10+1;
        
            end
        end
        %sprintf('Number of samples of %d th video of class %d is %d',f2,f1,n10)
    end
    %label=label+1;
    VidDataClass(f1,:,:,:,:)=vid_data;
    VidDataLabel(:,f1)=Y; 
end
%tt= vid_data(:,:,:,1:707);
%yy= 6+Y(1:707);
% for i2=1:numberOfClasses
%     vidDataAll=vertcat(vidDataAll,(genvarname(['VidDataClass' int2str(i2)] )));
%     vidDataLabesAll=vertcat(vidDataLabelAll,(genvarname(['VidDataLabel' int2str(i2)] )));
% end
% cd ('~/matlab_work');
% save('vid_data_train_1to5.mat','VidDataInStruct');
% save('vid_data_trainlabels_1to5.mat','VidDataOutStruct');
