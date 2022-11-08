clc;close all;clear;

dataset_path = 'xxx\Win5-LID\Distorted\'; % Set the dataset path here
savepath = 'xxx\Win5_160x160\'; % Set the save path here

% for Win5-LID dataset
load('win5_all_info.mat');
load('win5_all_mos.mat');
Distorted_sceneNum = 220; 

angRes = 5;             
patchsize = 32;         
stride = 64; 

for iScene = 1 : Distorted_sceneNum
    
    idx = 1;
    h5_savedir = [savepath, '\', win5_all_info{1}{iScene}, '\',  win5_all_info{2}{iScene}];
    if exist(h5_savedir, 'dir')==0
        mkdir(h5_savedir);
    end
    dataPath = [dataset_path, win5_all_info{6}{iScene}];
    LF = imread(dataPath);
    if size(LF,1) == 4608
        LF = permute(reshape(LF,[9, 512, 9, 512, 3]),[1,3,2,4,5]);
    else
        LF = permute(reshape(LF,[9, 434, 9, 625, 3]),[1,3,2,4,5]);
    end
    [U, V, ~, ~, ~] = size(LF);
    LF = LF(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), :, :, :);
    [U, V, H, W, ~] = size(LF);
 
    for h = 1 : stride : H - patchsize + 1
        for w = 1 : stride : W - patchsize + 1
            data = single(zeros(U * patchsize, V * patchsize));  
            label = win5_all_mos{iScene};
            for u = 1 : U
                for v = 1 : V                        
                    patch = squeeze(LF(u, v, h : h+patchsize-1, w : w+patchsize-1, :));
                    patch = rgb2ycbcr(patch);
                    patch = squeeze(patch(:,:,1)); 
                    data(u:angRes:U * patchsize, v:angRes:V * patchsize) = patch;  
                end
            end 
            SavePath_H5_name = [h5_savedir, '/', num2str(idx,'%06d'),'.h5'];
            h5create(SavePath_H5_name, '/data', size(data), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/data', single(data), [1,1], size(data));
            h5create(SavePath_H5_name, '/score_label', size(label), 'Datatype', 'single');
            h5write(SavePath_H5_name, '/score_label', single(label), [1,1], size(label));
            idx = idx + 1;
        end
    end
end




