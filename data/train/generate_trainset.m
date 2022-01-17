clc
clear
%%
dataset_name = 'Davinci_fold1'
img_list = dir(['./', dataset_name, '/*.png']);

for idx_file = 1:2:length(img_list)
    file_name = img_list(idx_file).name;
    idx_name = find(file_name == '_');
    file_name = file_name(1:idx_name-1);
    
    img_0 = imread(['./', dataset_name, '/' ,img_list(idx_file).name]);
    img_1 = imread(['./', dataset_name, '/', img_list(idx_file+1).name]);
	N = 10
    
    %% x2 or x4
    scale_lisht = [4];
    for idx_scale = 1:length(scale_lisht)
        scale = scale_lisht(idx_scale);
        
        %% generate HR & LR images
        img_hr_0 = modcrop(img_0, scale);
        img_hr_1 = modcrop(img_1, scale);
        img_lr_0 = imresize(img_hr_0, 1/scale, 'bicubic');
        img_lr_1 = imresize(img_hr_1, 1/scale, 'bicubic');
        
        %% extract patches 
        for idx_patch = 1:N
            x_lr = randi([1 size(img_lr_0,1)-33]);
            y_lr = randi([1 size(img_lr_0,2)-93]);
            x_hr = (x_lr-1) * scale + 1;
            y_hr = (y_lr-1) * scale + 1;
            hr_patch_0 = img_hr_0(x_hr:(x_lr+29)*scale,y_hr:(y_lr+89)*scale,:);
            hr_patch_1 = img_hr_1(x_hr:(x_lr+29)*scale,y_hr:(y_lr+89)*scale,:);
            lr_patch_0 = img_lr_0(x_lr:x_lr+29,y_lr:y_lr+89,:);
            lr_patch_1 = img_lr_1(x_lr:x_lr+29,y_lr:y_lr+89,:);
                
            mkdir(['./', dataset_name, '_patches/patches_x', num2str(scale), '/', file_name,'_', num2str(idx_patch, '%03d')]);
            imwrite(hr_patch_0, ['./', dataset_name, '_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/hr0.png']);
            imwrite(hr_patch_1, ['./', dataset_name, '_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/hr1.png']);
            imwrite(lr_patch_0, ['./', dataset_name, '_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/lr0.png']);
            imwrite(lr_patch_1, ['./', dataset_name, '_patches/patches_x', num2str(scale), '/', file_name, '_', num2str(idx_patch, '%03d'), '/lr1.png']);
            
        end
    end
end