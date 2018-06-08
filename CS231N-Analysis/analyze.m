clc; close all; clear;
w = warning ('off','all');

base_name='paint/';
end_name='.png';

num_cols = 7;
num_rows = 10;
num_img = 5;

paint_names = {'General', 'Animal', 'Food', 'Furniture', 'Outdoor', 'Person', 'Vehicle'};
for k=0:num_img
    filename =[base_name, num2str(k), end_name];
    total_image = im2double(imread(filename));
    [h, w, c] = size(total_image);
    
    for j = 0:num_rows-1
        PSNR_MAX = 0;
        SSIM_MAX = 0;
        i_max_ssmi = 0;
        
        
        for i = 0:num_cols-1
            
            rec = total_image(3+(j*130):130+(j*130),...
                                   3+(i*130):130+(i*130), :);
            ref  = total_image(3+(j*130):130+(j*130), ...
                                    3+130*7:130*(8), :);
            
          PSNR = psnr(rec, ref, max(rec(:)));
          SSIM = ssim(rec, ref);            

            if SSIM > SSIM_MAX
                SSIM_MAX = SSIM;
                i_max_ssmi = i;
            end 
            
            if PSNR > PSNR_MAX
                PSNR_MAX = PSNR;
                i_max_psnr = i;
            end 
        end
        
        X = total_image(3+(j*130):130+(j*130), 3+(i_max_ssmi*130):130+(i_max_ssmi*130), :);
        total_image(3+(j*130):130+(j*130), 3+(i_max_ssmi*130):130+(i_max_ssmi*130), :) = addborder(X, 10, [1, 1, 0], 'inside');
        
        X = total_image(3+(j*130):130+(j*130), 3+(i_max_psnr*130):130+(i_max_psnr*130), :);
        total_image(3+(j*130):130+(j*130), 3+(i_max_psnr*130):130+(i_max_psnr*130), :) = addborder(X, 10, [1, 0, 0], 'inside');
        
        if (i_max_psnr == i_max_ssmi)
            total_image(3+(j*130):130+(j*130), 3+(i_max_psnr*130):130+(i_max_psnr*130), :) = addborder(X, 10, [1, 1, 1], 'inside');
        end 
        
    end
    save_base_name='output/painter_comp_';
    save_end_name='.png';
    filename =[save_base_name, num2str(k), save_end_name];
    imwrite(total_image, filename);
end