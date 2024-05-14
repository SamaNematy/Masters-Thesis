%% shading correction using pixel-based algorithm according to:
    % Tak, Yoon-Oh, et al. "Simple shading correction method for brightfield
    % whole slide imaging." Sensors 20.11 (2020): 3084.

close all; clear; clc;

n = 100;
m = zeros(1,3);
I = imread('C:\Users\Sama\Desktop\Folders\Datasets\234-01-67.tif',1);
M = size(I, 1); N = size(I, 2);
flatfield = zeros(M, N, 3); % final RGB shading model image


for i = 1:3

    Is = zeros(M, N, n);
    % RGB channels separation
    for j = 1:n  
        I = imread('C:\Users\Sama\Desktop\Folders\Datasets\234-01-67.tif',j);
        I = im2double(I);
        Is(:,:,j) = I(:,:,i);
    end
    % reduce the size of image to speed up 
    Is = imresize(Is, 0.25);
    Is_sort = sort(Is, 3);
    candidates = Is_sort(:,:,65:100); % select candidates from sorted images 
    M_new = size(candidates,1); N_new = size(candidates,2);
    clear Is_sort
    LCoV = zeros(1, size(candidates,3)); % Local coefficient of variation 
    
    for k = 1:numel(LCoV)
        I = candidates(:,:,k);
%         siz = [5 5];
%         h = ones(siz);
%         r = conv2(I, h, 'same');
%         LCov = sum(r(:));
          LCov = 0; 
        for l = 3:(M_new - 2)
            for p = 3:(N_new - 2)
                I1 = I(l-2:l+2,p-2:p+2);
                LCov_new = std2(I1)/mean2(I1);
                LCov = LCov + LCov_new;
            end
        end 
        LCoV(k) = LCov;
    end
    [Min, inds] = min(LCoV(:));
    m(i) = Min;
    shading = candidates(:,:,inds);
    shading = imresize(shading, [M, N]);
    flatfield(:,:,i) = shading ./ mean(mean(shading(:)));
    
    
    
end

  
  
for i = 1:n
    I = imread('C:\Users\Sama\Desktop\Folders\Datasets\Avesinatech\Original images.tif',i);
    I = im2double(I);
    I = I ./ flatfield;
    imwrite(I, ['img' num2str(i) '.tif']);
end



