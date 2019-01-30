function [contour_image] =  xy_contour_to_image(contour_xy_sorted, sor_image)


[height,width,chanel] = size(sor_image);


contour_image =  false(height,width);

% contour_image_rgb  =  zeros(height,width,3,'uint8');

% contour_image_rgb(:,:,1) = 0;
% contour_image_rgb(:,:,2) = 0;
% contour_image_rgb(:,:,3) = 0;

for i = 1:length(contour_xy_sorted)
    contour_image(contour_xy_sorted(i,1),contour_xy_sorted(i,2)) = 1;
%     contour_image_rgb(contour_xy_sorted(i,1),contour_xy_sorted(i,2),1) =255;
%     contour_image_rgb(contour_xy_sorted(i,1),contour_xy_sorted(i,2),2) =0;
%     contour_image_rgb(contour_xy_sorted(i,1),contour_xy_sorted(i,2),3) =0;
end

%imwrite(contour_image,[data_dir,'contour_image.jpg']);
% imwrite(contour_image_rgb,[data_dir,'contour_image_rgb.jpg']);


end