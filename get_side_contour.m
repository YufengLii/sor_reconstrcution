function [contour_x, contour_y, hang, lie] = get_side_contour(data_dir, filename, contour_image_gray, ellipse_top_center, ellipse_bottom_center)

contour_image_gray_left = contour_image_gray;
contour_image_gray_right = contour_image_gray;

figure(1);imshow(contour_image_gray);
[hang,lie]=size(contour_image_gray);

%% 椭圆上的点
[x_top, y_top] = ellipse_xy(ellipse_top_center(3),ellipse_top_center(4), ...
    ellipse_top_center(5),ellipse_top_center(1),ellipse_top_center(2),'r');
[x_bottom, y_bottom]= ellipse_xy(ellipse_bottom_center(3),ellipse_bottom_center(4), ...
    ellipse_bottom_center(5),ellipse_bottom_center(1),ellipse_bottom_center(2),'r');


eval ( [filename(1:end-4),'_ellipse_top_center_xy=[ellipse_top_center(1) ellipse_top_center(2)]' ]) 
eval ( [filename(1:end-4),'_ellipse_bottom_center_xy=[ellipse_bottom_center(1) ellipse_bottom_center(2)]' ]) 

save([data_dir,filename(1:end-4),'_ellipse_top_center_xy'],[filename(1:end-4),'_ellipse_top_center_xy']);
save([data_dir,filename(1:end-4), '_ellipse_bottom_center_xy'],[filename(1:end-4), '_ellipse_bottom_center_xy']);


xy_top = [x_top; y_top]';
xy_bottom = [x_bottom; y_bottom]';

xy_top = sortrows(xy_top, 1);
xy_bottom = sortrows(xy_bottom, 1);

%% 过两椭圆中心点的直线的斜率
ks =  (ellipse_bottom_center(1) - ellipse_top_center(1)) / ...
    (ellipse_bottom_center(2) - ellipse_top_center(2)) 


for i = 1:hang
    y_split = (i-ellipse_top_center(2)) * ks + ellipse_top_center(1);
    y_split_int = abs(round(y_split));
    contour_image_gray_left(i, y_split_int:end ) = 0;
    contour_image_gray_right(i, 1:y_split_int ) = 0;
end


contour_image_gray_left(1:xy_top(1,2),:) = 0;
contour_image_gray_left( xy_bottom(1,2):end, :) = 0;


contour_image_gray_right(1:xy_top(1,end),:) = 0;
contour_image_gray_right( xy_bottom(1,end):end, :) = 0;


figure(2);
imshow(contour_image_gray_left);
axis equal;
figure(3)
imshow(contour_image_gray_right);
axis equal;
left_or_right=input('请选择左右轮廓中的一条（1/2）：');

if(left_or_right == 1)
    [contour_x, contour_y]= find(contour_image_gray_left == 1);
elseif(left_or_right == 2)
    [contour_x, contour_y] = find(contour_image_gray_right == 1);
end


close all;


end
