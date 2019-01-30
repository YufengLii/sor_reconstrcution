function []= rec_sor(dir)
    
%% 输入

% 图像    
    [filename,pathname]=uigetfile({'*.jpg';'*png'},'选择原图片');
    Image_in = imread([pathname,filename]);
    
    
%  椭圆   
%  坐标顺序 [x_center y_center 长轴 短轴 角度]
    load([dir,filename(1:end-4),'_ellipse_top_center.mat']);
    load([dir,filename(1:end-4),'_ellipse_bottom_center.mat']);
    ellipse_top = ellipse_2_abcdef(ellipse_top_center);
    ellipse_bottom = ellipse_2_abcdef(ellipse_bottom_center);
    
    
% 轮廓上的点
    load([dir,filename(1:end-4),'_xxx.mat']);
    load([dir,filename(1:end-4),'_yyy.mat']);
    contour_xy=[yyy' xxx'];
    contour_xy_sorted=sortrows(contour_xy,1);
    
    
    
    contour_image = xy_contour_to_image(contour_xy_sorted, Image_in);
    [contour_y, contour_x, hang, lie] = get_side_contour(dir, filename, contour_image, ellipse_top_center, ellipse_bottom_center);
    
    figure;scatter(contour_x,contour_y,'.');
    axis([0 lie 0 hang]);
    set(gca,'YDir','reverse');  
    grid on;
    
    
    contour_xy = [contour_y,contour_x];
    contour_xy_sorted=sortrows(contour_xy,1);
    
    
    
    %% 分段
    fenduan = [0 0 0 0 0 0];
    points_count = 0;
    fenduan_count = 0;
    while(true) 
        
        fenduandian=input(['请输入第' int2str(fenduan_count+1) '个分段点：']);
        
        for i = 1:length(contour_x)
            
            if(contour_xy_sorted(i,1) <= fenduandian)
                
                points_count=points_count+1;
            end
        end
        fenduan_count = fenduan_count + 1;
        fenduan(fenduan_count) = points_count;
        points_count = 0;
        
        shifoujixu=input(['是否继续（1/0）：']);
        if(shifoujixu==1)
            continue;
        else
            break;
        end
    end
    fenduan(fenduan_count+1) = length(contour_x);
    ks_all = [];
    xy_sampled_all= [];
    
    %% 拟合
    
    for i = 1:(fenduan_count+1)
        if( i == 1)
            num_start = 1;
        else
            num_start = fenduan(i-1)+1;
        end
        xy_data = [contour_xy_sorted( num_start:fenduan(i), 2), contour_xy_sorted( num_start:fenduan(i), 1)];
        nihefangshi=input(['拟合方式：直线/三次/五次：(1/3/5)：']);
        if (nihefangshi == 1)
            [ks, xy_sampled] = line_fitting(xy_data);
        elseif(nihefangshi == 3)
            [ks, xy_sampled]  = poly_fitting_3(xy_data);
        elseif(nihefangshi == 5)
            [ks, xy_sampled]  = poly_fitting_5(xy_data);
        end
        
        eval( ['ks_',int2str(i),'=ks']);
        eval( ['xy_sampled_',int2str(i),'=xy_sampled']);
        
        ks_all_tmp = ks_all;
        
        ks_all = [ks_all_tmp ks'];
        xy_sampled_all = [xy_sampled_all  xy_sampled]; 
        clear xydata ks xy_sampled ks_all_tmp;
          
    end
    
    ks_all = ks_all';
    
    all_in_one(dir,filename,ellipse_top, ellipse_bottom,Image_in, ks_all, xy_sampled_all);
    
    
    
    
    
    
end