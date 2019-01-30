function [ks_s,xy_sampled] = line_fitting(xy_data)

% width =768;
% height=1024;
width =1080;
height=1440;

scatter(xy_data(:,1),xy_data(:,2),'ok');
hold on


p=polyfit(xy_data(:,2),xy_data(:,1),1);

y_sampled = xy_data(:,2);
y_sampled = min(y_sampled):3:max(y_sampled);

if(y_sampled(end)~=max(xy_data(:,2)))
    y_sampled=[y_sampled,max(xy_data(:,2))];
end

x_sampled = polyval(p,y_sampled); 

plot(x_sampled,y_sampled,'m+');
hold on
xy_sampled = [x_sampled;y_sampled;ones(1,length(x_sampled))];

ks_1 = 1./p(1);
ks_s = zeros(length(xy_sampled),1);
tangent_lines = ones(3,length(xy_sampled));
for i = 1:length(xy_sampled)
    
    ks_s(i) = ks_1;  
    tangent_lines(1,i) = ks_s(i);
    tangent_lines(2,i) = -1;
    tangent_lines(3,i) = -1*ks_s(i)*xy_sampled(1,i)+xy_sampled(2,i);
    tangent_lines(:,i) = tangent_lines(:,i)/tangent_lines(3,i);
    
%     % 画出这些切线
%     b_l_tangent=-1*tangent_lines(3,i)/tangent_lines(2,i); 
%     k_l_tangent=ks_s(i);
%     x=0:1:width;
%     y=k_l_tangent*x+b_l_tangent;
%     plot(x,y,'--r');
%     hold on 
end

%% 显示调整 
set(gca,'ydir','reverse');
axis equal;
axis([0 width 0 height]);
