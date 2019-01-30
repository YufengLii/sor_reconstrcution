function [ks_1,xy_sampled] = poly_fitting_3(xy_data)
% 
% width =768;
% height=1024;
width =1080;
height=1440;

% xy_data =xy_data';
scatter(xy_data(:,1),xy_data(:,2),'ok');
hold on

fitting_curve = fit(xy_data(:,2),xy_data(:,1),'poly3');
plot(fitting_curve,xy_data(:,2),xy_data(:,1));
hold on;

%% 设置采样间隔，并默认设置采样点包含第一个点，不包含尾点，
y_sampled = xy_data(:,2);
y_sampled = min(y_sampled):4:max(y_sampled);
x_sampled = fitting_curve(y_sampled);
% if(y_sampled(end)~=xy_data(end,2))
%     y_sampled=[y_sampled,xy_data(end,2)];
%     x_sampled=[x_sampled,fitting_curve(xy_data(end,2))]
% end

plot(x_sampled,y_sampled,'m+');
hold on;
xy_sampled = [x_sampled';y_sampled;ones(1,length(x_sampled))];
ks= differentiate(fitting_curve,y_sampled);
ks_1 = 1./ks;

tangent_lines = ones(3,length(xy_sampled));
for i = 1:length(xy_sampled)
    % 切线
    tangent_lines(1,i) = ks_1(i);
    tangent_lines(2,i) = -1;
    tangent_lines(3,i) = -1*ks_1(i)*xy_sampled(1,i)+xy_sampled(2,i);
    tangent_lines(:,i) = tangent_lines(:,i)/tangent_lines(3,i);
%     % 画出这些切线
%     b_l_tangent=-1*tangent_lines(3,i)/tangent_lines(2,i); 
%     k_l_tangent=ks_1(i);
%     x=0:1:width;
%     y=k_l_tangent*x+b_l_tangent;
%     plot(x,y,'--r');
%     hold on     
end

%% 显示调整 
set(gca,'ydir','reverse');
axis equal;
axis([0 width 0 height]);