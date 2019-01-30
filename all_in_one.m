function [r,h]=all_in_one(data_dir, filename, ellipse_top, ellipse_bottom,sor_image, ks, xy_sampled)

%% 中间量保存完整，画图完整，未改善求椭圆与直线交点的方法，速度慢，

%% load data

%sor_image=imread([data_dir,'image.jpg']);


% load([data_dir,'ellipse_top.mat']);
% load([data_dir,'ellipse_bottom.mat']);
%load([data_dir,'xy_sampled.mat']);
%load([data_dir,'ks.mat']);

[height,width,chanel] = size(sor_image);

ellipse_top=ellipse_top/ellipse_top(6);
a=ellipse_top(1);
b=ellipse_top(2);
c=ellipse_top(3);
d=ellipse_top(4);
e=ellipse_top(5);
f=ellipse_top(6);

a_=ellipse_bottom(1);
b_=ellipse_bottom(2);
c_=ellipse_bottom(3);
d_=ellipse_bottom(4);
e_=ellipse_bottom(5);
f_=ellipse_bottom(6);

ellipse=[a,b/2,d/2; b/2,c,e/2;d/2,e/2,f];





%% 选择一个椭圆上的点
% 椭圆中心
x_ellipse_center = (b*e-2*c*d)/(4*a*c-b^2);
y_ellipse_center = (b*d-2*a*e)/(4*a*c-b^2);
x_ellipse_center_ = (b_*e_-2*c_*d_)/(4*a_*c_-b_^2);
y_ellipse_center_ = (b_*d_-2*a_*e_)/(4*a_*c_-b_^2);
% 椭圆长短半轴
q = 64*(f*(4*a*c-b*b)-a*e*e+b*d*e-c*d*d)/((4*a*c-b*b)^2);
s = 1/4*sqrt(abs(q)*sqrt(b*b+(a-c)^2));
r_max = 1/8*sqrt(2*abs(q)*sqrt(b*b+(a-c)^2)-2*q*(a+c));
r_min = sqrt(r_max^2-s^2);

q_ = 64*(f_*(4*a_*c_-b_^2)-a_*e_^2+b_*d_*e_-c_*d_^2)/((4*a_*c_-b_^2)^2);
s_ = 1/4*sqrt(abs(q_)*sqrt(b_^2+(a_-c_)^2));
r_max_ = 1/8*sqrt(2*abs(q_)*sqrt(b_^2+(a_-c_)^2)-2*q_*(a_+c_));
r_min_ = sqrt(r_max_^2-s_^2);

% 根据椭圆中心和长轴选择“任选点”
x_selected=x_ellipse_center+r_max*0.8;
y_selected=(-(b*x_selected+e)+sqrt((b*x_selected+e)^2-4*c*(a*x_selected^2+d*x_selected+f)))/2/c;
xy_selected = [x_selected,y_selected,1]';


%% 求椭圆交点，并确保x1_I ,x1_J为一对共轭复数
syms x;
syms y;
digits(10);
% ax^2 + bxy + cy^2 +dx + ey + f = 0
z1=ellipse_top(1)*x^2+ellipse_top(2)*x*y+ellipse_top(3)*y^2+ellipse_top(4)*x+ellipse_top(5)*y+ellipse_top(6);
z2=ellipse_bottom(1)*x^2+ellipse_bottom(2)*x*y+ellipse_bottom(3)*y^2+ellipse_bottom(4)*x+ellipse_bottom(5)*y+ellipse_bottom(6);

result=solve(z1,z2);
xx=result.x;
S_vpa_x = vpa(xx);
yy=result.y;
S_vpa_y = vpa(yy);
xxx=double(S_vpa_x);
yyy=double(S_vpa_y);
xk=[xxx,yyy];
x1_I=[xk(1,:),1]';
x1_J=[xk(2,:),1]';
x2_I=[xk(3,:),1]';
x2_J=[xk(4,:),1]';

if(real(x1_I) == real(x1_J))
    ;
elseif(real(x1_I) == real(x2_I))
    tmp = x1_J;
    x1_J=x2_I;
    x2_I = tmp;
elseif(real(x1_I) == real(x2_J))
    tmp = x1_J;
    x1_J=x2_J;
    x2_J = tmp;
end

if(real(x1_J) == real(x2_I))
    tmp = x1_I;
    x1_I=x2_I;
    x2_I = tmp;
elseif(real(x1_J) == real(x2_J))
    tmp = x1_I;
    x1_I=x2_J;
    x2_J = tmp;
end

if(real(x2_I) == real(x2_J))
    tmp = x1_I;
    x1_I=x2_I;
    x2_I = tmp;
    tmp = x1_J;
    x1_J=x2_J;
    x2_J = tmp;
end

%% l_infty
L_12 = cross(x1_I, x1_J);
L_34 = cross(x2_I, x2_J);
L_12 = L_12/L_12(3);
L_34 = L_34/L_34(3);
L12_infty = L_12;
L34_infty = L_34;

L_13 = cross(x1_I, x2_I);
L_24 = cross(x1_J, x2_J);

L_14 = cross(x1_I, x2_J);
L_23 = cross(x1_J, x2_I);


Y_limited=[y_ellipse_center_-r_min_, y_ellipse_center-r_min];
if(((-1*L12_infty(1)/L12_infty(2))*x_ellipse_center-1*L12_infty(3)/L12_infty(2))<Y_limited(2))
    L_infty = L12_infty;
    I = x1_I;
    J = x1_J;
elseif(((-1*L34_infty(1)/L34_infty(2))*x_ellipse_center-1*L34_infty(3)/L34_infty(2))<Y_limited(2))
    L_infty = L34_infty;
    I = x2_I;
    J = x2_J;
elseif(((-1*L12_infty(1)/L12_infty(2))*x_ellipse_center_-1*L12_infty(3)/L12_infty(2))>Y_limited(1))
    L_infty = L12_infty;
    I = x1_I;
    J = x1_J;
elseif(((-1*L34_infty(1)/L34_infty(2))*x_ellipse_center_-1*L34_infty(3)/L34_infty(2))>Y_limited(1))
    L_infty = L34_infty;
    I = x2_I;
    J = x2_J;
end


%% 选择无穷远线并确定虚圆点
% L_infty = L12_infty;
% I = x1_I;
% J = x1_J;
% L_infty = L34_infty;
% I = x2_I;
% J = x2_J;

%% v无穷
%v_infty = L12 x L34
v_infty =cross(L_12, L_34);
v_infty = v_infty/v_infty(3);

%% l_s
%l_s = (L_13 x L_24) x (L_14 x L_23)
l_s = cross(cross(L_13, L_24),cross(L_14, L_23));
l_s = l_s/l_s(3);

%% 求解绝对圆锥曲线IAC
a1 = I(1);
b1 = I(2);
a2 = J(1);
b2 = J(2);
A = v_infty(1);
B = v_infty(2);
as = l_s(1);
bs = l_s(2);

% 构造齐次方程
C=[  a1^2+b1^2, 2*a1,   2*b1,     1;
     a2^2+b2^2, 2*a2,   2*b2,     1;
     -B,        bs*A,   B*bs-1, bs; 
     A,         1-as*A, -as*B,    -as;
     as*B-bs*A, -bs,    as,       0];
% 奇异值分解
[U S V] = svd(C);
% 绝对圆锥曲线omega
omega = [V(1,4), 0, V(2,4); 0, V(1,4), V(3,4); V(2,4), V(3,4), V(4,4)];
%save([data_dir,'omega'],'omega');

%% 选择一个椭圆上的点
% 上椭圆中心
x_c = (b*e-2*c*d)/(4*a*c-b^2);
y_c = (b*d-2*a*e)/(4*a*c-b^2);
plot(x_c,y_c,'*');
hold on
% 椭圆长短半轴
q = 64*(f*(4*a*c-b*b)-a*e*e+b*d*e-c*d*d)/((4*a*c-b*b)^2);
s = 1/4*sqrt(abs(q)*sqrt(b*b+(a-c)^2));
r_max = 1/8*sqrt(2*abs(q)*sqrt(b*b+(a-c)^2)-2*q*(a+c));
r_min = sqrt(r_max^2-s^2);
% 根据椭圆中心和长轴选择“任选点”
point_x =x_c+r_max*0.8;
syms y
s=solve(ellipse_top(1)*point_x^2+ellipse_top(2)*point_x*y+ellipse_top(3)*y^2+ellipse_top(4)*point_x+ellipse_top(5)*y+ellipse_top(6)==0,y);
xy_selected_y = eval(s);
xy_selected = [point_x,xy_selected_y(2),1]';
%xy_sampled =[xy_selected,xy_sampled];

%% 求切线、与无穷远交点、过无穷远向椭圆的切线和切点、W矩阵、生成曲线
tangent_lines = ones(3,length(xy_sampled));
intersections_with_l_infty = zeros(3,length(xy_sampled));
polar_lines = zeros(3,length(xy_sampled));
intersections_with_ellipse= zeros(3,length(xy_sampled));
mu_ws = zeros(length(xy_sampled),1);
V_ws = zeros(3,length(xy_sampled));
W_s = zeros(3,3,length(xy_sampled));
Meridian_points = zeros(3,length(xy_sampled));

for i = 1:length(xy_sampled)
    % 切线
    tangent_lines(1,i) = ks(i);
    tangent_lines(2,i) = -1;
    tangent_lines(3,i) = -1*ks(i)*xy_sampled(1,i)+xy_sampled(2,i);
    tangent_lines(:,i) = tangent_lines(:,i)/tangent_lines(3,i);
    % 切线与无穷交点
    intersections_with_l_infty(:,i) = cross(tangent_lines(:,i), L_infty);
    intersections_with_l_infty(:,i) = intersections_with_l_infty(:,i)/intersections_with_l_infty(3,i);
    % 极线
    polar_lines(:,i) = ellipse*intersections_with_l_infty(:,i);
    polar_lines(:,i) = polar_lines(:,i)/polar_lines(3,i);
    % 极线与椭圆交点
    syms x y
    s=solve(a*x^2+b*x*y+c*y^2+d*x+e*y+f==0,polar_lines(1,i)*x+polar_lines(2,i)*y+polar_lines(3,i)==0,x,y);
    X=double(s.x);
    Y=double(s.y); 
    if(length(X)==0)
        continue;
    elseif(length(X)==1)
        ellipse_tangent_point1 = [X(1),Y(1), 1];
        ellipse_tangent_point2 = [X(1),Y(1), 1];
    else
        if(real(X(1)) <= real(X(2)))
            ellipse_tangent_point1 = [X(1),Y(1), 1];
            ellipse_tangent_point2 = [X(2),Y(2), 1];
        else
            ellipse_tangent_point1 = [X(2),Y(2), 1];
            ellipse_tangent_point2 = [X(1),Y(1), 1];
        end
            
    end
    
    % 选择与椭圆的交点
    intersections_with_ellipse(:,i) = ellipse_tangent_point1;
    %intersections_with_ellipse(:,i) = ellipse_tangent_point2;

    % 求W
    tmp = cross(intersections_with_ellipse(:,i),xy_sampled(:,i)');
    tmp = tmp/tmp(3);
    V_ws(:,i) = cross(tmp,l_s);
    V_ws(:,i) = V_ws(:,i)/V_ws(3,i);
    w_infty = cross(tmp,L_infty);
    w_infty = w_infty/w_infty(3);

    AC = sqrt((V_ws(1,i)-intersections_with_ellipse(1,i))^2 +(V_ws(2,i)-intersections_with_ellipse(2,i))^2);
    DB = sqrt((w_infty(1)-xy_sampled(1,i))^2 +(w_infty(2)-xy_sampled(2,i))^2);
    CB = sqrt((intersections_with_ellipse(1,i)-w_infty(1))^2 +(intersections_with_ellipse(2,i)-w_infty(2))^2); 
    AD = sqrt((V_ws(1,i)-xy_sampled(1,i))^2 +(V_ws(2,i)-xy_sampled(2,i))^2); 
    mu_ws(i) = AC*DB/CB/AD;
    
    W = eye(3)+(mu_ws(i)-1)*( (V_ws(:,i)*L_infty')/(V_ws(:,i)'*L_infty));
    W_s(:,:,i) = W;
    Meridian_points(:,i) = W_s(:,:,i)*xy_selected;
    Meridian_points(:,i) = Meridian_points(:,i)/Meridian_points(3,i);    
    
    eval ( [filename(1:end-4),'_Meridian_points=Meridian_points' ]);
    save([data_dir,filename(1:end-4), '_Meridian_points'],[filename(1:end-4), '_Meridian_points']);
    
    
    
%     % 画出这些切线
%     b_l_tangent=-1*tangent_lines(3,i)/tangent_lines(2,i); 
%     k_l_tangent=ks(i);
%     x=0:50:img_width;
%     y=k_l_tangent*x+b_l_tangent;
%     plot(x,y,'--r');
%     hold on     

%     % 画出极线
%     k_pole_polar=-polar_lines(1,i)/polar_lines(2,i);
%     b_pole_polar=-polar_lines(3,i)/polar_lines(2,i); 
%     x=0:0.1:img_width;
%     y = k_pole_polar*x+b_pole_polar;
%     plot(x,y,'c');

%      % 画出与椭圆的切点
%      plot(intersections_with_ellipse(1,i),intersections_with_ellipse(2,i), 'ko');

%     % 画出对应点连线
%     if(Meridian_points(1,i)~=0 && Meridian_points(2,i) ~= 0 )
%         plot([contour_xy_sampled(i,1),Meridian_points(1,i)], [contour_xy_sampled(i,2),Meridian_points(2,i)]);
%     end
end


%% x无穷、v无穷、m无穷
x_infty = cross(cross(xy_selected,(ellipse^-1)*L_infty),L_infty);
x_infty = x_infty/x_infty(3);
vt_infty = (omega^-1)*L_infty;
vt_infty = vt_infty/vt_infty(3);
m_infty = cross(x_infty,vt_infty);
m_infty = m_infty/m_infty(3);

%% m无穷与IAC的交点，即I、J点
iac_a = omega(1,1);
iac_b = 2*omega(1,2);
iac_c = omega(2,2);
iac_d = 2*omega(1,3);
iac_e = 2*omega(2,3);
iac_f = omega(3,3);

syms x y
s=solve(iac_a*x^2+iac_b*x*y+iac_c*y^2+iac_d*x+iac_e*y+iac_f==0,m_infty(1)*x+m_infty(2)*y+m_infty(3)==0,x,y);
X=double(s.x);
Y=double(s.y);
intersection_p1 = [X(1),Y(1), 1]'
intersection_p2 = [X(2),Y(2), 1]'

%% 求解 alpha beta
intersection_p1 = intersection_p1/intersection_p1(2);
intersection_p2 = intersection_p2/intersection_p2(2);
alpha = real(intersection_p1(1));
beta = abs(imag(intersection_p1(1)));

%% 矫正矩阵
Mr = [1.0/beta,   -alpha*(1.0/beta), 0;
      0,          1,                 0;
      m_infty(1), m_infty(2),        1];
  
Mr_1_T = (inv(Mr))';

%% 点修正
Meridian_points_rectified = Mr*Meridian_points;
for index=1:length(Meridian_points_rectified)
    Meridian_points_rectified(:,index) = Meridian_points_rectified(:,index)/Meridian_points_rectified(3,index);
end






%% 对称轴修正
l_z = Mr_1_T*l_s;
l_z = l_z/l_z(3);
% lz
b_lz=-1/l_z(2); 
k_lz=-1*l_z(1)/l_z(2);

%% 半径及其对应高度
intersections = zeros(2,length(Meridian_points_rectified));
h_increase = zeros(length(Meridian_points_rectified),1);
r = zeros(length(Meridian_points_rectified),1);
sum_ = 0;
h=zeros(1,length(r));
for index=1:length(Meridian_points_rectified)
    % 垂线
    intersection_points_tmp = cross(l_z,[-1/k_lz,-1,(1/k_lz*Meridian_points_rectified(1,index)+Meridian_points_rectified(2,index))]');
    intersection_points_tmp=intersection_points_tmp/intersection_points_tmp(3);
    % 垂足
    intersections(1,index)= intersection_points_tmp(1);
    intersections(2,index)= intersection_points_tmp(2);
    % 半径及相应的高度
    r(index) = real(sqrt((intersection_points_tmp(1)-Meridian_points_rectified(1,index))^2 + (intersection_points_tmp(2)-Meridian_points_rectified(2,index))^2));
    if(index==1)
        h_increase(index)= 0;
    else
        h_increase(index)=sqrt((intersection_points_tmp(1)-intersections(1,index-1))^2 + (intersection_points_tmp(2)-intersections(2,index-1))^2);
    end
    sum_ = sum_+h_increase(index);
    h(index) = sum_;
end
r=real(r);
h=real(h);

%% 画图
figure(1);
% l_infty
b_l_infty=-1*L_infty(3)/L_infty(2); 
k_l_infty=-1*L_infty(1)/L_infty(2);
x_l_infty=0:0.1:width;
y_l_infty = k_l_infty*x_l_infty+b_l_infty;

% l_s
b_ls=-1/l_s(2); 
k_ls=-1*l_s(1)/l_s(2);
x_l_s=0:0.1:width;
y_l_s = k_ls*x_l_s+b_ls;

% m无穷
b_m_infty=-1*m_infty(3)/m_infty(2); 
k_m_infty=-1*m_infty(1)/m_infty(2);
x_m_infty=0:0.1:width;
y_m_infty = k_m_infty*x_m_infty+b_m_infty;
% l_z
x_lz=0:0.1:width;
y_lz = k_lz*x_lz+b_lz;

% 画椭圆
h1=ezplot(z1,[0,width,0,height]);
set(h1,'Color','m');
hold on;
h2=ezplot(z2,[0,width,0,height]);
set(h2,'Color','b');
hold on;
plot(xy_sampled(1,1:3:end),xy_sampled(2,1:3:end),'g*', ...
    x_l_infty,y_l_infty,'r', ...
    x_l_s,y_l_s,'--g', ...
    xy_selected(1), xy_selected(2), 'rd', ...
    intersections_with_l_infty(1,1:3:end),intersections_with_l_infty(2,1:3:end), 'k+', ...
    Meridian_points(1,1:3:end), Meridian_points(2,1:3:end),'co', ...
    x_m_infty,y_m_infty,'m', ...
    Meridian_points_rectified(1,1:3:end),Meridian_points_rectified(2,1:3:end),'rp', ...
    x_lz,y_lz,'r--', ...
    intersections(1,1:3:end),intersections(2,1:3:end),'xk', ...
    [Meridian_points_rectified(1,10),intersections(1,10)],[Meridian_points_rectified(2,10),intersections(2,10)],':.k');
axis([0 width 0 height]);
set(gca,'ydir','reverse');
text((Meridian_points_rectified(1,10)+intersections(1,10))/2-10,(Meridian_points_rectified(2,10)+intersections(2,10))/2+5,'半径');
legend('椭圆1','椭圆2','采样点','无穷远线','成像后对称轴','任选点','与无穷远交点','成像后生成曲线','m无穷','修正后生成曲线','修正后对称轴','垂足');

%% 显示求得的子母线
figure(2);
title('子母线及旋转轴');
plot(r,h,'ob');
hold on;
% 旋转轴
line_rotate=line([0,0],[0,1000]);
set(line_rotate,'color','r');
hold on;
set(gca,'ydir','reverse');
axis equal;

%% 画出归一化三维模型
% 插值
xi=min(h):(max(h)/40):max(h);
yi=interp1(h,r,xi,'pchip');
figure(2);
plot(yi,xi);
set(gca,'ydir','reverse');
axis equal;
% 归一化
scale=max(xi);
xi=xi/scale;
yi=yi/scale;

figure(3);
alpha = linspace(0,2*pi,length(xi));
X = yi'*sin(alpha);
Y = yi'*cos(alpha);
Z = xi'*ones(1,length(xi));
mesh(X,Y,Z);
title('重建模型');
set(gca,'zdir','reverse');
hold on;
axis equal



surf2stl([data_dir, filename(1:end-4), '_model.stl'],X,Y,-Z);


%% 保存数据
% save([data_dir,'x1_I'],'x1_I');
% save([data_dir,'x1_J'],'x1_J');
% save([data_dir,'x2_I'],'x2_I');
% save([data_dir,'x2_J'],'x2_J');

% save([data_dir,'l_s'],'l_s');
% save([data_dir,'v_infty'],'v_infty');
% save([data_dir,'L12_infty'],'L12_infty');
% save([data_dir,'L34_infty'],'L34_infty');
% save([data_dir,'L_infty'],'L_infty');
% save([data_dir,'xy_selected'],'xy_selected');
% save([data_dir,'r'],'r');
% save([data_dir,'h_increase'],'h_increase');
% save([data_dir,'h'],'h');


% save([data_dir,'X'],'X');
% save([data_dir,'Y'],'Y');
% save([data_dir,'Z'],'Z');


