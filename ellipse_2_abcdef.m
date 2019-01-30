function [ellipse_abcdef] = ellipse_2_abcdef(ellipse)


ellipse_x_center=ellipse(1);
ellipse_y_center=ellipse(2);
ellipse_long_axis=ellipse(3);
ellipse_short_axis=ellipse(4);
ellipse_theta=ellipse(5);

ellipse_a = ellipse_long_axis^2*sin(ellipse_theta)^2+ellipse_short_axis^2*cos(ellipse_theta)^2;
ellipse_b = 2*(ellipse_long_axis^2-ellipse_short_axis^2)*sin(ellipse_theta)*cos(ellipse_theta);
ellipse_c = ellipse_long_axis^2*cos(ellipse_theta)^2+ellipse_short_axis^2*sin(ellipse_theta)^2;
ellipse_f = -ellipse_long_axis^2*ellipse_short_axis^2;

ellipse_d=-2*ellipse_a*ellipse_x_center-ellipse_b*ellipse_y_center;
ellipse_e=-2*ellipse_c*ellipse_y_center-ellipse_b*ellipse_x_center;
ellipse_f=ellipse_f+ellipse_a*ellipse_x_center^2+ellipse_b*ellipse_x_center*ellipse_y_center+ellipse_c*ellipse_y_center^2;

ellipse_abcdef = [ellipse_a;ellipse_b;ellipse_c;ellipse_d;ellipse_e;ellipse_f];

end