# sor_reconstrcution
### 使用示例：
>>rec_sor('./data/example/');
请选择左右轮廓中的一条（1/2）：
>>1
请输入第1个分段点：438
是否继续（1/0）：1
请输入第2个分段点：471
是否继续（1/0）：1
请输入第3个分段点：488
是否继续（1/0）: 0
拟合方式：直线/三次/五次：(1/3/5)：1
拟合方式：直线/三次/五次：(1/3/5)：3
拟合方式：直线/三次/五次：(1/3/5)：1
拟合方式：直线/三次/五次：(1/3/5)：3

### 程序输入（所有输入放在同一文件夹下，并按照示例命名）：
 |类型 |文件名| 变量名|
 | ----- | ------ | ------ |
 | 回转体照片  |  *.jpg/png | --- |
 |上椭圆方程  |   *_ellipse_top_center.mat| ellipse_top_center|
 |下椭圆方程 |    *_ellipse_bottom_center.mat |ellipse_bottom_center|
 | 轮廓线点横坐标  |  *_xxx.mat  |  xxx  (在图像中的列坐标-行向量)|
 | 轮廓线点纵坐标  |  *_yyy.mat  |  yyy  (在图像中的行坐标-行向量)|
### 程序输出： 
        类型                            文件名                            变量名
        上椭圆中心点                 *_ellipse_top_center_xy.mat        *_ellipse_top_center_xy              
        下椭圆中心点                 *_ellipse_bottom_center_xy.mat     *_ellipse_bottom_center_xy
        生成曲线的成像               *_Meridian_points.mat              *_Meridian_points
        回转体模型                   *_model.stl    
### 说明：
- 所有输入输出值使用图像坐标系，即左上角为（0,0）
- 椭圆方程存储格式： [椭圆中心图像列坐标  椭圆中心图像行坐标 长轴 短轴 夹角]
- 上椭圆中心点存储格式：行向量，第一列为中心点图像列坐标，第二列为中心点图像行坐标
- 生成曲线的成像：3*n矩阵，第一行代表图像列坐标，第二行代表图像横坐标，第三行全为1。
        
