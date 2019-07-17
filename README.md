# 传统车道线检测  
Advanced_Lane_Detection项目的复现，更换了其中的数据，修改了相应脚本，添加了可视化功能。话不多说，直接上图显示效果。  
<div align=center><img width="640" height="480" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/result.png"/></div>

## 环境要求
* Python 3.5
* Numpy
* OpenCV-Python
* Matplotlib
* Pickle  
## 文件简介
* camera_cal是相机标定用的棋盘格图片
* example_images是用图片测试整套流程的输出图片
* test_images是测试图片
* calibrate_camera.py是相机标定脚本
* combined_thresh.py是二值化脚本
* get_example_images.py是测试流程脚本
* Line.py是定义输出类型脚本
* line_fit.py是多项式拟合车道线脚本
* line_fit_video.py是主程序
* perspective_transform.py是透视变换脚本
* region_of_interest.py是ROI操作脚本
## 项目流程
### 一：相机标定
采用棋盘格对相机进行标定，消除畸变，可采用下列方法（本项目采用第二种）
* 法一：执行calibrate_camera.py脚本获得内参矩阵与畸变系数存入p文件
* 法二：借助Matlab相机标定工具包，傻瓜式操作得到内参矩阵与畸变系数存入p文件
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/camera_cal/calibration1.jpg"/></div>  

### 二：ROI操作
设置mask，只保留车辆前方ROI区域，以便减小二值化处理的难度
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/mask.png"/></div>  

### 三：二值化操作
基于图像的梯度和颜色信息，进行二值化处理，阈值自行调整，初步定位车道线所在位置
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/example_images/binary2.png"/></div>

### 四：透视变换
运用透视变换，将二值图像转换为鸟瞰图，其中源点与目标点坐标的设置非常重要，以便后期使用多项式拟合进一步精准检测车道线
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/example_images/warped2.png"/></div>

### 五：多项式拟合
* 直方图统计，定位峰值为左右车道线搜寻基点
* 运用滑窗检测车道线，滑窗的数目，大小，面积设置可调
* 多项式方程 **ay^{2}+by+c=x** 不断得到新的搜寻基点
* 可视化操作
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/example_images/polyfit2.png"/></div>

### 六：计算曲率半径与车辆中心距车道中心偏移量
借助像素与米的换算关系，可求出曲率半径以及车辆中心距车道中心偏移量，进而给出车辆状态

### 七：可视化
在视频上方分别实时显示消除畸变的图片，鸟瞰图片，多项式拟合图片以及曲率半径，车辆中心距车道中心偏移量
<div align=center><img width="480" height="320" src="https://github.com/chiyukunpeng/Tradition_Lane_Detection/blob/master/result.png"/></div>

## 总结与展望
### 总结
* 传统图像处理算法检测车道线受外界因素影响较大，精度一般
* 这个项目的bug在于车道线检测偶尔不准，后期曲率半径的计算会失准，但车辆中心距车道中心的偏移量准确
* 透视变换对于车道线检测精度提升有很大帮助
### 展望
* 利用偏移量可进一步实现自动修正方位功能
* 透视变换对于遮挡这一因素的消除十分有效，可应用到其他场合

# 鸣谢
* **特别鸣谢视觉测控与智能导航实验室泽阳大神，附上大神[Github主页链接](https://github.com/namemzy?tab=repositories)**
* 鸣谢临港六狼团队所有成员以及张老板的大力指导
