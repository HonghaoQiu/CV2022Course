#include <iostream>   
#include<fstream>
#include<string.h>
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/imgproc/types_c.h>
#include <eigen/Eigen/Dense>


using namespace Eigen;// 省去函数前面加Eigen::
using namespace std;  //省去屏幕输出函数cout前的std::
using namespace cv;   // 省去函数前面加cv::的必要性

#define DataPath "E:\\Study\\大二下\\计算机视觉与模式识别\\CV2022Course\\Data\\" 

struct MyPoint
{
	int ID; //点号
	double x,y,z; //坐标
};


void DrawCross(Mat& img, Point2f point, Scalar color, int size, int thickness = 1)
{
	//绘制横线  
	line(img, cvPoint(point.x - size / 2, point.y), cvPoint(point.x + size / 2, point.y), color, thickness, 16, 0);
	//绘制竖线  	
	line(img, cvPoint(point.x, point.y - size / 2), cvPoint(point.x, point.y + size / 2), color, thickness, 16, 0);
	return;
}

//二值化 判断分析法
Mat OSTU(Mat img, string dataname)
{
	cvtColor(img, img, COLOR_BGR2GRAY);//转换灰度图像
	GaussianBlur(img, img, Size(9, 9), 2, 2);
	int nThreshold = 0;//定义域值
	int nHeight = img.rows;
	int nWidth = img.cols;

	//定义并初始化灰度统计数组
	int hist[256];
	memset(hist, 0, sizeof(hist));

	//PS,PS1,PS2分别表示像素总数，类1像素，类2像素
	int PS = 0, PS1 = 0, PS2 = 0;
	//IP,IP1分别表示总的质量矩，和类1质量矩
	double IP = 0.0, IP1 = 0.0;
	//meanvalue1,meanvalue2分别代表类1灰度均值，类2灰度均值
	double meanvalue1 = 0.0, meanvalue2 = 0.0;
	//Dispersion1,Dispersion2,classinDis,classoutDis表示类1方差，类2方差，类内方差，类间方差，max表示类间最大差距
	double Dispersion1 = 0.0, Dispersion2 = 0.0, classinDis = 0.0, classoutDis = 0.0, max = 0.0;
	int graymax = 255, graymin = 0;

	for (int i = 0; i < nHeight; i++)
		for (int j = 0; j < nWidth; j++)
		{
			int gray = img.at<uchar>(i, j);
			hist[gray]++;
			if (gray > graymax) graymax = gray;
			if (gray < graymin) graymin = gray;
		}

	if (graymin == 0) graymin++;

	//计算总的质量矩IP和像素总和PS
	for (int k = graymin; k <= graymax; k++)
	{
		IP += (double)k * (double)hist[k];
		PS += hist[k];
	}

	double ratio = 0.0;
	//求阈值
	for (int k = graymin; k <= graymax; k++)
	{
		//计算类1像素总数
		PS1 += hist[k];
		if (!PS1) continue;
		//计算类2像素总数
		PS2 = PS - PS1;
		if (PS2 == 0) break;
		//计算类1质量矩
		IP1 += (double)k * hist[k];
		//计算类1均值
		meanvalue1 = IP1 / PS1;
		//计算类1间方差
		for (int n = graymin; n <= k; n++)
			Dispersion1 += ((n - meanvalue1) * (n - meanvalue1) * hist[n]);
		//计算类2均值
		meanvalue2 = (IP - IP1) / PS2;
		//计算类2间方差
		for (int m = k + 1; m <= graymax; m++)
			Dispersion2 += ((m - meanvalue2) * (m - meanvalue2) * hist[m]);
		//计算类内方差
		classinDis = Dispersion1 + Dispersion2;
		//计算类间方差
		classoutDis = (double)PS1 * (double)PS2 * (meanvalue1 - meanvalue2) * (meanvalue1 - meanvalue2);
		//计算类间方差和类内方差比值
		if (classinDis != 0)
			ratio = classoutDis / classinDis;
		//获取阈值
		if (ratio > max)
		{
			max = ratio;
			nThreshold = k;
		}

	}

	//二值化图像
	Mat imgBinary;
	imgBinary.create(nHeight, nWidth, CV_8UC1);//CV_8UC1 8为无符号整型 单通道
	for (int i = 0; i < nHeight; i++)
		for (int j = 0; j < nWidth; j++)
		{
			if (img.at<uchar>(i, j) < nThreshold)
				imgBinary.at<uchar>(i, j) = 0;
			else imgBinary.at<uchar>(i, j) = 255;
		}



	imwrite("C:\\Users\\HonghaoQiu\\Desktop\\判断分析法二值化图像_" + dataname + ".bmp", imgBinary);

	return imgBinary;
}

//半自动标定
MyPoint* SemiAutomaticCalibration(string& dataname , MyPoint* Control_Points)
{
	Mat Oringinimag = imread(DataPath + dataname, IMREAD_COLOR);   // 读入图片 
	int RoL = 0;
	if ((dataname.compare("Right.bmp")) == 0)
		RoL = 0;
	else if ((dataname.compare("Left.bmp")) == 0)
		RoL = 1;
	if (Oringinimag.empty())     // 判断文件是否正常打开  
	{
		cout << " FAIL";
		waitKey(6000);  // 等待6000 ms后窗口自动关闭   
		return 0;
	}

	Mat otsu = OSTU(Oringinimag, dataname);
	Mat canny = otsu.clone();
	Canny(otsu, canny, 50, 100);//边缘检测
	imwrite("C:\\Users\\HonghaoQiu\\Desktop\\边缘检测图像_" + dataname + ".bmp", canny);

	Mat fc = canny.clone();


	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	 if (RoL == 0)
		 findContours(fc, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//查找出所有的圆边界
	 if (RoL == 1)
		 findContours(fc, contours, hierarchy, CV_RETR_LIST, CV_RETR_LIST);//查找出所有的圆边界
	
	//int nHeight = fc.rows;
	//int nWidth = fc.cols;
	//Mat d;
	//d.create(nHeight, nWidth, CV_8UC1);//CV_8UC1 8为无符号整型 单通道
	//for (int index = 0; index >= 0; index = hierarchy[index][0])
	//{
	//	Scalar color(255, 105, 180);
	//	drawContours(d, contours, index, color,1, 8, hierarchy);
	//}


	//imwrite("C:\\Users\\HonghaoQiu\\Desktop\\findContours_" + dataname + ".bmp", d);



	Mat c = Oringinimag.clone();
	Mat pointsf;
	RotatedRect* box= new RotatedRect[60];
	/*Mat(contours[22]).convertTo(pointsf, CV_32F);
	RotatedRect box = fitEllipse(pointsf);
	DrawCross(c, box.center, Scalar(255, 105, 180), 30, 2);
	putText(c, to_string(0), box.center, FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 100), 2);*/

	if (RoL == 0) {
		
		for (int i = 0; i <= 49; i++)

		{
			Mat(contours[i]).convertTo(pointsf, CV_32F);
			box[i] = fitEllipse(pointsf);

			////人工选取图像点
			//DrawCross(c, box[i].center, Scalar(255, 105, 180), 30, 2);
			//putText(c, to_string(i), box[i].center, FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 100), 2);
		}
	}
	
	if (RoL == 1)
	{
		
		for (int i =2; i <= 112; i=i+2)

		{
			if (i == 32) i = 34;
			if (i == 62) i++;
		
			Mat(contours[i]).convertTo(pointsf, CV_32F);
			int d = i / 2;
			box[d] = fitEllipse(pointsf);

			//人工选取图像点
			DrawCross(c, box[d].center, Scalar(255, 105, 180), 30, 2);
			putText(c, to_string(d), box[d].center, FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 100), 2);
		}
	}

	
	
	double XandY[4][2];
	double xandy[4][2];
	
	switch (RoL)
	{
	case 0:
		//右图0-25,14-7,32-47，54-48
	  //空间坐标
		XandY[0][0] = Control_Points[0].x; XandY[0][1] = Control_Points[0].y;
		XandY[1][0] = Control_Points[14].x; XandY[1][1] = Control_Points[14].y;
		XandY[2][0] = Control_Points[32].x; XandY[2][1] = Control_Points[32].y;
		XandY[3][0] = Control_Points[54].x; XandY[3][1] = Control_Points[54].y;
	
		//图像坐标
		xandy[0][0] = box[25].center.x;	xandy[0][1] = box[25].center.y;
		xandy[1][0] = box[7].center.x;	xandy[1][1] = box[7].center.y;
		xandy[2][0] = box[47].center.x;	xandy[2][1] = box[47].center.y;
		xandy[3][0] = box[48].center.x;	xandy[3][1] = box[48].center.y;
		

		break;

	case 1:
		//左图56-27,8-1,26-36，50-47
	  //空间坐标
		XandY[0][0] = Control_Points[56].x; XandY[0][1] = Control_Points[56].y;
		XandY[1][0] = Control_Points[8].x; XandY[1][1] = Control_Points[8].y;
		XandY[2][0] = Control_Points[26].x; XandY[2][1] = Control_Points[26].y;
		XandY[3][0] = Control_Points[50].x; XandY[3][1] = Control_Points[50].y;

		//图像坐标
		xandy[0][0] = box[27].center.x;	xandy[0][1] = box[27].center.y;
		xandy[1][0] = box[1].center.x;	xandy[1][1] = box[1].center.y;
		xandy[2][0] = box[36].center.x;	xandy[2][1] = box[36].center.y;
		xandy[3][0] = box[47].center.x;	xandy[3][1] = box[47].center.y;
       break;
	}

	//初始化系数矩阵
	double matrix[8][9];//射影变换 矩阵
	int n = 0;//行计数器
	for (int i = 0; i < 4; i++)//矩阵初始化
	{
		//初始化基数行
		matrix[n][0] = -XandY[i][0];
		matrix[n][1] = -XandY[i][1];
		matrix[n][2] = -1;
		matrix[n][3] = 0;
		matrix[n][4] = 0;
		matrix[n][5] = 0;
		matrix[n][6] = xandy[i][0] * XandY[i][0];
		matrix[n][7] = xandy[i][0] * XandY[i][1];
		matrix[n][8] = -xandy[i][0];
		n++;//换行
		//初始化偶数行
		matrix[n][0] = 0;
		matrix[n][1] = 0;
		matrix[n][2] = 0;
		matrix[n][3] = -XandY[i][0];
		matrix[n][4] = -XandY[i][1];
		matrix[n][5] = -1;
		matrix[n][6] = xandy[i][1] * XandY[i][0];
		matrix[n][7] = xandy[i][1] * XandY[i][1];
		matrix[n][8] = -xandy[i][1];
		n++;//换行
	}

	//解方程
	int n1, m1;
	n1 = 8; m1 = 9;//矩阵大小 [ 8 * 9 ]
	for (int j = 0; j < n1; j++)
	{
		double max = 0;
		double imax = 0;
		for (int i = j; i < n1; i++)
			if (imax < fabs(matrix[i][j]))
			{
				imax = fabs(matrix[i][j]);
				max = matrix[i][j];//得到各行中所在列最大元素
				m1 = i;
			}
		if (fabs(matrix[j][j]) != max)
		{
			double b = 0;
			for (int k = j; k < n1 + 1; k++)
			{
				b = matrix[j][k];
				matrix[j][k] = matrix[m1][k];
				matrix[m1][k] = b;
			}
		}
		for (int r = j; r < n1 + 1; r++) //让该行的所在列除以所在列的第一个元素，目的是让首元素为1
			matrix[j][r] = matrix[j][r] / max;
		for (int i = j + 1; i < n1; i++)
		{
			double c = matrix[i][j];
			if (c == 0)
				continue;
			for (int s = j; s < n1 + 1; s++)//前后行数相减，使下一行或者上一行的首元素为0
			{
				double tempdata = matrix[i][s];
				matrix[i][s] = matrix[i][s] - matrix[j][s] * c;
			}
		}
	}
	for (int i = n1 - 2; i >= 0; i--)
		for (int j = i + 1; j < n1; j++)
		{
			double tempData = matrix[i][j];
			double data1 = matrix[i][n1];
			double data2 = matrix[j][n1];
			matrix[i][n1] = matrix[i][n1] - matrix[j][n1] * matrix[i][j];
		}

	MyPoint* Points = new MyPoint[60];//存下图像点坐标
	//逆推
	//空间坐标->图像 标记+
	for (int i = 0; i < 60; i++)
	{
		Point P;
		Points->ID = i;
		double temp1, temp2;
		temp1 = (matrix[0][8] * Control_Points[i].x + matrix[1][8] * Control_Points[i].y + matrix[2][8]);
		temp2 = (matrix[6][8] * Control_Points[i].x + matrix[7][8] * Control_Points[i].y + 1);
		P.x = temp1 / temp2;
		Points[i].x = temp1 / temp2;
		temp1 = (matrix[3][8] * Control_Points[i].x + matrix[4][8] * Control_Points[i].y + matrix[5][8]);
		temp2 = (matrix[6][8] * Control_Points[i].x + matrix[7][8] * Control_Points[i].y + 1);
		P.y = temp1 / temp2;
		Points[i].y = temp1 / temp2;
		Points[i].z = Control_Points[i].z;
		//画反算点
		DrawCross(c, P, Scalar(255, 105, 180), 30, 2);
		P.x += 5; P.y -= 10;//错位显示文字
		putText(c, to_string(i), P, FONT_HERSHEY_SIMPLEX, 2, CV_RGB(255, 105, 180), 2);
	}
	
	imwrite("C:\\Users\\HonghaoQiu\\Desktop\\标点_" + dataname + ".bmp", c);

	return Points;
}



void ExternalParameters(MyPoint* ControlPoints, MyPoint* PicturePoints, string& dataname) {
	int RoL = 0;
	if ((dataname.compare("Right.bmp")) == 0)
		RoL = 0;
	else if ((dataname.compare("Left.bmp")) == 0)
		RoL = 1;
	
	
	//设置内部参数(单位：mm)
	const double x0 = 1935.5000000000;
	const double y0 = 1295.5000000000;
	const double f = 7935.786962;
	const double width=3872;
	const double height = 2592;

	//设置外部参数初值(单位：mm)
	double Xs, Ys, Zs, phi, omega, kappa;
	if (RoL == 1) {
		Xs = 350.0; Ys = 520.0; Zs = 300.0;
		phi = -0.9209585845351669; omega = -0.8780690992255569; kappa = 2.1102441253730908;
	}
	else if (RoL == 0) {
		Xs = 130.0; Ys = 610.0; Zs = 300.0;
		phi = -0.4545695618308865; omega = -1.1219147656947568; kappa = 2.6531277188826015;
	}
	else {
		cout << "error:The image doesn't exist!" << endl;
		return;
	}

	//设计计算参数
	double R[10];//旋转矩阵
	Matrix<double, 8, 6> A;//A矩阵
	Matrix<double, 8, 1> L;//L向量
	Matrix<double, 6, 1> result;//结果

	//迭代循环计算
	double Precision = 0.0001;//精度
	int count_cycles = 0;//统计循环次数
	do {
		count_cycles++;
		//旋转矩阵R
		R[1] = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa);
		R[2] = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa);
		R[3] = -sin(phi) * cos(omega);
		R[4] = cos(omega) * sin(kappa);
		R[5] = cos(omega) * cos(kappa);
		R[6] = -sin(omega);
		R[7] = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa);
		R[8] = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa);
		R[9] = cos(phi) * cos(omega);

		for (int i = 0; i < 4; i++) {
			double x = R[1] * (ControlPoints[13 * i].x - Xs) + R[4] * (ControlPoints[13 * i].y - Ys) + R[7] * (ControlPoints[13 * i].z - Zs);
			double y = R[2] * (ControlPoints[13 * i].x - Xs) + R[5] * (ControlPoints[13 * i].y - Ys) + R[8] * (ControlPoints[13 * i].z - Zs);
			double z = R[3] * (ControlPoints[13 * i].x - Xs) + R[6] * (ControlPoints[13 * i].y - Ys) + R[9] * (ControlPoints[13 * i].z - Zs);

			float fX_Z = -f * x / z;
			float fY_Z = -f * y / z;

			L[2 * i] = PicturePoints[13 * i].x - (-f * x / z + x0);
			L[2 * i + 1] = (height - PicturePoints[13 * i].y) - (-f * y / z + y0);

			A(2 * i, 0) = 1 / z * (R[1] * f + R[3] * fX_Z);
			A(2 * i, 1) = 1 / z * (R[4] * f + R[6] * fX_Z);
			A(2 * i, 2) = 1 / z * (R[7] * f + R[9] * fX_Z);
			A(2 * i, 3) = fY_Z * sin(omega) -
				(fX_Z / f * (fX_Z * cos(kappa) - fY_Z * sin(kappa)) + f * cos(kappa)) * cos(omega);
			A(2 * i, 4) = -f * sin(kappa) - fX_Z / f * (fX_Z * sin(kappa) + fY_Z * cos(kappa));
			A(2 * i, 5) = fY_Z;

			A(2 * i + 1, 0) = 1 / z * (R[2] * f + R[3] * fX_Z);
			A(2 * i + 1, 1) = 1 / z * (R[5] * f + R[6] * fX_Z);
			A(2 * i + 1, 2) = 1 / z * (R[8] * f + R[9] * fX_Z);
			A(2 * i + 1, 3) = -fX_Z * sin(omega) -
				(fY_Z / f * (fX_Z * cos(kappa) - fY_Z * sin(kappa)) - f * sin(kappa)) * cos(omega);
			A(2 * i + 1, 4) = -f * cos(kappa) - fY_Z / f * (fX_Z * sin(kappa) + fY_Z * cos(kappa));
			A(2 * i + 1, 5) = -fX_Z;
		}

		result = ((A.transpose() * A).inverse()) * (A.transpose()) * L;

		Xs += result[0];
		Ys += result[1];
		Zs += result[2];
		phi += result[3];
		omega += result[4];
		kappa += result[5];
	} while (fabs(result[0]) >= Precision || fabs(result[1]) >= Precision || fabs(result[2]) >= Precision|| fabs(result[3]) >= Precision || fabs(result[4]) >= Precision || fabs(result[5]) >= Precision);//计算完成

	ofstream output;
	
	output.open("C:\\Users\\HonghaoQiu\\Desktop\\外部参数_" + dataname + ".txt");
	output << endl << "循环了" << count_cycles << "次" << endl;
	output << "Xs=" << Xs << endl;
	output << "Ys=" << Ys << endl;
	output << "Zs=" << Zs << endl;
	output << "Phi =" << phi << endl;
	output << "Omega=" << omega << endl;
	output << "Kappa=" << kappa << endl;
	output << "R: " << endl;
	output << R[1] << "  " << R[2] << "  " << R[3] << endl;
	output << R[4] << "  " << R[5] << "  " << R[6] << endl;
	output << R[7] << "  " << R[8] << "  " << R[9] << endl;
}

void main() {



	//读取文件
	FILE* f;
	fopen_s(&f, "E:\\Study\\大二下\\计算机视觉与模式识别\\CV2022Course\\Data\\Control_Points.txt", "r");//读取文件
	if (!f)//文件打开失败
		return;
	char* buffer = new char[1000];//缓冲
	if (!buffer)//分配失败
		return;
	fgets(buffer, 100, f);//读取第一行
	delete[]buffer;//删除
	MyPoint* Control_Points = new MyPoint[60];
	for (int i = 0; i < 60; i++)//格式化读取文件
	{
		int temp1, temp2;
		double temp3;
		fscanf_s(f, "%d", &Control_Points[i].ID);  
		fgetc(f);
		fscanf_s(f, "%lf", &Control_Points[i].x);
		fgetc(f);
		fscanf_s(f, "%lf", &Control_Points[i].y);
		fgetc(f);
		fscanf_s(f, "%lf", &Control_Points[i].z);
		fgetc(f);
		fscanf_s(f, "%d", &temp2);
	}
	fclose(f);


	/*for (int i = 0; i < 60; i++
	{
		cout << Control_Points[i].ID << " " << Control_Points[i].x << " " << Control_Points[i].y << endl;
	}*/




	string Right = "Right.bmp";
	MyPoint* Right_points = SemiAutomaticCalibration(Right, Control_Points);


	/*ofstream output;
	output.open("C:\\Users\\HonghaoQiu\\Desktop\\R.txt");
	for (int i = 0; i <= 59; i++)
	{
		
		output << i << " "<< Right_points[i].x << " " << Right_points[i].y << " " << Right_points[i].z  << endl;
		
	}*/
	ExternalParameters(Control_Points, Right_points, Right);

	string Left = "Left.bmp";
	MyPoint* Left_points = SemiAutomaticCalibration(Left, Control_Points);
	/*ofstream o;
	o.open("C:\\Users\\HonghaoQiu\\Desktop\\L.txt");
	for (int i = 0; i <= 59; i++)
	{

		o << i << " " << Left_points[i].x << " " << Left_points[i].y << " " << Left_points[i].z << endl;

	}*/
	ExternalParameters(Control_Points, Left_points, Left);

}