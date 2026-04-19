#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void DrawCircle(Mat& imgOut, int x, int y, int radius)
{
    for (int theta = 0; theta < 360; theta++) {
        double radians = theta * CV_PI / 180;
        int pointX = x + cos(theta) * radius;
        int pointY = y + sin(theta) * radius;

        if (pointX >= 0 && pointX < imgOut.cols - 2 && pointY >= 0 && pointY < imgOut.rows - 2) 
            imgOut.at<Vec3b>(pointY, pointX) = Vec3b(0, 0, 255);
    }
}

void DrawLine(Mat& imgOut, float x1, float y1, float x2, float y2) //Works very slow with big images
{
    float stepX, stepY;
    int distanceX = round(abs(x1 - x2));
    int distanceY = round(abs(y1 - y2));

    if (distanceX == 0)
    {
        stepX = 0;
        stepY = 1;
    }
    else if (distanceY == 0)
    {
        stepX = 1;
        stepY = 0;
    }
    else if (distanceY > distanceX)
    {
        stepX = static_cast<float>(distanceX) / distanceY;
        stepY = 1;
    }
    else
    {
        stepX = 1;
        stepY = static_cast<float>(distanceY) / distanceX;
    }
    while (true)
    {
        if (round(x1) < round(x2))
        {
            x1 += stepX;
            if (round(y1) < round(y2))
                y1 += stepY;
            else if (round(y1) > round(y2))
                y1 -= stepY;
        }
        else if (round(x1) > round(x2))
        {
            x1 -= stepX;
            if (round(y1) < round(y2))
                y1 += stepY;
            else if (round(y1) > round(y2))
                y1 -= stepY;
        }
        else if (round(x1) == round(x2))
        {
            if (round(y1) < round(y2))
                y1 += stepY;
            else if (round(y1) > round(y2))
                y1 -= stepY;
            else
                break;
        }
        if (x1 < 0 && x2 < 0)
        {
            if (y1 < 0 && y2 < 0 || y1 > imgOut.rows - 2 && y2 > imgOut.rows - 2)
                break;
        }
        else if (x1 >= imgOut.cols -2 && x2 >= imgOut.cols - 2)
        {
            if (y1 < 0 && y2 < 0 || y1 >= imgOut.rows - 2 && y2 >= imgOut.rows - 2)
                break;
        }

        if (x1 >= 0 && x1 < imgOut.cols - 2 && y1 >= 0 && y1 < imgOut.rows - 2)
        {
            imgOut.at<Vec3b>(static_cast<int>(y1), static_cast<int>(x1)) = Vec3b(0, 0, 255);
        }
    }
}

void GaussianSmoothing(Mat& img)
{
    double gaussianKernel[3][3] = {
    { 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 },
    { 2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0 },
    { 1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0 } };

    for (int i = 0; i < img.rows - 2; i++)//i satýr
    {
        for (int j = 0; j < img.cols - 2; j++)//j sütun
        {
            double sum = 0;
            for (int k = i; k < i + 3; k++)
            {
                for (int l = j; l < j + 3; l++)
                {
                    sum += img.at<uchar>(k, l) * gaussianKernel[k - i][l - j];
                }
            }
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;

            int sumInt = static_cast<int>(sum);
            img.at<uchar>(i, j) = sumInt;
        }
    }
}

void EdgeDetection(Mat& img, Mat& img2)//img2 açýlarýn bulunduðu dosya
{
    int sobelX[3][3] = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1} };
    int sobelY[3][3] = {{-1, -2, -1},{0, 0, 0},{1, 2, 1} };

    int gx, gy, g; //g = gradient
    double angle; //gradient direction image için ve sonrasýnda non maximum suppression için kullanýyoruz

    for (int i = 0; i < img.rows - 2; i++){
        for (int j = 0; j < img.cols - 2; j++){
            gx = 0;
            gy = 0;
            for (int k = i; k < i + 3; k++){
                for (int l = j; l < j + 3; l++){
                    gx += img.at<uchar>(k, l) * sobelY[k - i][l - j];
                    gy += img.at<uchar>(k, l) * sobelX[k - i][l - j];
                }
            }

            g = sqrt(gx * gx + gy * gy); //Açýyý buluyoruz
            angle = atan2(gy, gx) * 180 / CV_PI;

            if ((angle >= -22.5 && angle < 22.5) || (angle > 157.5 || angle <= -157.5)) 
                angle = 0;
            else if ((angle >= 22.5 && angle < 67.5) || (angle > -157.5 && angle <= -112.5)) 
                angle = 45;
            else if ((angle >= 67.5 && angle < 112.5) || (angle > -112.5 && angle <= -67.5)) 
                angle = 90;
            else angle = 135;

            if (g < 0) g = 0;
            if (g >= 255) g = 255;

            if (i >= 0 && i < img.rows && j >= 0 && j < img.cols){
                img.at<uchar>(i, j) = g;
                img2.at<uchar>(i, j) = angle;
            }
        }
    }
}

void NonMaximumSuppression(Mat img, Mat img2)
{
    for (int i = 0; i < img.rows - 2; i++){
        for (int j = 0; j < img.cols - 2; j++){
            if (img2.at<uchar>(i, j) == 90)
            {
                if (j == 0)
                {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i, j + 1))
                        img.at<uchar>(i, j) = 0;
                }
                else
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i, j + 1) || img.at<uchar>(i, j) <= img.at<uchar>(i, j - 1))
                        img.at<uchar>(i, j) = 0;
            }
            if (img2.at<uchar>(i, j) == 135)
            {
                if (j == 0 && i != 0)
                {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i - 1, j + 1))
                        img.at<uchar>(i, j) = 0;
                }
                else if (i == 0 && j != 0)
                {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j - 1))
                        img.at<uchar>(i, j) = 0;
                }
                else if (i != 0 && j != 0) {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i - 1, j + 1) || img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j - 1))
                        img.at<uchar>(i, j) = 0;
                }
            }
            if (img2.at<uchar>(i, j) == 0)
            {
                if (i == 0)
                {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j))
                        img.at<uchar>(i, j) = 0;
                }
                else
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j) || img.at<uchar>(i, j) <= img.at<uchar>(i - 1, j))
                        img.at<uchar>(i, j) = 0;
            }
            if (img2.at<uchar>(i, j) == 45)
            {
                if (j == 0 || i == 0)
                {
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j + 1))
                        img.at<uchar>(i, j) = 0;
                }
                else
                    if (img.at<uchar>(i, j) <= img.at<uchar>(i - 1, j - 1) || img.at<uchar>(i, j) <= img.at<uchar>(i + 1, j + 1))
                        img.at<uchar>(i, j) = 0;
            }
        }
    }
}

void HysterizedThreshold(Mat img, int thresLow, int thresHigh)
{
    Mat imgCpy = img.clone();

    for (int i = 0; i < imgCpy.rows - 2; i++)
    {
        for (int j = 0; j < imgCpy.cols - 2; j++)
        {
            //kesin olarak 1 ya da 0 olan pikseller burada belirleniyor
            if (imgCpy.at<uchar>(i, j) < thresLow) img.at<uchar>(i, j) = 0;
            if (imgCpy.at<uchar>(i, j) >= thresHigh) img.at<uchar>(i, j) = 255;

            if (thresLow < imgCpy.at<uchar>(i, j) && imgCpy.at<uchar>(i, j) < thresHigh) // Köþeleri ve kenarlarý daha iyi nasýl kontrol edebileceðimi bilmiyorum
            {
                if (j == 0 && i == 0)//sol üst köþe
                {
                    if (imgCpy.at<uchar>(i + 1, j) == 255 || imgCpy.at<uchar>(i, j + 1) == 255)
                        img.at<uchar>(i, j) = 255;
                    else
                        img.at<uchar>(i, j) = 0;
                }
                else if (j == 0)//üst kenar
                {
                    if (imgCpy.at<uchar>(i - 1, j) == 255 || imgCpy.at<uchar>(i, j + 1) == 255 || imgCpy.at<uchar>(i + 1, j) == 255)
                        img.at<uchar>(i, j) = 255;
                    else
                        img.at<uchar>(i, j) = 0;
                }
                else if (i == 0)//sol kenar
                {
                    if (imgCpy.at<uchar>(i, j + 1) == 255 || imgCpy.at<uchar>(i, j - 1) == 255 || imgCpy.at<uchar>(i + 1, j) == 255)
                        img.at<uchar>(i, j) = 255;
                    else
                        img.at<uchar>(i, j) = 0;
                }
                else //kalan kýsýmlarýn hepsi. Sað kenar ve alt kenarý ayarlamadým çünkü 2 pixel gidecek kadar alanýmýz var orada
                {
                    if (imgCpy.at<uchar>(i, j + 1) == 255 || imgCpy.at<uchar>(i, j - 1) == 255 || imgCpy.at<uchar>(i + 1, j) == 255 || imgCpy.at<uchar>(i - 1, j) == 255)
                        img.at<uchar>(i, j) = 255;
                    else
                        img.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}

void LineDetection(Mat img, Mat imgOut,int numberOfLines, int thetaResolution)
{
    int maxRho = sqrt(pow(img.rows, 2) + pow(img.cols, 2));

    int** lines = new int* [numberOfLines];
    for (int i = 0; i < numberOfLines; i++){
        lines[i] = new int[2] {0};
    }
    int highestNumber = 0; //To find the strongest line

    int** accumulator = new int* [2 * maxRho];
    for (int i = 0; i < 2 * maxRho; i++) {
        accumulator[i] = new int [thetaResolution] {0};
    }

    // Iterate through edge pixels
    for (int y = 0; y < img.rows - 2; y++) {
        for (int x = 0; x < img.cols - 2; x++) {
            if (img.at<uchar>(y, x) == 255) { // Edge pixel detected
                for (int theta = 0; theta < thetaResolution; theta++) {
                    double radians = theta * CV_PI / 180;
                    int rho = round(x * cos(radians) + y * sin(radians));
                    if (rho >= 0 && rho < 2 * maxRho) {
                        accumulator[rho][theta]++;
                    }
                }
            }
        }
    }

    for (int i = 0; i < numberOfLines; i++){
        highestNumber = 0;
        for (int rho = 0; rho < 2 * maxRho; rho++) {
            for (int theta = 0; theta < thetaResolution; theta++)
            {
                if (accumulator[rho][theta] > highestNumber) {
                    highestNumber = accumulator[rho][theta];
                    lines[i][0] = rho;
                    lines[i][1] = theta;
                }
            }
        }
        accumulator[lines[i][0]][lines[i][1]] = 0;
    }

    for (int i = 0; i < numberOfLines; i++) {
        int rho = lines[i][0];
        int theta = lines[i][1];
        double radians = theta * CV_PI / 180;
        double a = cos(radians), b = sin(radians);
        int x0 = rho * a;
        int y0 = rho * b;
        DrawLine(imgOut, x0 + 2000 * (-b), y0 + 2000 * a, x0 - 2000 * (-b), y0 - 2000 * a);
    }

    for (int i = 0; i < maxRho * 2; i++) {
        delete[] accumulator[i];
    }
    delete[] accumulator;

    for (int i = 0; i < numberOfLines; i++) {
        delete[] lines[i];
    }
    delete[] lines;
}

void CircleDetection(Mat img, Mat imgOut, int numberOfCircles, int radius)
{
    int** accumulator = new int* [img.rows];
    for (int i = 0; i < img.rows; i++) {
        accumulator[i] = new int [img.cols] {0};
    }

    int** circles = new int* [numberOfCircles];
    for (int i = 0; i < numberOfCircles; i++) {
        circles[i] = new int[2] {0};
    }
    int highestNumber = 0; //To find the strongest circle

    // Iterate through edge pixels
    for (int y = 0; y < img.rows - 2; y++) {
        for (int x = 0; x < img.cols - 2; x++) {
            if (img.at<uchar>(y, x) == 255) { // Edge pixel detected
                for (int theta = 0; theta < 360; theta++) {
                    double radians = theta * CV_PI / 180;
                    int pointX = x + cos(theta) * radius;
                    int pointY = y + sin(theta) * radius;

                    if (pointX >= 0 && pointX < img.cols && pointY >= 0 && pointY < img.rows)
                        accumulator[pointY][pointX]++;
                }
            }
        }
    }

    for (int i = 0; i < numberOfCircles; i++) {
        highestNumber = 0;
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) 
            {
                if (accumulator[y][x] > highestNumber) {
                    highestNumber = accumulator[y][x];
                    circles[i][0] = y;
                    circles[i][1] = x;
                }
            }
        }
        accumulator[circles[i][0]][circles[i][1]] = 0;
    }

    for (int i = 0; i < numberOfCircles; i++) {
        int y = circles[i][0];
        int x = circles[i][1];

        DrawCircle(imgOut, x, y, radius);
    }

    for (int i = 0; i < img.rows; i++) {
        delete[] accumulator[i];
    }
    delete[] accumulator;

    for (int i = 0; i < numberOfCircles; i++) {
        delete[] circles[i];
    }
    delete[] circles;
}

int main()
{
    Mat imgColor = imread("a.png", IMREAD_COLOR);

    if (imgColor.empty()) {
        cout << "Error: Could not load image!" << endl;
        return -1;
    }

    Mat img;
    cv::cvtColor(imgColor, img, COLOR_BGR2GRAY);

    Mat imgCpy2;
    img.copyTo(imgCpy2);

    GaussianSmoothing(img);

    EdgeDetection(img, imgCpy2);//imgCpy2 is the direction image

    NonMaximumSuppression(img, imgCpy2);

    HysterizedThreshold(img, 10, 150); //img, thresLow,thresHigh 

    LineDetection(img, imgColor, 4, 180);// input image, output image, number of lines, theta resolution

    CircleDetection(img, imgColor, 10, 67);// input image, output image, number of circles, radius

    cv::imshow("Edges", img);
    cv::imshow("Lines", imgColor);
    cv::imshow("Direction image", imgCpy2);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}