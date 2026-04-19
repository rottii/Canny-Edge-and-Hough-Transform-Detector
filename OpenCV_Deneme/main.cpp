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
    Mat imgColor = imread("C:\\Users\\iboro\\OneDrive\\Belgeler\\Goruntu isleme resimler\\b.png", IMREAD_COLOR);

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

    //cv::namedWindow("Edges", cv::WINDOW_NORMAL);
    //cv::namedWindow("Lines", cv::WINDOW_NORMAL);
    //cv::namedWindow("Direction image", cv::WINDOW_NORMAL);

    cv::imshow("Edges", img);
    cv::imshow("Lines", imgColor);
    cv::imshow("Direction image", imgCpy2);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}


//#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <vector>
//#include <cmath>
//
////using namespace cv;
////using namespace std;
////
////Mat img, img_copy;
////vector<Point2f> pts;
////int selectedPoint = -1;
////const int radius = 10;
////
////// Check if click is near a point
////int getNearbyPoint(Point p) {
////    Point2f pf((float)p.x, (float)p.y); // convert to Point2f
////    for (int i = 0; i < pts.size(); ++i)
////        if (norm(pf - pts[i]) < radius) return i;
////    return -1;
////}
////
////
////// Mouse callback
////void onMouse(int event, int x, int y, int, void*) {
////    Point mousePos(x, y);
////    if (event == EVENT_LBUTTONDOWN) {
////        selectedPoint = getNearbyPoint(mousePos);
////    }
////    else if (event == EVENT_MOUSEMOVE && selectedPoint != -1) {
////        pts[selectedPoint] = Point2f((float)x, (float)y);
////        img_copy = img.clone();
////        // draw all points and trapezoid
////        for (auto& p : pts) circle(img_copy, p, 5, Scalar(0, 0, 255), -1);
////        if (pts.size() == 4) {
////            vector<Point> poly;
////            for (auto& p : pts) poly.push_back(Point((int)p.x, (int)p.y));
////            polylines(img_copy, poly, true, Scalar(0, 255, 0), 2);
////        }
////        imshow("Move points", img_copy);
////    }
////    else if (event == EVENT_LBUTTONUP) {
////        selectedPoint = -1;
////    }
////}
////
////int main() {
////    img = imread("C:\\Users\\iboro\\OneDrive\\Belgeler\\Goruntu isleme resimler\\w3.jpeg");
////    if (img.empty()) { cout << "Image not found!\n"; return -1; }
////    img_copy = img.clone();
////
////    // Initial points (hardcoded or picked previously)
////    pts = { {0,0}, {200,0}, {300,400}, {0,400} };
////
////    namedWindow("Move points", WINDOW_AUTOSIZE);
////    setMouseCallback("Move points", onMouse, nullptr);
////
////    // Draw initial points
////    for (auto& p : pts) circle(img_copy, p, 5, Scalar(0, 0, 255), -1);
////    vector<Point> poly;
////    for (auto& p : pts) poly.push_back(Point((int)p.x, (int)p.y));
////    polylines(img_copy, poly, true, Scalar(0, 255, 0), 2);
////    imshow("Move points", img_copy);
////
////    cout << "Drag points to adjust, then press any key to warp.\n";
////    waitKey(0);
////
////    // Warp using updated points
////    vector<Point2f> src = pts;
////    // Order TL, TR, BR, BL
////    sort(src.begin(), src.end(), [](Point2f a, Point2f b) { return a.y < b.y; });
////    vector<Point2f> top = { src[0], src[1] }, bottom = { src[2], src[3] };
////    if (top[0].x > top[1].x) swap(top[0], top[1]);
////    if (bottom[0].x > bottom[1].x) swap(bottom[0], bottom[1]);
////    src = { top[0], top[1], bottom[1], bottom[0] };
////
////    int width = (int)round(max(norm(src[0] - src[1]), norm(src[2] - src[3])));
////    int height = (int)round(max(norm(src[0] - src[3]), norm(src[1] - src[2])));
////    vector<Point2f> dst = { {0,0}, {(float)width - 1,0}, {(float)width - 1,(float)height - 1}, {0,(float)height - 1} };
////
////    Mat H = getPerspectiveTransform(src, dst);
////    Mat warped;
////    warpPerspective(img, warped, H, Size(width, height));
////
////    imshow("Warped", warped);
////    imwrite("warped.png", warped);
////    cout << "Warped image saved as warped.png\n";
////    waitKey(0);
////
////    return 0;
////}
//
////float calculateWilkinson(float d1, float e0, float d2, float e1)//QR
////{
////    float delta, mu;
////    std::vector<float> T(4, 0);//we dont even need this array
////
////    T[0] = e0 * e0 + d1 * d1;
////    T[1] = d1 * e1;
////    T[2] = d1 * e1;
////    T[3] = e1 * e1 + d2 * d2;
////
////    delta = (T[0] - T[3]) / 2;
////    mu = T[3] - T[1] * T[1] / (delta + sign(delta) * sqrt(delta * delta + T[1] * T[1]));
////
////    return mu;//mu is the shift
////}
//
//float findDistance(float x0, float y0, float x1, float y1)
//{
//    return sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2));
//}
//
//void inverseMatrix3x3(std::vector<float>& H)
//{
//    float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
//        H[1] * (H[3] * H[8] - H[5] * H[6]) +
//        H[2] * (H[3] * H[7] - H[4] * H[6]);
//
//    float invDet = 1.0f / det;
//
//    std::vector<float> H_inv(9, 0);
//
//    // Row 1
//    H_inv[0] = (H[4] * H[8] - H[5] * H[7]) * invDet;
//    H_inv[1] = (H[2] * H[7] - H[1] * H[8]) * invDet;
//    H_inv[2] = (H[1] * H[5] - H[2] * H[4]) * invDet;
//
//    // Row 2
//    H_inv[3] = (H[5] * H[6] - H[3] * H[8]) * invDet;
//    H_inv[4] = (H[0] * H[8] - H[2] * H[6]) * invDet;
//    H_inv[5] = (H[2] * H[3] - H[0] * H[5]) * invDet;
//
//    // Row 3
//    H_inv[6] = (H[3] * H[7] - H[4] * H[6]) * invDet;
//    H_inv[7] = (H[1] * H[6] - H[0] * H[7]) * invDet;
//    H_inv[8] = (H[0] * H[4] - H[1] * H[3]) * invDet;
//
//    float epsilon = 1e-4;
//    for (int i = 0; i < 9; i++)
//    {
//        if (abs(H_inv[i]) < epsilon) {
//            H_inv[i] = 0.0f;
//        }
//        if (abs(H_inv[i] - round(H_inv[i])) < epsilon) {
//            H_inv[i] = round(H_inv[i]);
//        }
//
//        H[i] = H_inv[i];
//    }
//
//}
//
//float calculateWilkinson(std::vector<float>& A, int start, int end) {
//    //Formül tam olmamasýna raðmen çalýþýyor, neden bilmiyorum
//    // We look at the bottom 2x2 of the active block
//    // [ d[n-1]   e[n-1] ]
//    // [   0      d[n]   ]
//
//    float d1 = A[(end - 1) * 9 + (end - 1)]; // Bottom-left of 2x2
//    float e1 = A[(end - 1) * 9 + end];       // Top-right of 2x2
//    float d2 = A[end * 9 + end];             // Bottom-right of 2x2
//
//    // This is the standard formula for the Wilkinson Shift
//    float d = (d1 * d1 - d2 * d2) / 2.0f;
//    float sign_d = (d >= 0) ? 1.0f : -1.0f;
//    float mu = d2 * d2 - (e1 * e1) / (d + sign_d * sqrt(d * d + e1 * e1));
//
//    return mu;
//}
//
//void calculateGivens(float& r, float& c, float& s, float first, float second)//QR
//{
//    if (second == 0) {
//        c = 1; s = 0; r = first;
//        return;
//    }
//
//    r = sqrt(first * first + second * second);
//    c = first / r;
//    s = - second / r;
//}
//
//void QR(std::vector<float>& A, int column, int row, std::vector<float>& VT)
//{
//    //calculate wilkinson shift
//    //chase the bulge till end(bottom right corner)
//    //calculate the shift again and it goes like this
//    //if you have a diagonal zero, you must first zero its super diagonal(right to the diagonal)
//    //it creates separate blocks that needs to be calculated separately
//    //find out how to calculate them separately
//
//    float r = 0, c = 0, s = 0;
//
//    for (int i = 0; i < row; i++)// diagonal zero control
//    {
//        if (A[i * column + i] == 0)// if the diagonal is zero
//        {
//            for (int j = i + 1; j < row; j++)//since superdiagonal is right of diagonal it's i + 1
//            {
//                calculateGivens(r, c, s, A[j * column + j], A[i * column + j]);
//                for (int k = i; k < column; k++)
//                {
//                    float oldA = A[i * column + k];//because of the order we calculate, we only need oldA
//                    A[i * column + k] = c * oldA + s * A[j * column + k];//row which had zero
//                    A[j * column + k] = -s * oldA + c * A[j * column + k];//row which the bulge is in now
//                }
//            }
//        }
//    }
//
//    int end = 8;
//    while (end > 0)
//    {
//        while (end > 0 && abs(A[(end - 1) * column + end]) < 1e-9) {
//            end--;
//        }
//        if (end == 0) break;
//
//        // Find the start of the current non-zero block
//        int start = end - 1;
//        while (start > 0 && abs(A[(start - 1) * column + start]) > 1e-9) {
//            start--;
//        }
//
//        float shift = calculateWilkinson(A, start, end);
//
//        float y = A[start * column + start] * A[start * column + start] - shift;
//        float z = A[start * column + start] * A[start * column + start + 1];
//
//        calculateGivens(r, c, s, y, z);
//        for (int k = 0; k < 9; k++) { // Full length for A columns and VT rows
//            float a1 = A[k * column + start];
//            float a2 = A[k * column + start + 1];
//            A[k * column + start] = c * a1 - s * a2;
//            A[k * column + start + 1] = s * a1 + c * a2;
//
//            float v1 = VT[start * 9 + k];
//            float v2 = VT[(start + 1) * 9 + k];
//            VT[start * 9 + k] = c * v1 - s * v2;
//            VT[(start + 1) * 9 + k] = s * v1 + c * v2;
//        }
//
//        for (int j = start; j < end; j++)//bulge chasing
//        {
//            //Row
//            calculateGivens(r, c, s, A[j * column + j], A[(j + 1) * column + j]);
//            for (int k = j; k < column; k++) {
//                float r1 = A[j * column + k];
//                float r2 = A[(j + 1) * column + k];
//                A[j * column + k] = c * r1 - s * r2;
//                A[(j + 1) * column + k] = s * r1 + c * r2;
//            }
//
//            //Column
//            // Only if we haven't reached the end of the block
//            if (j < end - 1) {
//                calculateGivens(r, c, s, A[j * column + j + 1], A[j * column + j + 2]);
//
//                for (int k = 0; k < 9; k++) {
//                    // Update Matrix A Columns
//                    float c1 = A[k * column + j + 1];
//                    float c2 = A[k * column + j + 2];
//                    A[k * column + j + 1] = c * c1 - s * c2;
//                    A[k * column + j + 2] = s * c1 + c * c2;
//
//                    // Update Matrix VT Rows
//                    float v1 = VT[(j + 1) * 9 + k];
//                    float v2 = VT[(j + 2) * 9 + k];
//                    VT[(j + 1) * 9 + k] = c * v1 - s * v2;
//                    VT[(j + 2) * 9 + k] = s * v1 + c * v2;
//                }
//            }
//        }
//    }
//}
//
//void triangulate(std::vector<float>& A, int column, int row, std::vector<float>& matrix, int offset, int i)
//{
//    for (i; i < row - offset; i++)//sütun sayýsý kadar
//    {
//        int vecStart = i + offset;
//
//        std::vector<float> v(row - vecStart, 0);//Her sütunla birlikte 1 aþaðý indiðimiz için v row - vecStart boyutunda
//
//        float x = 0;
//        for (int j = vecStart; j < row; j++)
//        {
//            v[j - vecStart] = A[j * column + i];
//            x += A[j * column + i] * A[j * column + i];
//        }
//
//        if (A[vecStart * column + i] > 0) x = -sqrt(x);
//        else x = sqrt(x);
//        v[0] -= x;
//
//        float vNorm = 0;
//        for (int j = 0; j < row - vecStart; j++)
//            vNorm += v[j] * v[j];
//
//        if (vNorm == 0) vNorm = 1; //when it's 0 it makes P undefined
//
//        //A =  A - v(beta * v^T * A)
//        float beta = 2.0f / vNorm;
//
//        for (int j = i; j < column; j++)//A yý güncelleme
//        {
//            float sum = 0.0f;
//            for (int k = vecStart; k < row; k++)
//                sum += v[k - vecStart] * A[k * column + j];
//
//            sum *= beta;
//
//            for (int k = vecStart; k < row; k++)
//                A[k * column + j] -= sum * v[k - vecStart];
//        }
//
//        for (int j = 0; j < row; j++)//VT ve U nun boyutu row a baðlý bu yüzden hepsi row
//        {
//            float sum = 0.0f;
//            for (int k = vecStart; k < row; k++)
//                sum += v[k - vecStart] * matrix[k * row + j];
//
//            sum *= beta;
//
//            for (int k = vecStart; k < row; k++)
//                matrix[k * row + j] -= sum * v[k - vecStart];
//        }
//    }
//}
//
//void computeSVD(std::vector<float>& A, std::vector<float>& w, std::vector<float>& u, std::vector<float>& v)
//{
//    //Bidiagonalization
//    std::vector<float> newA(72, 0);
//    for (int k = 0; k < 8; k++)//yazdýr
//    {
//        for (int j = 0; j < 9; j++)
//            std::printf("%10.4f ", A[k * 9 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//
//    for (int i = 0; i < 9; i++)
//        v[10 * i] = 1;
//
//    for(int adim = 0; adim < 8; adim++)//HOUSEHOLD BIDIAGONALIZATION
//    {
//        triangulate(A, 9, 8, u, 0, adim);
//
//        for (int k = 0; k < 8; k++)//yazdýr A
//        {
//            for (int j = 0; j < 9; j++)
//                std::printf("%10.4f ", A[k * 9 + j]);
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//
//        std::fill(newA.begin(), newA.end(), 0.0f);
//        for (int i = 0; i < 8; i++)//Transpose A to use the same triangulate function
//        {
//            for (int j = 0; j < 9; j++)
//                newA[j * 8 + i] = A[i * 9 + j];
//        }
//
//        for (int k = 0; k < 9; k++)//yazdýr
//        {
//            for (int j = 0; j < 8; j++)
//                std::printf("%10.4f ", newA[k * 8 + j]);
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//
//        //Right zeroing
//        triangulate(newA, 8, 9, v, 1, adim);
//
//        for (int i = 0; i < 8; i++)//Transpose again
//        {
//            for (int j = 0; j < 9; j++)
//                A[i * 9 + j] = newA[j * 8 + i];
//        }
//    }
//    std::cout << "____________________________________________________________________" << std::endl;
//
//    for (int k = 0; k < 9; k++)//yazdýr
//    {
//        for (int j = 0; j < 9; j++)
//            std::printf("%10.4f ", v[k * 9 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    for (int k = 0; k < 8; k++)//yazdýr A
//    {
//        for (int j = 0; j < 9; j++)
//            std::printf("%10.4f ", A[k * 9 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < 9; i++)//Removing float errors
//    {
//        for (int j = 0; j < 8; j++)
//        {
//            if (j == i || j == i + 1) continue;
//        
//            if (std::abs(newA[i * 8 + j]) < 0.01f)
//                newA[i * 8 + j] = 0;
//        }
//    }
//
//    for (int i = 0; i < 9; i++)
//        A.push_back(0.0f);
//
//    QR(A, 9, 9, v);
//
//    for (int k = 0; k < 9; k++)//yazdýr
//    {
//        for (int j = 0; j < 9; j++)
//            std::printf("%10.4f ", A[k * 9 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    for (int k = 0; k < 9; k++)//yazdýr
//    {
//        for (int j = 0; j < 9; j++)
//            std::printf("%10.4f ", v[k * 9 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//}
//
//void computeHomography(std::vector<float>& H, std::vector<float> src, std::vector<float> dst)
//{
//    std::vector<float> A(9 * 8, 0);
//    std::vector<float> u(8 * 8, 0);//It should be 8x8 but since i dont need it, im just doing it like this to not break my code
//    std::vector<float> v(9 * 9, 0);
//
//    for (int i = 0; i < 4; i++)//noktalarý ver
//    {
//        A[i * 18 + 0] = -src[i * 2 + 0];
//        A[i * 18 + 1] = -src[i * 2 + 1];
//        A[i * 18 + 2] = -1;
//        A[i * 18 + 3] = 0;
//        A[i * 18 + 4] = 0;
//        A[i * 18 + 5] = 0;
//        A[i * 18 + 6] = src[i * 2 + 0] * dst[i * 2 + 0];
//        A[i * 18 + 7] = src[i * 2 + 1] * dst[i * 2 + 0];
//        A[i * 18 + 8] = dst[i * 2 + 0];
//        A[i * 18 + 9] = 0;
//        A[i * 18 + 10] = 0;
//        A[i * 18 + 11] = 0;
//        A[i * 18 + 12] = -src[i * 2 + 0];
//        A[i * 18 + 13] = -src[i * 2 + 1];
//        A[i * 18 + 14] = -1;
//        A[i * 18 + 15] = src[i * 2 + 0] * dst[i * 2 + 1];
//        A[i * 18 + 16] = src[i * 2 + 1] * dst[i * 2 + 1];
//        A[i * 18 + 17] = dst[i * 2 + 1];
//    }
//
//    computeSVD(A, dst, u, v);
//
//    int smallestIndex = 0;
//    for (int i = 0; i < 9; i++)
//    {
//        if ((abs(A[i * 10]) < abs(A[smallestIndex * 10])))
//            smallestIndex = i;
//    }
//    std::cout << smallestIndex << std::endl;
//
//    for (int i = 0; i < 9; i++)
//        H[i] = v[smallestIndex * 9 + i];
//
//}
//
//void getPixelColor(const uchar* inputImage, int x, int y, int width, uchar color[3])
//{
//    int index = (y * width + x) * 3;
//    for (int i = 0; i < 3; i++)
//        color[i] = inputImage[index + i];
//}
//
//void mapImage(const uchar* inputImage, std::vector<uchar>& outputImage, int inputWidth, int inputHeight, int destWidth ,int destHeight, std::vector<float> H)//H = H^-1
//{
//    uchar tl[3], tr[3], bl[3], br[3];
//    float topMix, bottomMix, finalColor;
//
//    for (int y_dest = 0; y_dest < destHeight; y_dest++)
//    {
//        //we precalculate these for optimization
//        float h_u_y = H[1] * y_dest + H[2];
//        float h_v_y = H[4] * y_dest + H[5];
//        float h_w_y = H[7] * y_dest + H[8];
//        
//        for (int x_dest = 0; x_dest < destWidth; x_dest++)
//        {
//            float u = H[0] * x_dest + h_u_y;
//            float v = H[3] * x_dest + h_v_y;
//            float w = H[6] * x_dest + h_w_y;
//
//            float x_src = u / w;
//            float y_src = v / w;
//
//            int dest_idx = (y_dest * destWidth + x_dest) * 3;
//
//            if (x_src < 0 || x_src >= inputWidth - 1 || y_src < 0 || y_src >= inputHeight - 1)
//            {
//                for (int k = 0; k < 3; k++)
//                    outputImage[dest_idx + k] = 0;
//                continue;
//            }
//
//            int x_floor = (int)x_src;      // Integer part (left)
//            int y_floor = (int)y_src;      // Integer part (top)
//            int x_ceil = x_floor + 1;      // Right neighbor
//            int y_ceil = y_floor + 1;      // Bottom neighbor
//
//            float x_weight = x_src - x_floor;
//            float y_weight = y_src - y_floor;
//
//            //Get the colors of the 4 neighbors
//            //i can change these to normal array, idk if it makes any difference tho
//            //it definitely did
//            getPixelColor(inputImage, x_floor, y_floor, inputWidth, tl); // Top-Left
//            getPixelColor(inputImage, x_ceil,  y_floor, inputWidth, tr); // Top-Right
//            getPixelColor(inputImage, x_floor, y_ceil, inputWidth, bl); // Bottom-Left
//            getPixelColor(inputImage, x_ceil,  y_ceil, inputWidth, br); // Bottom-Right
//
//            // Interpolate Top pair and Bottom pair and get the final color
//            for (int k = 0; k < 3; k++)
//            {
//                topMix = tl[k] * (1.0f - x_weight) + tr[k] * x_weight;
//                bottomMix = bl[k] * (1.0f - x_weight) + br[k] * x_weight;
//
//                finalColor = topMix * (1.0f - y_weight) + bottomMix * y_weight;
//
//                outputImage[dest_idx + k] = finalColor;
//            }
//        }
//    }
//}
//
//cv::Mat img, img_copy;
//std::vector<cv::Point2f> pts;
//int selectedPoint = -1;
//const int radius = 10;
//
//int getNearbyPoint(cv::Point p) {
//    cv::Point2f pf((float)p.x, (float)p.y); // convert to Point2f
//    for (int i = 0; i < pts.size(); ++i)
//        if (norm(pf - pts[i]) < radius) return i;
//    return -1;
//}
//
//void onMouse(int event, int x, int y, int, void*) {
//    cv::Point mousePos(x, y);
//    if (event == cv::EVENT_LBUTTONDOWN) {
//        selectedPoint = getNearbyPoint(mousePos);
//    }
//    else if (event == cv::EVENT_MOUSEMOVE && selectedPoint != -1) {
//        pts[selectedPoint] = cv::Point2f((float)x, (float)y);
//        img_copy = img.clone();
//        // draw all points and trapezoid
//        for (auto& p : pts) circle(img_copy, p, 5, cv::Scalar(0, 0, 255), -1);
//        if (pts.size() == 4) {
//            std::vector<cv::Point> poly;
//            for (auto& p : pts) poly.push_back(cv::Point((int)p.x, (int)p.y));
//            polylines(img_copy, poly, true, cv::Scalar(0, 255, 0), 2);
//        }
//        imshow("Move points", img_copy);
//    }
//    else if (event == cv::EVENT_LBUTTONUP) {
//        selectedPoint = -1;
//    }
//}
//
//int main()
//{
//    img = cv::imread("C:\\Users\\iboro\\OneDrive\\Belgeler\\Goruntu isleme resimler\\w3.jpeg");
//    if (img.empty()) { std::cout << "Image not found!\n"; return -1; }
//    img_copy = img.clone();
//
//    // Ensure it's continuous
//    if (!img.isContinuous()) {
//        img = img.clone();
//    }
//    uchar* inputImageArray = img.data;
//
//    std::vector<uchar> outputImageArray;
//    
//    // Initial points (hardcoded or picked previously)
//    pts = {
//        cv::Point2f(img.cols / 2 - 150, img.rows / 2 - 150),
//        cv::Point2f(img.cols / 2,      img.rows / 2 - 150),
//        cv::Point2f(img.cols / 2,      img.rows / 2),
//        cv::Point2f(img.cols / 2 - 150, img.rows / 2)
//    };
//
//    cv::namedWindow("Move points");
//    cv::setMouseCallback("Move points", onMouse, nullptr);
//
//    // Draw initial points
//    for (auto& p : pts) circle(img_copy, p, 5, cv::Scalar(0, 0, 255), -1);
//    std::vector<cv::Point> poly;
//    for (auto& p : pts) poly.push_back(cv::Point((int)p.x, (int)p.y));
//    polylines(img_copy, poly, true, cv::Scalar(0, 255, 0), 2);
//    imshow("Move points", img_copy);
//
//    cv::waitKey(0);
//
//    // Warp using updated points
//    std::vector<cv::Point2f> src = pts;
//    // Order TL, TR, BR, BL
//    sort(src.begin(), src.end(), [](cv::Point2f a, cv::Point2f b) { return a.y < b.y; });
//    std::vector<cv::Point2f> top = { src[0], src[1] }, bottom = { src[2], src[3] };
//    if (top[0].x > top[1].x) swap(top[0], top[1]);
//    if (bottom[0].x > bottom[1].x) swap(bottom[0], bottom[1]);
//    src = { top[0], top[1], bottom[1], bottom[0] };
//
//    int width = (int)round(std::max(norm(src[0] - src[1]), norm(src[2] - src[3])));
//    int height = (int)round(std::max(norm(src[0] - src[3]), norm(src[1] - src[2])));
//    std::vector<cv::Point2f> dst = { {0,0}, {(float)width,0}, {(float)width,(float)height}, {0,(float)height} };
//
//
//    //---CORNER POINTS---
//    //input corners, used to calculate H 
//    std::vector<float> inputCorners(src.size() * 2);
//    std::memcpy(inputCorners.data(), src.data(), src.size() * sizeof(cv::Point2f));
//    //output corners, output rectangular image
//    std::vector<float> outputCorners(dst.size() * 2);
//    std::memcpy(outputCorners.data(), dst.data(), dst.size() * sizeof(cv::Point2f));
//
//    //not needed, but keeping them to remind me where the numbers came from
//    int rows = 4;
//    int cols = 2;
//    int rowc = rows * cols;
//
//    //change these into arrays
//    std::vector<float> A(9 * rowc, 0);
//    std::vector<float> H(9, 0);
//
//    float scale = 100;
//    for (int i = 0; i < rowc; i++)
//    {
//        inputCorners[i] /= scale;
//        outputCorners[i] /= scale;
//    }
//
//    ////Bunlarý ekrandan seçeceðiz
//    ////input
//    //matrix[0 * cols + 0] = 118 / scale;
//    //matrix[0 * cols + 1] = 10 / scale;
//    //matrix[1 * cols + 0] = 586 / scale;
//    //matrix[1 * cols + 1] = 25 / scale;
//    //matrix[2 * cols + 0] = 640 / scale;
//    //matrix[2 * cols + 1] = 480 / scale;
//    //matrix[3 * cols + 0] = 20 / scale;
//    //matrix[3 * cols + 1] = 470 / scale;
//    ////output
//    //matrix2[0 * cols + 0] = 0 / scale;
//    //matrix2[0 * cols + 1] = 0 / scale;
//    //matrix2[1 * cols + 0] = 600 / scale;
//    //matrix2[1 * cols + 1] = 0 / scale;
//    //matrix2[2 * cols + 0] = 600 / scale;
//    //matrix2[2 * cols + 1] = 800 / scale;
//    //matrix2[3 * cols + 0] = 0 / scale;
//    //matrix2[3 * cols + 1] = 800 / scale;
//
//    /*
//        std::vector<cv::Point2f> srcPoints;
//        srcPoints.push_back(cv::Point2f(118, 10));   // Sol Üst (x, y)
//        srcPoints.push_back(cv::Point2f(586, 25));   // Sað Üst
//        srcPoints.push_back(cv::Point2f(640, 480));  // Sað Alt
//        srcPoints.push_back(cv::Point2f(20, 470));   // Sol Alt
//
//        // Hedef Noktalar (Düzeltilmiþ/Çýkýþ Görüntüsü)
//        std::vector<cv::Point2f> dstPoints;
//        dstPoints.push_back(cv::Point2f(0, 0));      // Sol Üst
//        dstPoints.push_back(cv::Point2f(600, 0));    // Sað Üst
//        dstPoints.push_back(cv::Point2f(600, 800));  // Sað Alt
//        dstPoints.push_back(cv::Point2f(0, 800));    // Sol Alt
//
//
//    [   1.253149   0.266975 -150.541286 ]
//    [  -0.073504   2.293330 -14.259807 ]
//    [  -0.000057   0.000700   1.000000 ]*/
//
//    //for (int i = 0; i < 4; i++)
//    //{
//    //    for (int k = 0; k < 2; k++)
//    //        std::cout << matrix[i * cols + k] << " ";
//    //    std::cout << std::endl;
//    //}
//    //std::cout << std::endl;
//    //for (int i : matrix)
//    //    std::cout << i << "\n";
//
//    computeHomography(H, inputCorners, outputCorners);
//
//    //scale it back so you can use it in mapping
//    for (int i = 0; i < rowc; i++)
//    {
//        outputCorners[i] *= scale;
//    }
//
//    for (int i = 0; i < 2; i++)
//        H[i * 3 + 2] *= scale;
//    for (int i = 0; i < 2; i++)
//        H[6 + i] /= scale;
//    for (int i = 0; i < 9; i++)//normalization
//        H[i] /= H[8];
//
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 3; j++)
//            std::printf("%10.4f ", H[i * 3 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//
//    //std::vector<float> points(8, 0);
//    ////these are the points we selected on input image
//    //int widthTop = findDistance(points[0], points[1], points[2], points[3]);//x0, y0, x1, y1
//    //int widthBottom = findDistance(points[6], points[7], points[4], points[5]);//x3, y3, x2, y2
//    //int destWidth = int(widthTop > widthBottom ? widthTop : widthBottom);//finds the max
//
//    //int heightTop = findDistance(points[0], points[1], points[6], points[7]);//x0, y0, x3, y3
//    //int heightBottom = findDistance(points[2], points[3], points[4], points[5]);//x1, y1, x2, y2
//    //int destHeight = int(heightTop > heightBottom ? heightTop : heightBottom);
//    /*
//    float minX = 99999.0f, minY = 99999.0f;//For offsets
//    float maxX = -99999.0f, maxY = -99999.0f;//For source width and height
//
//    for (int i = 0; i < 4; i++)
//    {
//        float x = points[i * 2];
//        float y = points[i * 2 + 1];
//
//        float u = H[0] * x + H[1] * y + H[2];
//        float v = H[3] * x + H[4] * y + H[5];
//        float w = H[6] * x + H[7] * y + H[8];
//
//        float x_prime = u / w;
//        float y_prime = v / w;
//
//        if (x_prime < minX) minX = x_prime;
//        if (x_prime > maxX) maxX = x_prime;
//        if (y_prime < minY) minY = y_prime;
//        if (y_prime > maxY) maxY = y_prime;
//    }
//
//    int destWidth = (int)ceil(maxX - minX);
//    int destHeight = (int)ceil(maxY - minY);
//
//    int offsetX = -(int)floor(minX);
//    int offsetY = -(int)floor(minY);
//
//    float offX = (float)offsetX; 
//    float offY = (float)offsetY; 
//
//    //T * H (T is the translation matrix)
//    H[0] += offX * H[6];
//    H[1] += offX * H[7];
//    H[2] += offX * H[8];
//    H[3] += offY * H[6];
//    H[4] += offY * H[7];
//    H[5] += offY * H[8];*/
//
//    inverseMatrix3x3(H);
//
//    int inputWidth = img.cols;
//    int inputHeight = img.rows;
//
//    outputImageArray.resize(width * height * 3);
//    std::cout << "Calculated Width: " << width << " | Calculated Height: " << height << std::endl;
//    
//    mapImage(inputImageArray, outputImageArray, inputWidth, inputHeight, width, height, H);
//
//    //for (int i = 0; i < height; i++)
//    //{
//    //    for (int j = 0; j < width; j++)
//    //    {
//    //        for (int k = 0; k < 3; k++)
//    //            std::cout << (int)outputImageArray[(i * width + j) * 3 + k] << ",";
//    //        std::cout << " ";
//    //    }
//    //    std::cout << std::endl;
//    //}
//
//    cv::Mat reconstructedImg(height, width, CV_8UC3, outputImageArray.data());
//    cv::Mat finalImage = reconstructedImg.clone();
//    cv::imshow("Result", finalImage);
//    cv::waitKey(0);
//
//    for (int i = 0; i < 3; i++)
//    {
//        for (int j = 0; j < 3; j++)
//            std::printf("%10.4f ", H[i * 3 + j]);
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;
//    
//
//    //for (int y_dest = 0; y_dest < destHeight; y_dest++)
//    //{
//    //    for (int x_dest = 0; x_dest < destWidth; x_dest++)
//    //    {
//    //        float u = H[0] * x_dest + H[1] * y_dest + H[2];
//    //        float v = H[3] * x_dest + H[4] * y_dest + H[5];
//    //        float w = H[6] * x_dest + H[7] * y_dest + H[8];
//    //
//    //        float x_src = u / w;
//    //        float y_src = v / w;
//    //
//    //        if (x_src < 0 || x_src >= srcWidth - 1 || y_src < 0 || y_src >= srcHeight - 1) {
//    //            // Set pixel to Black (0)
//    //            // setPixel(outputImage, x_dest, y_dest, 0); 
//    //            continue;
//    //        }
//    //
//    //        //Bilinear interpolation
//    //        int x_floor = (int)x_src;       // Integer part (left)
//    //        int y_floor = (int)y_src;       // Integer part (top)
//    //        int x_ceil = x_floor + 1;      // Right neighbor
//    //        int y_ceil = y_floor + 1;      // Bottom neighbor
//    //
//    //        // Calculate the "weights" (how close are we to the next pixel?)
//    //        float x_weight = x_src - x_floor;
//    //        float y_weight = y_src - y_floor;
//    //
//    //        /*// Get the colors of the 4 neighbors
//    //        // (You would implement getPixel to retrieve the value from your specific image format)
//    //        // For simplicity, let's assume we are dealing with a single channel (Grayscale)
//    //        unsigned char tl = getPixel(inputImage, x_floor, y_floor, srcWidth); // Top-Left
//    //        unsigned char tr = getPixel(inputImage, x_ceil,  y_floor, srcWidth); // Top-Right
//    //        unsigned char bl = getPixel(inputImage, x_floor, y_ceil,  srcWidth); // Bottom-Left
//    //        unsigned char br = getPixel(inputImage, x_ceil,  y_ceil,  srcWidth); // Bottom-Right
//    //
//    //        // Interpolate Top pair and Bottom pair
//    //        float top_mix    = tl * (1.0f - x_weight) + tr * x_weight;
//    //        float bottom_mix = bl * (1.0f - x_weight) + br * x_weight;
//    //
//    //        // Combine Top and Bottom
//    //        float final_color = top_mix * (1.0f - y_weight) + bottom_mix * y_weight;
//    //
//    //        // 5. ASSIGN TO DESTINATION
//    //        setPixel(outputImage, x_dest, y_dest, destWidth, (unsigned char)final_color);*/
//    //    }
//    //}
//
//    //for (int i = 0; i < rows; i++)
//    //{
//    //    A[i * 18 + 0] = -matrix[i * cols + 0];
//    //    A[i * 18 + 1] = -matrix[i * cols + 1];
//    //    A[i * 18 + 2] = -1;
//    //    A[i * 18 + 3] = 0;
//    //    A[i * 18 + 4] = 0;
//    //    A[i * 18 + 5] = 0;
//    //    A[i * 18 + 6] = matrix[i * cols + 0] * matrix2[i * cols + 0];
//    //    A[i * 18 + 7] = matrix[i * cols + 1] * matrix2[i * cols + 0];
//    //    A[i * 18 + 8] = matrix2[i * cols + 0];
//    //    A[i * 18 + 9] = 0;
//    //    A[i * 18 + 10] = 0;
//    //    A[i * 18 + 11] = 0;
//    //    A[i * 18 + 12] = -matrix[i * cols + 0];
//    //    A[i * 18 + 13] = -matrix[i * cols + 1];
//    //    A[i * 18 + 14] = -1;
//    //    A[i * 18 + 15] = matrix[i * cols + 0] * matrix2[i * cols + 1];
//    //    A[i * 18 + 16] = matrix[i * cols + 1] * matrix2[i * cols + 1];
//    //    A[i * 18 + 17] = matrix2[i * cols + 1];
//    //}
//    //
//    //for (int i = 0; i < 8; i++)
//    //{
//    //    for (int j = 0; j < 9; j++)
//    //        std::cout << A[i * 9 + j] << " ";
//    //    std::cout << std::endl;
//    //}
//    //
//    ////SVD yi kendin hesaplayacaksýn
//    //cv::Mat B(8,9, CV_32F, A.data());
//    //cv::Mat w, u, vt;
//    //cv::SVD::compute(B, w, u, vt, cv::SVD::FULL_UV);
//    //
//    //std::vector<float> result_w = (std::vector<float>)w;
//    //std::vector<float> result_vt = (std::vector<float>)vt;
//    //
//    //std::cout << "Standard Mat Print:\n" << vt << "\n" << std::endl;
//    //
//    //
//    //float* rowPtr = vt.ptr<float>(vt.rows - 1); // Get pointer to the 8th row
//    //for (int i = 0; i < 9; i++) {
//    //    H[i] = rowPtr[i];
//    //    if (H[i] < 0.0001) H[i] = 0;
//    //}
//    //
//    //for (int i = 0; i < 9; i++) {
//    //    H[i] = H[i] / H[8];
//    //}
//    //
//    //for (int i = 0; i < 3; i++)
//    //{
//    //    for (int j = 0; j < 3; j++)
//    //    {
//    //        std::cout << H[i * 3 + j] << " ";
//    //    }
//    //    std::cout << std::endl;
//    //}
//    //
//    //
//    //
//    //for (int i = 0; i < 9; i++)
//    //{
//    //    H[i] = result_vt[64 + i];
//    //    std::cout << H[i] << " ";
//    //}
//    //
//    //
//    //std::cout << vt * B << std::endl;
//    //
//    //
//    //cv::Mat vt_clean;
//    // Set any value smaller than 1e-5 to zero
//    //cv::threshold(cv::abs(vt), vt_clean, 1e-5, 0, cv::THRESH_TOZERO);
//    //
//    // Restore the signs (thresholding absolute values loses signs)
//    //cv::multiply(vt_clean, (vt >= 0) / 255 + (vt < 0) / -255, vt_clean, 1, CV_32F);
//    //// Note: Simpler to just use a custom loop for precise control:
//    //
//    //for (int i = 0; i < vt.rows; i++) {
//    //    for (int j = 0; j < vt.cols; j++) {
//    //        float val = vt.at<float>(i, j);
//    //        if (std::abs(val) < 0.0001) val = 0.0f; // Kill the noise
//    //        std::printf("%8.4f ", val);           // Print with 4 decimal places
//    //    }
//    //    std::printf("\n");
//    //}
//    //
//    //// 3. Print using Vector Loop
//    //std::cout << "Manual Vector Print:" << std::endl;
//    //for (size_t i = 0; i < result_w.size(); ++i) {
//    //    std::cout << "Singular Value " << i << ": " << result_w[i] << std::endl;
//    //}
//    //
//    //// Create an 8x9 matrix of zeros
//    //cv::Mat Sigma = cv::Mat::zeros(8, 9, CV_32F);
//    //
//    //// Fill the diagonal with the values from w
//    //for (int i = 0; i < w.rows; i++) {
//    //    Sigma.at<float>(i, i) = w.at<float>(i);
//    //}
////
//    //std::cout << "Sigma Matrix (8x9):\n"
//    //    << cv::format(Sigma, cv::Formatter::FMT_NUMPY)
//    //    << std::endl;
//    //std::cout << "Value at (1,2): " << matrix[1 * cols + 2] << std::endl;
//
//    return 0;
//}
//
//
////
////#include <iostream>
////#include <vector>
////#include <opencv2/opencv.hpp>
////#include <opencv2/calib3d.hpp>
////
////int main() {
////    // ==========================================
////    // 1. KOORDÝNATLARI GÝR (Senin Resim Verilerin)
////    // ==========================================
////    // Buradaki deðerleri kendi projedeki deðerlerle birebir ayný yap.
////
////    // Kaynak Noktalar (Bozuk/Giriþ Görüntüsü)
////    std::vector<cv::Point2f> srcPoints;
////    srcPoints.push_back(cv::Point2f(118, 10));   // Sol Üst (x, y)
////    srcPoints.push_back(cv::Point2f(586, 25));   // Sað Üst
////    srcPoints.push_back(cv::Point2f(640, 480));  // Sað Alt
////    srcPoints.push_back(cv::Point2f(20, 470));   // Sol Alt
////
////    // Hedef Noktalar (Düzeltilmiþ/Çýkýþ Görüntüsü)
////    std::vector<cv::Point2f> dstPoints;
////    dstPoints.push_back(cv::Point2f(0, 0));      // Sol Üst
////    dstPoints.push_back(cv::Point2f(600, 0));    // Sað Üst
////    dstPoints.push_back(cv::Point2f(600, 800));  // Sað Alt
////    dstPoints.push_back(cv::Point2f(0, 800));    // Sol Alt
////
////    // ==========================================
////    // 2. REFERANS MATRÝSÝ HESAPLA (OpenCV ile)
////    // ==========================================
////    // findHomography fonksiyonu en iyi matrisi RANSAC vb. kullanmadan direkt hesaplar (0 parametresi)
////    cv::Mat H = cv::findHomography(srcPoints, dstPoints, 0);
////
////    if (H.empty()) {
////        std::cerr << "Hata: Matris hesaplanamadý! Noktalarý kontrol et." << std::endl;
////        return -1;
////    }
////
////    // ==========================================
////    // 3. NORMALÝZASYON (Kritik Adým!)
////    // ==========================================
////    // Senin kodun ile bu kodu kýyaslayabilmek için son elemaný 1 yapýyoruz.
////    // Çünkü homografi ölçekten baðýmsýzdýr.
////    double scale = H.at<double>(2, 2);
////    H = H / scale;
////
////    // ==========================================
////    // 4. SONUCU YAZDIR
////    // ==========================================
////    std::cout << "=== OPENCV TARAFINDAN HESAPLANAN DOGRU MATRIS ===" << std::endl;
////    std::cout << "(Kendi kodunun ciktisini bu degerlerle kiyasla)" << std::endl << std::endl;
////
////    for (int i = 0; i < H.rows; i++) {
////        std::cout << "[ ";
////        for (int j = 0; j < H.cols; j++) {
////            // Virgülden sonra 6 basamak hassasiyetle yazdýr
////            printf("%10.6f ", H.at<double>(i, j));
////        }
////        std::cout << "]" << std::endl;
////    }
////
////    std::cout << "\nNot: Eger senin kodun bu degerlerin aynisini (veya cok yakinini) buluyorsa," << std::endl;
////    std::cout << "SVD ve DLT algoritman kusursuz calisiyor demektir." << std::endl;
////
////    return 0;
////}