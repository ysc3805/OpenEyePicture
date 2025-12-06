#include <vector>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

using namespace std;
using namespace cv;

char mode = 'F';

struct Eyepoints {
    Point point2x2[2][2];
    Point center;
    Rect eyebox;
    int eyesize;
};

struct Eye_pair {
    Eyepoints lefteye;
    Eyepoints righteye;
    int eyesize2;
};

vector<vector<Eye_pair>> eyes;

vector<Mat> imgs;
int imgnum = 0;
int facenum = 0;
vector<int> maxeyeidx;


vector<vector<Point>> face_center;
vector<vector<Rect>> face_crop_rect;

Mat eyesizes;

void Readimg() {
    while (1) {
        cout << "이미지 수를 입력하세요(2장 이상) : ";
        cin >> imgnum;
        if (imgnum > 1) break;
        else cout << "2장 이상으로 입력하세요\n";
    }

    imgs.reserve(imgnum);

    for (int i = 1; i <= imgnum; i++) {
        string filename = "img (" + to_string(i) + ").jpg";
        Mat temp = imread(filename);
        if (temp.empty()) continue;
        imgs.push_back(temp);
    }
}

vector<Rect> FaceDetection(Mat img) {
    CascadeClassifier cascade;
    cascade.load("C:/Users/tkdcj/Downloads/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector<Rect> detected;
    cascade.detectMultiScale(gray, detected, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    vector<Rect> out;
    Mat debug = img.clone();

    for (int i = 0; i < detected.size(); i++) {
        Rect r = detected[i];
        rectangle(debug, r, Scalar(0, 255, 0), 2);

        int cx = r.x + r.width / 2;
        int cy = r.y + r.height / 2;
        int new_w = r.width * 1.5;
        int new_h = r.height * 1.5;
        int new_x = cx - new_w / 2;
        int new_y = cy - new_h / 2;

        if (new_x < 0) new_x = 0;
        if (new_y < 0) new_y = 0;
        if (new_x + new_w > img.cols) new_w = img.cols - new_x;
        if (new_y + new_h > img.rows) new_h = img.rows - new_y;

        Rect nr(new_x, new_y, new_w, new_h);
        rectangle(debug, nr, Scalar(0, 0, 255), 2);

        out.push_back(nr);
    }

    imshow("FaceDetection_Debug", debug);
    return out;
}


vector<Mat> CropFaces(vector<Mat> imgs, vector<Rect> faces) {
    vector<Mat> crops;

    for (int j = 0; j < faces.size(); j++) {
        Rect r = faces[j];
        for (int i = 0; i < imgs.size(); i++) {
            Mat img = imgs[i];
            if (r.x + r.width <= img.cols && r.y + r.height <= img.rows)
                crops.push_back(img(r));
        }
    }
    return crops;
}

Mat Merge(vector<Mat> imgs) {
    if (imgs.empty()) return Mat();

    int h = imgs[0].rows;
    int w_total = 0;

    for (auto& im : imgs) w_total += im.cols;

    Mat merged(h, w_total, imgs[0].type());
    int offset = 0;

    for (auto& im : imgs) {
        im.copyTo(merged(Rect(offset, 0, im.cols, im.rows)));
        offset += im.cols;
    }

    return merged;
}

void EyeDataUpdate(Mat shape, int i, int j, int N) {

    eyes[j][i].lefteye.point2x2[0][0] = Point(shape.at<float>(37), shape.at<float>(37 + N));
    eyes[j][i].lefteye.point2x2[0][1] = Point(shape.at<float>(38), shape.at<float>(38 + N));
    eyes[j][i].lefteye.point2x2[1][0] = Point(shape.at<float>(41), shape.at<float>(41 + N));
    eyes[j][i].lefteye.point2x2[1][1] = Point(shape.at<float>(40), shape.at<float>(40 + N));

    eyes[j][i].righteye.point2x2[0][0] = Point(shape.at<float>(43), shape.at<float>(43 + N));
    eyes[j][i].righteye.point2x2[0][1] = Point(shape.at<float>(44), shape.at<float>(44 + N));
    eyes[j][i].righteye.point2x2[1][0] = Point(shape.at<float>(47), shape.at<float>(47 + N));
    eyes[j][i].righteye.point2x2[1][1] = Point(shape.at<float>(46), shape.at<float>(46 + N));

    eyes[j][i].lefteye.eyesize =
        (norm(eyes[j][i].lefteye.point2x2[0][0] - eyes[j][i].lefteye.point2x2[1][0]) +
            norm(eyes[j][i].lefteye.point2x2[0][1] - eyes[j][i].lefteye.point2x2[1][1])) / 2;

    eyes[j][i].righteye.eyesize =
        (norm(eyes[j][i].righteye.point2x2[0][0] - eyes[j][i].righteye.point2x2[1][0]) +
            norm(eyes[j][i].righteye.point2x2[0][1] - eyes[j][i].righteye.point2x2[1][1])) / 2;

    eyes[j][i].eyesize2 =
        (eyes[j][i].lefteye.eyesize + eyes[j][i].righteye.eyesize) / 2;

    int lx1 = eyes[j][i].lefteye.point2x2[0][0].x;
    int ly1 = eyes[j][i].lefteye.point2x2[0][0].y;
    int lx2 = eyes[j][i].lefteye.point2x2[0][1].x;
    int ly2 = eyes[j][i].lefteye.point2x2[0][1].y;
    int lx3 = eyes[j][i].lefteye.point2x2[1][0].x;
    int ly3 = eyes[j][i].lefteye.point2x2[1][0].y;
    int lx4 = eyes[j][i].lefteye.point2x2[1][1].x;
    int ly4 = eyes[j][i].lefteye.point2x2[1][1].y;

    int Lminx = min(min(lx1, lx2), min(lx3, lx4));
    int Lminy = min(min(ly1, ly2), min(ly3, ly4));
    int Lmaxx = max(max(lx1, lx2), max(lx3, lx4));
    int Lmaxy = max(max(ly1, ly2), max(ly3, ly4));

    eyes[j][i].lefteye.eyebox = Rect(Lminx, Lminy, Lmaxx - Lminx, Lmaxy - Lminy);
    eyes[j][i].lefteye.center = Point(Lminx + (Lmaxx - Lminx) / 2, Lminy + (Lmaxy - Lminy) / 2);

    int rx1 = eyes[j][i].righteye.point2x2[0][0].x;
    int ry1 = eyes[j][i].righteye.point2x2[0][0].y;
    int rx2 = eyes[j][i].righteye.point2x2[0][1].x;
    int ry2 = eyes[j][i].righteye.point2x2[0][1].y;
    int rx3 = eyes[j][i].righteye.point2x2[1][0].x;
    int ry3 = eyes[j][i].righteye.point2x2[1][0].y;
    int rx4 = eyes[j][i].righteye.point2x2[1][1].x;
    int ry4 = eyes[j][i].righteye.point2x2[1][1].y;

    int Rminx = min(min(rx1, rx2), min(rx3, rx4));
    int Rminy = min(min(ry1, ry2), min(ry3, ry4));
    int Rmaxx = max(max(rx1, rx2), max(rx3, rx4));
    int Rmaxy = max(max(ry1, ry2), max(ry3, ry4));

    eyes[j][i].righteye.eyebox = Rect(Rminx, Rminy, Rmaxx - Rminx, Rmaxy - Rminy);
    eyes[j][i].righteye.center = Point(Rminx + (Rmaxx - Rminx) / 2, Rminy + (Rmaxy - Rminy) / 2);

    {
        Rect box = eyes[j][i].lefteye.eyebox;
        Point c(box.x + box.width / 2, box.y + box.height / 2);
        int new_w = box.width * 3;
        int new_h = box.height * 2;
        int new_x = c.x - new_w / 2;
        int new_y = c.y - new_h / 2;
        eyes[j][i].lefteye.eyebox = Rect(new_x, new_y, new_w, new_h);
    }

    {
        Rect box = eyes[j][i].righteye.eyebox;
        Point c(box.x + box.width / 2, box.y + box.height / 2);
        int new_w = box.width * 3;
        int new_h = box.height * 2;
        int new_x = c.x - new_w / 2;
        int new_y = c.y - new_h / 2;
        eyes[j][i].righteye.eyebox = Rect(new_x, new_y, new_w, new_h);
    }
}
void CalcEyeSize(vector<Mat> crops) {

    ldmarkmodel modelt;
    string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) cin >> modelFilePath;

    vector<vector<Mat>> Landmarkimg(facenum, vector<Mat>(imgnum));

    eyes.resize(facenum);
    for (int j = 0; j < facenum; j++) eyes[j].resize(imgnum);

    face_center.clear();
    face_center.resize(facenum);

    for (int j = 0; j < facenum; j++) {
        for (int i = 0; i < imgnum; i++) {

            int idx = j * imgnum + i;

            Landmarkimg[j][i] = crops[idx].clone();
            Mat& crop = Landmarkimg[j][i];
            Mat gray, shape;

            cvtColor(crop, gray, COLOR_BGR2GRAY);
            modelt.track(crop, shape);
            if (shape.empty() || shape.cols < 68) continue;

            int N = shape.cols / 2;

            float sx = 0, sy = 0;
            int cnt = 0;

            for (int k = 0; k < N; k++) {
                float x = shape.at<float>(k);
                float y = shape.at<float>(k + N);
                if (x < 1 || y < 1 || x >= gray.cols - 1 || y >= gray.rows - 1) continue;
                sx += x;
                sy += y;
                cnt++;
                circle(crop, Point(x, y), 2, Scalar(0, 0, 255), -1);
            }

            if (cnt > 0) {
                Point c(sx / cnt, sy / cnt);
                face_center[j].push_back(c);
                if (mode == 'F') circle(crop, c, 4, Scalar(255, 0, 0), -1);
            }

            EyeDataUpdate(shape, i, j, N);

            if (mode == 'E') {
                rectangle(crop, eyes[j][i].lefteye.eyebox, Scalar(0, 255, 0), 2);
                rectangle(crop, eyes[j][i].righteye.eyebox, Scalar(0, 255, 0), 2);
            }

            putText(crop, "Eye: " + to_string(eyes[j][i].eyesize2),
                Point(5, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }

        Mat mergedimg = Merge(Landmarkimg[j]);
        imshow("Landmarks_Face" + to_string(j), mergedimg);
    }

    waitKey(1);
}


void FindMaxEyeIdx() {
    maxeyeidx.clear();
    maxeyeidx.resize(facenum);

    for (int j = 0; j < facenum; j++) {

        int maxvalue = -1;
        int maxindex = 0;

        for (int i = 0; i < imgnum; i++) {
            int v = eyes[j][i].eyesize2;
            if (v > maxvalue) {
                maxvalue = v;
                maxindex = i;
            }
        }

        maxeyeidx[j] = maxindex;
    }
}

void EyeGrabCut(vector<Rect> faces, vector<vector<Eye_pair>> eyes, int* maxeyeidx) {

    Mat grabcutfaces(imgs[0].size(), CV_8UC3, Scalar(255, 255, 255));

    for (int f = 0; f < facenum; f++) {

        int idx = maxeyeidx[f];
        Mat img = imgs[idx].clone();

        auto do_cut = [&](Rect local_eye, Point ref_center_idx, Point ref_center_0)
            {
                Rect face = faces[f];
                Rect r(face.x + local_eye.x, face.y + local_eye.y, local_eye.width, local_eye.height);
                r &= Rect(0, 0, img.cols, img.rows);

                Mat result, bg, fg;
                grabCut(img, result, r, bg, fg, 5, GC_INIT_WITH_RECT);
                compare(result, GC_PR_FGD, result, CMP_EQ);

                for (int y = r.y; y < r.y + r.height; y++) {
                    for (int x = r.x; x < r.x + r.width; x++) {

                        if (result.at<uchar>(y, x) != 0) {

                            grabcutfaces.at<Vec3b>(y, x) = img.at<Vec3b>(y, x);

                            int mapped_y = y - ref_center_idx.y + ref_center_0.y;
                            int mapped_x = x - ref_center_idx.x + ref_center_0.x;

                            if (mapped_y >= 0 && mapped_x >= 0 &&
                                mapped_y < imgs[0].rows && mapped_x < imgs[0].cols)
                            {
                                imgs[0].at<Vec3b>(mapped_y, mapped_x) = img.at<Vec3b>(y, x);
                            }
                        }
                    }
                }
            };

        do_cut(eyes[f][idx].lefteye.eyebox,
            eyes[f][idx].lefteye.center,
            eyes[f][0].lefteye.center);

        do_cut(eyes[f][idx].righteye.eyebox,
            eyes[f][idx].righteye.center,
            eyes[f][0].righteye.center);

        waitKey(1);
    }

    imshow("GrabCut_Face", grabcutfaces);
    imwrite("GrabCut_Face.jpg", grabcutfaces);
}


void FaceGrabCut(vector<Rect> faces, int* maxeyeidx) {

    Mat grabcutfaces(imgs[0].size(), CV_8UC3, Scalar(255, 255, 255));

    for (int f = 0; f < facenum; f++) {

        int idx = maxeyeidx[f];
        Mat img = imgs[idx].clone();

        Rect r = faces[f];

        Point source_center = face_center[f][idx];
        Point target_center = face_center[f][0];

        int dx = target_center.x - source_center.x;
        int dy = target_center.y - source_center.y;

        Mat result, bg, fg;
        grabCut(img, result, r, bg, fg, 5, GC_INIT_WITH_RECT);
        compare(result, GC_PR_FGD, result, CMP_EQ);


        for (int y = r.y; y < r.y + r.height; y++) {
            for (int x = r.x; x < r.x + r.width; x++) {

                if (result.at<uchar>(y, x) != 0) {

                    int new_x2 = x + dx;
                    int new_y2 = y + dy;

                    if (0 <= new_x2 && new_x2 < imgs[0].cols &&
                        0 <= new_y2 && new_y2 < imgs[0].rows) {

                        grabcutfaces.at<Vec3b>(new_y2, new_x2) = img.at<Vec3b>(y, x);
                        imgs[0].at<Vec3b>(new_y2, new_x2) = img.at<Vec3b>(y, x);
                    }
                }
            }
        }

        waitKey(1);
    }

    imshow("GrabCut_Face", grabcutfaces);
    imwrite("GrabCut_Face.jpg", grabcutfaces);
}

void main() {
    Readimg();

    while (1) {

        cout << "[F] Face Grabcut Mode\n[E] Eye Grabcut Mode\n";
        cin >> mode;
        if (mode == 'F') {
            vector<Rect> faces = FaceDetection(imgs[0]);
            facenum = faces.size();

            if (facenum == 0) {
                cout << "얼굴을 찾지 못했습니다.\n";
                return;
            }

            eyesizes = Mat(facenum, imgnum, CV_8UC1);

            vector<Mat> crops = CropFaces(imgs, faces);
            CalcEyeSize(crops);

            FindMaxEyeIdx();
            FaceGrabCut(faces, maxeyeidx.data());

            imshow("result", imgs[0]);
            imwrite("result.jpg", imgs[0]);

            waitKey(0);
        }
        else if (mode == 'E') {

            vector<Rect> faces = FaceDetection(imgs[0]);
            facenum = faces.size();

            if (facenum == 0) {
                cout << "얼굴을 찾지 못했습니다.\n";
                return;
            }

            vector<Mat> crops = CropFaces(imgs, faces);
            CalcEyeSize(crops);

            FindMaxEyeIdx();
            EyeGrabCut(faces, eyes, maxeyeidx.data());

            imshow("result", imgs[0]);
            imwrite("result.jpg", imgs[0]);

            waitKey(0);
        }
        else {
            cout << "다시 입력하세요\n";
        }
    }
}
