#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <pthread.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
#define NUM_KP 1500
#define PTHREAD
#define nthread 16
int n_times=100000;

struct KP{
    double x1;
    double y1;
    double x2;
    double y2;
};

typedef struct
{
    int thread_id;
    int step;
    int len;
    double **CorList;
} Arg; // 傳入 thread 的參數型別

typedef struct
{
    int thread_id;
    int start;
    int end;
    const uint8_t *img;
    int *img_shape;
    double **homography;
    uint8_t *out_img;
    int *output_shape;
} Arg2; // 傳入 thread 的參數型別

typedef struct
{
    int thread_id;
    int start;
    int end;
    const uint8_t *img;
    int *img_shape;
    int offset;
    uint8_t *out_img;
    int *output_shape;
} Arg3; // 傳入 thread 的參數型別

int index_array[16][4];
int line_array[16];
int su_rand(unsigned int seed)
{
    int32_t val = ((seed * 1103515245U) + 12345U) & 0x7fffffff;
    return val;
}
double GeoDis(double *points, double **H,Mat homography_matrix)
{
    double Estm[3];
    Estm[2] = points[0]*H[2][0] + points[1]*H[2][1] + 1*H[2][2];

    Estm[0] = 1/Estm[2]*(points[0]*H[0][0] + points[1]*H[0][1] + 1*H[0][2]);
    Estm[1] = 1/Estm[2]*(points[0]*H[1][0] + points[1]*H[1][1] + 1*H[1][2]);
    Estm[2] = 1;

    //norm 平方
    double err = pow(points[2]-Estm[0], 2) + pow(points[3]-Estm[1], 2);

    return err;
}
Mat CalH(double **RanPoints, double **H)
{
    vector<cv::Point2f> obj;
    vector<cv::Point2f> scene;

    cv::Point2f A,A_P;

    for (int i = 0; i < 4; ++i)
    {
        A.x = RanPoints[i][0];
        A.y = RanPoints[i][1];
        A_P.x=RanPoints[i][2];
        A_P.y=RanPoints[i][3];
        obj.push_back(A);
        scene.push_back(A_P);
    }

    //find Homography
    Mat homography_matrix= getPerspectiveTransform(obj,scene);

    //copy Mat to **H
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            H[i][j] = homography_matrix.at<double>(i, j);
        }
    }

    return homography_matrix;

}       

void *cal(void *para){
    unsigned int seed = time(NULL);
    // cout<<"seed = "<<seed<<endl;
    Arg *thread_arg = (Arg*)para;
    int max_line = 0;
    double **Cor;
    Cor = new double*[4];
    for (int i = 0; i < 4; ++i) {
        Cor[i] = new double[4];
    }   
    for (int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            Cor[i][j] = 0.0;
    for (int i = 0; i < thread_arg->step; ++i){
        int Lines = 0;
        int t1, t2, t3, t4;
        for (int j = 0; j < 4; ++j){
            seed = seed*(thread_arg->thread_id*thread_arg->step+i+1);
            // cout<<"seed = "<<seed<<endl;

            seed = su_rand(seed);
            int rnad_Num = seed%thread_arg->len;
            // cout<<"seed = "<<seed<<endl;
            // cout<<"rand = "<<rnad_Num<<endl;
            Cor[j][0] = thread_arg->CorList[rnad_Num][0];
            Cor[j][1] = thread_arg->CorList[rnad_Num][1];
            Cor[j][2] = thread_arg->CorList[rnad_Num][2];
            Cor[j][3] = thread_arg->CorList[rnad_Num][3];
            if(j==0)
                t1 = rnad_Num;
            if(j==1)
                t2 = rnad_Num;
            if(j==2)
                t3 = rnad_Num;
            if(j==3)
                t4 = rnad_Num;
        }

        // cal H
        double **H;
        H = new double*[3];
        for (int i = 0; i < 3; ++i) {
            H[i] = new double[3];
        }   
        for (int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                H[i][j] = 0.0;

        Mat homography_matrix;
        homography_matrix = CalH(Cor, H);
        for (int j = 0; j < thread_arg->len; ++j)	{		
            double dis;
            dis = GeoDis(&thread_arg->CorList[j][0], H, homography_matrix);
            if (dis < 9){
                Lines++;
            }
        }
        if(Lines > max_line){
            max_line = Lines;
            index_array[thread_arg->thread_id][0] = t1;
            index_array[thread_arg->thread_id][1] = t2;
            index_array[thread_arg->thread_id][2] = t3;
            index_array[thread_arg->thread_id][3] = t4;
            line_array[thread_arg->thread_id] = Lines;
        }
    }
}
void cal_ransac(double **CorList, int Clen, double **AnsH)
{
    
    unsigned int seed = time(NULL);
    #ifdef PTHREAD
    pthread_t threadID[nthread];
    Arg thread_arg[nthread];
    int iter = n_times / nthread;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i = 0; i < nthread; i++){
        thread_arg[i].thread_id = i;
        thread_arg[i].step = iter;
        thread_arg[i].len = Clen;
        thread_arg[i].CorList = CorList;
        pthread_create(&threadID[i], NULL, cal, (void *)&thread_arg[i]);
    }
    for(int i = 0; i < nthread; i++){
        pthread_join(threadID[i], NULL);
    }
    pthread_attr_destroy(&attr);
    int temp_max = 0;
    int temp_index = 0;
    for(int i = 0; i < nthread; i++){
        if (line_array[i] > temp_max){
            temp_max = line_array[i];
            temp_index = i;
        }
    }
    double **Cor;
    Cor = new double*[4];
    for (int i = 0; i < 4; ++i) {
        Cor[i] = new double[4];
    }   
    for (int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            Cor[i][j] = 0.0;
    for(int i = 0; i<4;i++)
        for(int j=0;j<4;j++)
            Cor[i][j] = CorList[index_array[temp_index][i]][j];
    double **H;
    H = new double*[3];
    for (int i = 0; i < 3; ++i) {
        H[i] = new double[3];
    }   
    for (int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            H[i][j] = 0.0;

    Mat homography_matrix;
    homography_matrix = CalH(Cor, H);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            AnsH[i][j] = H[i][j];
    #endif

    #ifndef PTHREAD
    for (int i = 0; i < n_times; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            seed = seed*(i+1)*(j+1);
            int rnad_Num = su_rand(seed)%Clen;
            seed = su_rand(seed);
            Cor[j][0] = CorList[rnad_Num][0];
            Cor[j][1] = CorList[rnad_Num][1];
            Cor[j][2] = CorList[rnad_Num][2];
            Cor[j][3] = CorList[rnad_Num][3];
        }

        // cal H
        double **H;
        H = new double*[3];
        for (int i = 0; i < 3; ++i) {
            H[i] = new double[3];
        }   
        for (int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                H[i][j] = 0.0;

        Mat homography_matrix;
        homography_matrix = CalH(Cor, H);
        int Lines = 0;
        for (int j = 0; j < Clen; ++j)	{		
            double dis;
            dis = GeoDis(&CorList[j][0], H, homography_matrix);
            if (dis < 9){
                Lines++;
            }
        }
        // #pragma omp critical
        // {
        if (Lines > Max_Lines){
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++){
                    AnsH[i][j] = H[i][j];
                }
            }
            Max_Lines = Lines;
        }
        // }

        for (int i = 0; i < 3; i++) 
        {
                delete H[i];
        }
        delete H;
    }
    #endif
    for (int i = 0; i < 4; i++) 
    {
            delete Cor[i];
    }
    delete Cor;

}

double** sift_match(string imname1, string imname2){
    // 讀取圖
    const Mat imp_1 = imread(imname1, 0);
    const Mat imp_2 = imread(imname2, 0);

    //Detect the keypoints:
    std::vector<KeyPoint> keypoints_1, keypoints_2;   
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(); 
    f2d->detect( imp_1, keypoints_1 );
    f2d->detect(imp_2, keypoints_2);

    //Calculate descriptors (feature vectors)    
    Mat descriptors_1, descriptors_2;    
    f2d->compute( imp_1, keypoints_1, descriptors_1 );
    f2d->compute( imp_2, keypoints_2, descriptors_2 );

    // Matching descriptors using Brute Force
    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    
    //-- Quick calculation of max and min distances between Keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < descriptors_1.rows; i++ ){
        double dist = matches[i].distance;
        if( dist < min_dist ) 
            min_dist = dist;
        if( dist > max_dist ) 
            max_dist = dist;
    }   

    // keep only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //std::vector< DMatch > good_matches;
    // KP CorList[300];
    double **CorList;
    CorList = new double*[NUM_KP]; // dynamic array (size 10) of pointers to int

    for (int i = 0; i < NUM_KP; ++i) {
        CorList[i] = new double[4];
    // each i-th pointer is now pointing to dynamic array (size 10) of actual int values
    }   
    for (int i = 0; i < NUM_KP; ++i)
        for(int j = 0; j < 4; ++j)
            CorList[i][j] = 0.0;
    int num_kp = 0;
    double threshold = 150.0;
    for( int i = 0; i < descriptors_1.rows; i++ ){
        if( matches[i].distance <= max(2*min_dist, threshold) ){ 
            //good_matches.push_back(matches[i]);
            CorList[num_kp][0] = keypoints_1[matches[i].queryIdx].pt.x;
            CorList[num_kp][1] = keypoints_1[matches[i].queryIdx].pt.y;
            CorList[num_kp][2] = keypoints_2[matches[i].trainIdx].pt.x;
            CorList[num_kp][3] = keypoints_2[matches[i].trainIdx].pt.y;
            num_kp++;
        }
    }
    return CorList;

}

#ifndef PTHREAD
void* warp(
    const uint8_t *img,
    int *img_shape,
    double **homography,
    uint8_t *out_img,
    int *output_shape)
{
    // Create output image
    //uint8_t *out_img = (uint8_t *)malloc(output_shape[0] * output_shape[1] *
    //                                    output_shape[2] * sizeof(uint8_t));

    // Backproject pixels on @out_img to @img
 
    for (int row = 0; row < output_shape[0]; row++) {
        for (int col = 0; col < output_shape[1]; col++) {
            float pj_x, pj_y;   // back projected coordinate of x and y
            int pj_xi, pj_yi;
            float pj_factor;

            // Unroll matrix multiplication of (@homography * [row, col, 1])
            // Prospective factor
            pj_factor = homography[2][0]*(col-img_shape[1]) + homography[2][1]*row +
                        homography[2][2];
            // Back projected x
            pj_y = (
                homography[0][0]*(col-img_shape[1]) + homography[0][1]*row + homography[0][2]
            ) / pj_factor;
            // Back projected y
            pj_x = (
                homography[1][0]*(col-img_shape[1]) + homography[1][1]*row + homography[1][2]
            ) / pj_factor;

            // Apply bilinear interpolation and fill the value in @out_img
            pj_xi = (int)pj_x;
            pj_yi = (int)pj_y;
            if ((0 <= pj_xi && pj_xi+1 < img_shape[0])
                && (0 <= pj_yi && pj_yi+1 < img_shape[1]))
            {
                float lt_factor = ((pj_xi+1) - pj_x) * ((pj_yi+1) - pj_y);
                float lb_factor = (pj_x - pj_xi) * ((pj_yi+1) - pj_y);
                float rt_factor = ((pj_xi+1) - pj_x) * (pj_y - pj_yi);
                float rb_factor = (pj_x - pj_xi) * (pj_y - pj_yi);
                for (int channel = 0; channel < img_shape[2]; channel++) {
                    out_img[row*output_shape[1]*output_shape[2] + col*output_shape[2] + channel] = \
                        img[pj_xi*img_shape[1]*img_shape[2] + pj_yi*img_shape[2] + channel] * lt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + pj_yi*img_shape[2] + channel] * lb_factor +
                        img[pj_xi*img_shape[1]*img_shape[2] + (pj_yi+1)*img_shape[2] + channel] * rt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + (pj_yi+1)*img_shape[2] + channel] * rb_factor;
                }
            }
            else {
                for (int channel = 0; channel < img_shape[2]; channel++) {
                    out_img[row*output_shape[1]*output_shape[2] + col*output_shape[2] + channel] = 0;
                }
            }
        }
    }
    
    return (void *)out_img;
}
#endif

#ifdef PTHREAD
void* warp(void *para)
{
    Arg2 *thread_arg = (Arg2*)para;
    const u_int8_t *img = thread_arg->img;
    int *img_shape = thread_arg->img_shape;
    double **homography = thread_arg->homography;
    uint8_t *out_img = thread_arg->out_img;
    int *output_shape = thread_arg->output_shape;
    for (int row = thread_arg->start; row < thread_arg->end; row++) {
        for (int col = 0; col < output_shape[1]; col++) {
            float pj_x, pj_y;   // back projected coordinate of x and y
            int pj_xi, pj_yi;
            float pj_factor;

            // Unroll matrix multiplication of (@homography * [row, col, 1])
            // Prospective factor
            pj_factor = homography[2][0]*(col-img_shape[1]) + homography[2][1]*row +
                        homography[2][2];
            // Back projected x
            pj_y = (
                homography[0][0]*(col-img_shape[1]) + homography[0][1]*row + homography[0][2]
            ) / pj_factor;
            // Back projected y
            pj_x = (
                homography[1][0]*(col-img_shape[1]) + homography[1][1]*row + homography[1][2]
            ) / pj_factor;

            // Apply bilinear interpolation and fill the value in @out_img
            pj_xi = (int)pj_x;
            pj_yi = (int)pj_y;
            if ((0 <= pj_xi && pj_xi+1 < img_shape[0])
                && (0 <= pj_yi && pj_yi+1 < img_shape[1]))
            {
                float lt_factor = ((pj_xi+1) - pj_x) * ((pj_yi+1) - pj_y);
                float lb_factor = (pj_x - pj_xi) * ((pj_yi+1) - pj_y);
                float rt_factor = ((pj_xi+1) - pj_x) * (pj_y - pj_yi);
                float rb_factor = (pj_x - pj_xi) * (pj_y - pj_yi);
                for (int channel = 0; channel < img_shape[2]; channel++) {
                    out_img[row*output_shape[1]*output_shape[2] + col*output_shape[2] + channel] = \
                        img[pj_xi*img_shape[1]*img_shape[2] + pj_yi*img_shape[2] + channel] * lt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + pj_yi*img_shape[2] + channel] * lb_factor +
                        img[pj_xi*img_shape[1]*img_shape[2] + (pj_yi+1)*img_shape[2] + channel] * rt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + (pj_yi+1)*img_shape[2] + channel] * rb_factor;
                }
            }
            else {
                for (int channel = 0; channel < img_shape[2]; channel++) {
                    out_img[row*output_shape[1]*output_shape[2] + col*output_shape[2] + channel] = 0;
                }
            }
        }
    }
    
}

void* copy_img2out_img(void *para) {
    Arg3 *thread_arg = (Arg3*)para;
    const u_int8_t *img = thread_arg->img;
    int *img_shape = thread_arg->img_shape;
    int offset = thread_arg->offset;
    uint8_t *out_img = thread_arg->out_img;
    int *output_shape = thread_arg->output_shape;

    for (int i = thread_arg->start; i < thread_arg->end; i++) {
        for (int j = 0; j < img_shape[1]; j++) {
            for (int c = 0; c < img_shape[2]; c++) {
                if (out_img[i*output_shape[1]*output_shape[2] + (j+offset)*output_shape[2] + c] == 0)
                    out_img[i*output_shape[1]*output_shape[2] + (j+offset)*output_shape[2] + c] = img[i*img_shape[1]*img_shape[2] + j*img_shape[2] + c];
            }
        }
    }
}
#endif

int main(int argc, const char* argv[])
{
    // 放右邊的圖
    string imname1 = "le2.jpg";
    // 放左邊的圖
    string imname2 = "ri.jpg";
    // 讀取兩張圖的SIFT並做brute force match
    double ** CorList = sift_match(imname1, imname2);
    // 計算特徵點的數量
    int c_len = 0;
    for(; c_len < NUM_KP; c_len++){
        if(CorList[c_len][0]<=0)
            break;
    }

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // RANSAC
    double **ans_H;
    ans_H = new double*[3];
    for (int i = 0; i < 3; ++i) {
        ans_H[i] = new double[3];
    }
    cal_ransac(CorList, c_len, ans_H);

    // -----> RANSAC time
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << "RANSAC execution time: " << setprecision(6) << elapsed << endl;
    // <-----

    // Warp images
    const Mat img1 = imread("le2.jpg", IMREAD_COLOR); // Load as grayscale
    const Mat img2 = imread("ri.jpg", IMREAD_COLOR); // Load as grayscale
    Mat H_inv = (Mat_<double>(3,3) << ans_H[0][0], ans_H[0][1], ans_H[0][2],
                                      ans_H[1][0], ans_H[1][1], ans_H[1][2],
                                      ans_H[2][0], ans_H[2][1], ans_H[2][2]);
    int *img_shape = (int *)malloc(3 * sizeof(int));
    int *output_shape = (int *)malloc(3 * sizeof(int));

    H_inv = H_inv.inv();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ans_H[i][j] = H_inv.at<double>(i, j);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &start);

    img_shape[0] = img1.rows;
    img_shape[1] = img1.cols;
    img_shape[2] = img1.channels();
    output_shape[0] = img1.rows;
    output_shape[1] = img1.cols + img2.cols;
    output_shape[2] = img1.channels();

    // Create output image
    uint8_t *out_img = (uint8_t *)malloc(output_shape[0] * output_shape[1] *
                                         output_shape[2] * sizeof(uint8_t));

    
    pthread_t threadID[nthread];
    Arg2 thread_arg[nthread];
    Arg3 thread_arg2[nthread];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    int step = output_shape[0] / nthread;
    
    for(int i = 0; i < nthread; i++){
        thread_arg[i].thread_id = i;
        thread_arg[i].start = i * step;
        thread_arg[i].end = (i + 1) * step;
        thread_arg[i].img = (const uint8_t *)img1.data;
        thread_arg[i].img_shape = img_shape;
        thread_arg[i].homography = ans_H;
        thread_arg[i].out_img = out_img;
        thread_arg[i].output_shape = output_shape;
        if( i == (nthread -1))
            thread_arg[i].end = output_shape[0];
        pthread_create(&threadID[i], &attr, warp, (void *)&thread_arg[i]);
    }
    // Stitch image
    // out_img = (uint8_t *)(warp((const uint8_t *)img1.data, img_shape, ans_H, out_img, output_shape));

    for(int i = 0; i < nthread; i++){
        pthread_join(threadID[i], NULL);
    }

    img_shape[0] = img2.rows;
    img_shape[1] = img2.cols;
    img_shape[2] = img2.channels();
    step = img2.rows / nthread;
    for(int i = 0; i < nthread; i++){
        thread_arg2[i].thread_id = i;
        thread_arg2[i].start = i * step;
        thread_arg2[i].end = (i + 1) * step;
        thread_arg2[i].img = (const uint8_t *)img2.data;
        thread_arg2[i].img_shape = img_shape;
        thread_arg2[i].offset = img1.cols;
        thread_arg2[i].out_img = out_img;
        thread_arg2[i].output_shape = output_shape;
        if( i == (nthread -1))
            thread_arg2[i].end = img2.rows;
        pthread_create(&threadID[i], &attr, copy_img2out_img,
                       (void *)&thread_arg2[i]);
    }

    // 回收性質設定
    pthread_attr_destroy(&attr);

    // Copy @img2 to output image
    /*for (int i = 0; i < img2.rows; i++) {
        for (int j = 0; j < img2.cols; j++) {
            for (int c = 0; c < img2.channels(); c++) {
                if (out_img[i*output_shape[1]*output_shape[2] + (j+img1.cols)*output_shape[2] + c] == 0)
                    out_img[i*output_shape[1]*output_shape[2] + (j+img1.cols)*output_shape[2] + c] = img2.data[i*img2.cols*img2.channels() + j*img2.channels() + c];
            }
        }
    }*/

    for(int i = 0; i < nthread; i++){
        pthread_join(threadID[i], NULL);
    }

    // -----> Warping time
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    cout << "Warping execution time: " << setprecision(6) << elapsed << endl;
    // <-----

    // Draw output image
    Mat result(output_shape[0], output_shape[1], CV_8UC3, (void *)out_img);
    imwrite("pthread_warp_result.jpg", result);

    // Free memory
    free(img_shape);
    free(output_shape);
    free(out_img);
    for (int i = 0; i < c_len; i++) {
        free(CorList[i]);
    }
    free(CorList);
    for (int i = 0; i < 3; i++) {
        delete ans_H[i];
    }
    delete ans_H;

    return 0;
}
