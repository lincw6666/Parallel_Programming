#include <iostream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace cv::xfeatures2d;


void* warp(
    const uint8_t *img,
    int *img_shape,
    double *homography,
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
            pj_factor = homography[6]*col + homography[7]*row +
                        homography[8];
            // Back projected x
            pj_y = (
                homography[0]*col + homography[1]*row + homography[2]
            ) / pj_factor;
            // Back projected y
            pj_x = (
                homography[3]*col + homography[4]*row + homography[5]
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
                    out_img[row*output_shape[1]*output_shape[2] + col + channel] = \
                        img[pj_xi*img_shape[1]*img_shape[2] + pj_yi + channel] * lt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + pj_yi + channel] * lb_factor +
                        img[pj_xi*img_shape[1]*img_shape[2] + pj_yi+1 + channel] * rt_factor +
                        img[(pj_xi+1)*img_shape[1]*img_shape[2] + pj_yi+1 + channel] * rb_factor;
                }
            }
            /*else {
                for (int channel = 0; channel < img_shape[2]; channel++) {
                    out_img[row*output_shape[1]*output_shape[2] + col + channel] = 0;
                }
            }*/
        }
    }

    return (void *)out_img;
}

void debug(void)
{
    const Mat img1 = imread("le2.jpg", 0); // Load as grayscale
    const Mat img2 = imread("ri.jpg", 0); // Load as grayscale

    // Keypoints detection
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    detector->detectAndCompute(img1, Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img2, Mat(), keypoints_2, descriptors_2);

    // Add results to image and save.
    /*Mat output;
    drawKeypoints(input, keypoints, output);
    imwrite("sift_result.jpg", output);*/

    // Keypoints matching
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    double max_dist = 0; double min_dist = 100;

    // Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    // Find good matches
    std::vector<DMatch> good_matches;

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= 3*min_dist) {
            good_matches.push_back(matches[i]);
        }
    }

    // Draw good matches
    /*Mat img_matches;
    drawMatches(img1, keypoints_1, img2, keypoints_2,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imwrite("keypoints_matching_results.jpg", img_matches);*/

    // Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ ) {
        // Get the keypoints from the good matches
        obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
    }

    struct timespec start, finish;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &start);

    Mat H = findHomography(obj, scene, RANSAC, 3, noArray(), 100000, 0.999999);

    // -----> RANSAC time
    clock_gettime(CLOCK_MONOTONIC, &finish);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout << "RANSAC execution time: " << std::setprecision(6) << elapsed << std::endl;
    // <-----

    Mat H_inv = H.inv();
    double *_H = (double *)malloc(H_inv.rows * H_inv.cols * sizeof(double));

    // Convert @H to uint8_t array
    for (int row = 0; row < H_inv.rows; row++) {
        for (int col = 0; col < H_inv.cols; col++) {
            _H[row*H_inv.cols + col] = H_inv.at<double>(row, col);
        }
    }

    // Warp images
    int *img_shape = (int *)malloc(3 * sizeof(int));
    int *output_shape = (int *)malloc(3 * sizeof(int));

    img_shape[0] = img1.rows;
    img_shape[1] = img1.cols;
    img_shape[2] = img1.channels();
    output_shape[0] = img1.rows;
    output_shape[1] = img1.cols; // + img2.cols;
    output_shape[2] = img1.channels();

    // Create output image
    uint8_t *out_img = (uint8_t *)malloc(output_shape[0] * output_shape[1] *
                                         output_shape[2] * sizeof(uint8_t));

    // Copy @img2 to output image
    for (int i = 0; i < img2.rows; i++) {
        for (int j = 0; j < img2.cols; j++) {
            out_img[i*output_shape[1] + j] = img2.data[i*img2.cols + j];
        }
    }

    // Stitch image
    out_img = (uint8_t *)(warp((const uint8_t *)img1.data, img_shape, _H, out_img, output_shape));

    // Draw output image
    Mat result(output_shape[0], output_shape[1], CV_8U, (void *)out_img);
    //Mat result = Mat::zeros(output_shape[0], output_shape[1], img1.type());
    //warpPerspective(img1, result, H, img1.size());
    imwrite("warp_result.jpg", result);

    // Free memory
    free(img_shape);
    free(output_shape);
    free(out_img);
    free(_H);
}

int main() {
    debug();
    return 0;
}
