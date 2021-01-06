__kernel void convolution(
    __constant int *img_height,
    __constant int *img_width,
    __constant int *filter_width,
    __constant float *filter,
    __constant int *offset_x,
    __constant int *offset_y,
    __read_only __global float *in_img,
    __write_only __global float *out_img)
{
    const int x = get_global_id(0) + *offset_x;
    const int y = get_global_id(1) + *offset_y;

    float tmp = 0;
    int a, b;
    const int ws = (*filter_width) >> 1;

    if (*offset_x == *offset_y && *offset_x == ws) {
        for (int j = 0; j < *filter_width; ++j) {
            for (int i = 0; i < *filter_width; ++i) {
                a = x + i - ws;
                b = y + j - ws;
                tmp += filter[j*(*filter_width) + i] * in_img[b*(*img_width) + a];
            }
        }
    }
    else {
        for (int j = 0; j < *filter_width; ++j) {
            for (int i = 0; i < *filter_width; ++i) {
                a = x + i - ws;
                b = y + j - ws;
                if ((0 <= a && a < *img_width) && (0 <= b && b < *img_height)) {
                    tmp += filter[j*(*filter_width) + i] * in_img[b*(*img_width) + a];
                }
            }
        }
    }

    out_img[y*(*img_width) + x] = tmp;
}
