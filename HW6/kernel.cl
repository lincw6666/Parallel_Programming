__kernel void convolution(
    __global int *img_height,
    __global int *img_width,
    __global int *filter_width,
    __global float *filter,
    __global float *in_img,
    __global float *out_img)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float tmp = 0;
    int a, b;
    const int ws = (*filter_width) >> 1;

    for (int j = 0; j < *filter_width; ++j) {
        for (int i = 0; i < *filter_width; ++i) {
            a = x + i - ws;
            b = y + j - ws;
            if ((0 <= a && a < *img_width) && (0 <= b && b < *img_height)) {
                tmp += filter[j*(*filter_width) + i] * in_img[b*(*img_width) + a];
            }
        }
    }

    out_img[y*(*img_width) + x] = tmp;
}
