#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
    const int halfFilterWidth = filterWidth >> 1;

    // ------------------------------------------------------------------------
    // Split image into 5 parts: middle, upper, bottom, left, right
    //
    // For the middle part, we DON'T need to check whether the neighbors of a
    // pixel exceed the boundary (imageHeight && imageWidth).
    // For the rest parts, we SHOULD check whether the neighbors of a pixel
    // exceed the boundary.
    // ------------------------------------------------------------------------

    // Set the parameters for each part.
    size_t global_work_size[5][2] = {
        {(size_t) imageWidth-filterWidth+1, (size_t) imageHeight-filterWidth+1},
        {(size_t) imageWidth, (size_t) halfFilterWidth},
        {(size_t) imageWidth, (size_t) halfFilterWidth},
        {(size_t) halfFilterWidth, (size_t) imageHeight-filterWidth+1},
        {(size_t) halfFilterWidth, (size_t) imageHeight-filterWidth+1}
    };
    int offset_x[5], offset_y[5];   // 0: middle, 1: upper, 2: bottom, 3: left,
                                    // 4: right
    offset_x[0] = halfFilterWidth, offset_y[0] = halfFilterWidth;
    offset_x[1] = 0, offset_y[1] = 0;
    offset_x[2] = 0, offset_y[2] = imageHeight - halfFilterWidth;
    offset_x[3] = 0, offset_y[3] = halfFilterWidth;
    offset_x[4] = imageWidth - halfFilterWidth, offset_y[4] = halfFilterWidth;

    // ------------------------------------------------------------------------
    // Create some OpenCL stuffs.
    // ------------------------------------------------------------------------

    // Create a command queue.
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device,
                                                          0, &status);

    // Create memory buffers on the device.
    cl_mem img_height_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                           sizeof(int), NULL, &status);
    cl_mem img_width_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                          sizeof(int), NULL, &status);
    cl_mem filter_width_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                             sizeof(int), NULL, &status);
    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                       filterSize * sizeof(float),
                                       NULL, &status);
    cl_mem offset_x_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                         sizeof(int), NULL, &status);
    cl_mem offset_y_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                         sizeof(int), NULL, &status);
    cl_mem in_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY,
                                   imageSize * sizeof(float),
                                   NULL, &status);
    cl_mem out_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                    imageSize * sizeof(float),
                                    NULL, &status);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // ------------------------------------------------------------------------
    // Deal with the middle part.
    // ------------------------------------------------------------------------

    // Copy data to their respective memory buffers.
    status = clEnqueueWriteBuffer(command_queue, img_height_mem, CL_TRUE, 0,
                                  sizeof(int), &imageHeight, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, img_width_mem, CL_TRUE, 0,
                                  sizeof(int), &imageWidth, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, filter_width_mem, CL_TRUE, 0,
                                  sizeof(int), &filterWidth, 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0,
                                  filterSize * sizeof(float), filter,
                                  0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, offset_x_mem, CL_TRUE, 0,
                                  sizeof(int), &offset_x[0], 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, offset_y_mem, CL_TRUE, 0,
                                  sizeof(int), &offset_y[0], 0, NULL, NULL);
    printf("%d ", status);
    status = clEnqueueWriteBuffer(command_queue, in_mem, CL_TRUE, 0, 
                                  imageSize * sizeof(float), inputImage,
                                  0, NULL, NULL);
    printf("%d ", status);

    // Set the arguments of the kernel
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_height_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_width_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_width_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&filter_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &offset_x_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &offset_y_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&in_mem);
    printf("%d ", status);
    status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&out_mem);
    printf("%d ", status);

    // Execute the OpenCL kernel
    status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                    global_work_size[0], 0, NULL, NULL, NULL);
    printf("%d ", status);

    // ------------------------------------------------------------------------
    // Now deal with upper, bottom, left and right parts.
    // ------------------------------------------------------------------------
    for (int i = 1; i < 5; ++i) {
        status = clEnqueueWriteBuffer(command_queue, offset_x_mem, CL_TRUE, 0,
                                      sizeof(int), &offset_x[i], 0, NULL, NULL);
        printf("%d ", status);
        status = clEnqueueWriteBuffer(command_queue, offset_y_mem, CL_TRUE, 0,
                                      sizeof(int), &offset_y[i], 0, NULL, NULL);
        printf("%d ", status);

        // Set the arguments of the kernel
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_height_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_width_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_width_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&filter_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &offset_x_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &offset_y_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&in_mem);
        printf("%d ", status);
        status = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&out_mem);
        printf("%d ", status);

        status = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                                        global_work_size[i], 0, NULL, NULL, NULL);
        printf("%d ", status);
    }

    // Read the memory buffer C on the device to the local variable C
    status = clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0, 
                                 imageSize * sizeof(float), outputImage,
                                 0, NULL, NULL);
    printf("%d ", status);

    // Clean up
    status = clFinish(command_queue);
    printf("%d\n", status);
    status = clReleaseKernel(kernel);
    status = clReleaseMemObject(img_height_mem);
    status = clReleaseMemObject(img_width_mem);
    status = clReleaseMemObject(filter_width_mem);
    status = clReleaseMemObject(filter_mem);
    status = clReleaseMemObject(offset_x_mem);
    status = clReleaseMemObject(offset_y_mem);
    status = clReleaseMemObject(in_mem);
    status = clReleaseMemObject(out_mem);
    status = clReleaseCommandQueue(command_queue);
}
