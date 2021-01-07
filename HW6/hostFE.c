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
    clEnqueueWriteBuffer(command_queue, img_height_mem, CL_TRUE, 0,
                         sizeof(int), &imageHeight, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, img_width_mem, CL_TRUE, 0,
                         sizeof(int), &imageWidth, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, filter_width_mem, CL_TRUE, 0,
                         sizeof(int), &filterWidth, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, filter_mem, CL_TRUE, 0,
                         filterSize * sizeof(float), filter,
                         0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, offset_x_mem, CL_TRUE, 0,
                         sizeof(int), &offset_x[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, offset_y_mem, CL_TRUE, 0,
                         sizeof(int), &offset_y[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, in_mem, CL_TRUE, 0, 
                         imageSize * sizeof(float), inputImage,
                         0, NULL, NULL);

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_height_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_width_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_width_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&filter_mem);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &offset_x_mem);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &offset_y_mem);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&in_mem);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&out_mem);

    // Execute the OpenCL kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                           global_work_size[0], 0, NULL, NULL, NULL);

    // ------------------------------------------------------------------------
    // Now deal with upper, bottom, left and right parts.
    // ------------------------------------------------------------------------
    for (int i = 1; i < 5; ++i) {
        clEnqueueWriteBuffer(command_queue, offset_x_mem, CL_TRUE, 0,
                             sizeof(int), &offset_x[i], 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, offset_y_mem, CL_TRUE, 0,
                             sizeof(int), &offset_y[i], 0, NULL, NULL);

        // Set the arguments of the kernel
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_height_mem);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_width_mem);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &filter_width_mem);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&filter_mem);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &offset_x_mem);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &offset_y_mem);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&in_mem);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&out_mem);

        clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                               global_work_size[i], 0, NULL, NULL, NULL);
    }

    // Read the memory buffer C on the device to the local variable C
    clEnqueueReadBuffer(command_queue, out_mem, CL_TRUE, 0, 
                        imageSize * sizeof(float), outputImage,
                        0, NULL, NULL);

    // Clean up
    clFinish(command_queue);
    clReleaseKernel(kernel);
    clReleaseMemObject(img_height_mem);
    clReleaseMemObject(img_width_mem);
    clReleaseMemObject(filter_width_mem);
    clReleaseMemObject(filter_mem);
    clReleaseMemObject(offset_x_mem);
    clReleaseMemObject(offset_y_mem);
    clReleaseMemObject(in_mem);
    clReleaseMemObject(out_mem);
    clReleaseCommandQueue(command_queue);
}
