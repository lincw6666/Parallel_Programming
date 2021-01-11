Implement Parallelization on Image Stitching
===

Group 02
Member: 林正偉、陳則佑、許芳瑀

---

# Introduction

This repository contains the implementation of image stitching using SIMD, OpenMP and Pthread.

In image stitching, we use RANSAC to find the best homography, which maps keypoints from one image to another image. Then, by
using homography, we stitch one image to another image. We call this procedure "warping".

In this project, we'll focus on parallelizing RANSAC and warping. For more details, please refer to our report.

---

# Requirements

- GCC 5.4.0 or above
- OpenMP 4.0 or above
- OpenCV 3.2.0 or **BELOW** (Since newer version does not support SIFT for free)
- Pthread 2.23 or above
- Test successfully on Ubuntu 16.04 LTS

---

# Build
```
make [omp|simd|pthread]
```

- Options
  - `omp`: Implementation using OpenMP
  - `simd`: Implementation using SIMD
  - `pthread`: Implementation using Pthread
  - Without arguments: The serial version
- The name of executable files are the same as the arguments. `main` if there is no arguments.

---

# Usage
1. Put two images under the same directory with the executable file.
2. Name the left input image "le2.jpg" and name the right input image "ri.jpg".
3. Run the executable file with no arguments. Such as `./omp`, `./simd`, or `./pthread`.
