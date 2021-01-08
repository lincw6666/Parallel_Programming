all:
	g++ -g3 -O3 main.cpp `pkg-config --cflags --libs opencv` -lutil -lboost_iostreams -lboost_system -lboost_filesystem -lopencv_xfeatures2d -o surftestexecutable -o main

warp: warp.cpp
	g++ -g3 -O3 warp.cpp `pkg-config --cflags --libs opencv` -o warp

omp: omp.cpp
	g++ -g3 -O3 omp.cpp `pkg-config --cflags --libs opencv` -fopenmp -o omp

simd: simd.cpp
	g++ -g3 -O3 simd.cpp -mavx2 `pkg-config --cflags --libs opencv` -o simd

cuda: cuda.cpp
	g++ -g3 -O3 cuda.cpp `pkg-config --cflags --libs opencv` -o cuda --lpthread

pthread: pthread.cpp
	g++ -g3 -O3 pthread.cpp `pkg-config --cflags --libs opencv` -lpthread -o pthread

cuda_cu: cuda.cu
	nvcc -g -O3 `pkg-config --cflags --libs opencv` -o cuda_cu cuda.cu

pthread_yo: pthread_yo.cpp
	g++ -g3 -O3 pthread_yo.cpp `pkg-config --cflags --libs opencv` -lpthread -o pthread_yo
