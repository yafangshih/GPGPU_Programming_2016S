nvcc -std=c++11 -arch=sm_30 -O -c lab2.cu -o lab2.o `pkg-config opencv --cflags --libs`
nvcc -std=c++11 -arch=sm_30 -O main.cu lab2.o -o main `pkg-config opencv --cflags --libs`
