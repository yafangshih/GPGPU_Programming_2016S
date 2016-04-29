g++ -std=c++11 -c pgm.cpp 
nvcc -std=c++11 -arch=sm_30 -O -c lab3.cu -o lab3.o `pkg-config opencv --cflags --libs`
nvcc -std=c++11 -arch=sm_30 -O main.cu lab3.o pgm.o -o main `pkg-config opencv --cflags --libs`
