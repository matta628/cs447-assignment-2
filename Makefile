CFLAGS = -g -Wall -Wextra -pedantic -O2

all: k-nn

k-nn: k-nn.o KDtree.o
	g++ $(CFLAGS) -pthread -o k-nn k-nn.o KDtree.o
k-nn.o: k-nn.cpp KDtree.hpp
	g++ $(CFLAGS) -c k-nn.cpp
KDtree.o: KDtree.cpp KDtree.hpp
	g++ $(CFLAGS) -c KDtree.cpp

clean:
	rm k-nn.o KDtree.o k-nn 

1d: all
	./k-nn 1 input/t1d.dat input/q1d.dat input/r1d.dat

2d: all
	./k-nn 1 input/t2d.dat input/q2d.dat input/r2d.dat

3d: all
	./k-nn 1 input/t3d.dat input/q3d.dat input/r3d.dat

time1d: all
	time -p ./k-nn 1 input/t1d.dat input/q1d.dat input/r1d.dat
time2d: all
	time -p ./k-nn 1 input/t2d.dat input/q2d.dat input/r2d.dat
time3d: all
	time -p ./k-nn 1 input/t3d.dat input/q3d.dat input/r3d.dat
