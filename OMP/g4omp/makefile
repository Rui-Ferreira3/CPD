# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings

# The build target
all: tsp-omp tsp-omp.o element.o

tsp-omp.o: tsp-omp.cpp tsp-omp.h
	g++ -c -O3 -fopenmp tsp-omp.cpp

element.o: element.cpp element.h
	g++ -c -O3 -fopenmp element.cpp 

tsp-omp: tsp-omp.o element.o
	g++ -g -Wall -O3 -o tsp-omp -fopenmp tsp-omp.o element.o

clean:
	-rm -f tsp-omp *.o
