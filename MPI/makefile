# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
NUM_PROCESSES = 4
FILE = gen10-20.in
VAL = 20
# The build target
all: tsp-mpi tsp-mpi.o

tsp-mpi.o: tsp-mpi.cpp tsp-mpi.h
	mpic++ -c -O3 -fopenmp tsp-mpi.cpp

tsp-mpi: tsp-mpi.o
	mpic++ -g -Wall -O3 -o tsp-mpi -fopenmp tsp-mpi.o

clean:
	-rm -f tsp-mpi *.o

run: 
	srun -n $(NUM_PROCESSES) tsp-mpi $(FILE) $(VAL)

run_all:
	@echo gen10
	srun -n $(NUM_PROCESSES) tsp-mpi gen10-20.in 20
	@echo gen15
	srun -n $(NUM_PROCESSES) tsp-mpi gen15-25.in 25
	@echo gen19
	srun -n $(NUM_PROCESSES) tsp-mpi gen19-23.in 23
	@echo gen20
	srun -n $(NUM_PROCESSES) tsp-mpi gen20-5000.in 5000
	@echo gen22
	srun -n $(NUM_PROCESSES) tsp-mpi gen22-25000.in 25000
	@echo gen24
	srun -n $(NUM_PROCESSES) tsp-mpi gen24-50000.in 50000
	@echo gen26
	srun -n $(NUM_PROCESSES) tsp-mpi gen26-50000.in 50000
	@echo gen30
	srun -n $(NUM_PROCESSES) tsp-mpi gen30-5000.in 5000

run_local:
	mpirun -n $(NUM_PROCESSES) tsp-mpi $(FILE) $(VAL)