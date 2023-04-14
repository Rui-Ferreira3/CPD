# the compiler: gcc for C program, define as g++ for C++

# compiler flags:
#  -g     - this flag adds debugging information to the executable file
#  -Wall  - this flag is used to turn on most compiler warnings
NUM_PROCESSES = 2
FILE = gen24-50000.in
VAL = 50000
# The build target
all: tsp-mpi tsp-mpi.o element.o

tsp-mpi.o: tsp-mpi.cpp tsp-mpi.h
	mpic++ -c -O3 -fopenmp tsp-mpi.cpp

element.o: element.cpp
	mpic++ -c -O3 -fopenmp element.cpp

tsp-mpi: tsp-mpi.o element.o
	mpic++ -g -Wall -O3 -o tsp-mpi -fopenmp tsp-mpi.o element.o

clean:
	-rm -f tsp-mpi *.o

run: 
	srun -n $(NUM_PROCESSES) tsp-mpi $(FILE) $(VAL)

run_all:
	# @echo gen10
	# srun -n $(NUM_PROCESSES) tsp-mpi gen10-20.in 20
	# @echo gen15
	# srun -n $(NUM_PROCESSES) tsp-mpi gen15-25.in 25
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

run_all_local:
	@echo gen10
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen10-20.in 20
	@echo gen15
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen15-25.in 25
	@echo gen19
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen19-23.in 23
	@echo gen20
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen20-5000.in 5000
	@echo gen22
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen22-25000.in 25000
	@echo gen24
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen24-50000.in 50000
	@echo gen26
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen26-50000.in 50000
	@echo gen30
	mpirun -n $(NUM_PROCESSES) tsp-mpi gen30-5000.in 5000

run_n:
	@echo N=1
	mpirun -n 1 tsp-mpi $(FILE) $(VAL)
	@echo N=2
	mpirun -n 2 tsp-mpi $(FILE) $(VAL)
	@echo N=3
	mpirun -n 3 tsp-mpi $(FILE) $(VAL)
	@echo N=4
	mpirun -n 4 tsp-mpi $(FILE) $(VAL)
	# @echo N=8
	# srun -n 8 tsp-mpi $(FILE) $(VAL)
	# @echo N=16
	# srun -n 16 tsp-mpi $(FILE) $(VAL)
	# @echo N=32
	# srun -n 32 tsp-mpi $(FILE) $(VAL)
	# @echo N=64
	# srun -n 64 tsp-mpi $(FILE) $(VAL)