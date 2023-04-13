#include "tsp-mpi.h"

#define NUM_SWAPS 3
#define NUM_ITERATIONS 30

int main(int argc, char *argv[]) {
    double exec_time;
    double start_time, end_time; 

    // omp_set_num_threads(2);
    
    int num_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        parse_inputs(argc, argv);
        start_time = MPI_Wtime();
    }

    //MPI_Bcast sends the message from the root process to all other processes
    MPI_Bcast(&numCities, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numRoads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BestTourCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        distances.resize(numCities);
        for(int i=0; i<numCities; i++) {
            distances[i].resize(numCities);
        }
    }

    MPI_Bcast(&distances[0][0], distances.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 1) {
        for(int i=0; i<numCities; i++) {
            for(int j=0; j<numCities; j++) {
                cout << distances[i][j] << " ";
            }
            cout << endl;
        }
    }

    vector<QueueElem> startElems;
    if(rank == 0)
        startElems = split_work(num_processes);

    // create an MPI data type for QueueElem
    MPI_Datatype elem_type;
    int lengths[] = {1, 1, 1, 1};
    MPI_Aint displacements[] = {
            offsetof(QueueElem, cost),
            offsetof(QueueElem, bound),
            offsetof(QueueElem, length),
            offsetof(QueueElem, node)};
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &elem_type);
    MPI_Type_commit(&elem_type);
    QueueElem elem = {{0}, 0.0, 100.0, 1, 0};

    int elementPerProcess = startElems.size()/num_processes;
    MPI_Bcast(&elementPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    PriorityQueue<QueueElem> myQueue;
    if (rank == 0) {
        // send the array of QueueElem data to process 1
        int last;
        for(int i=1; i<num_processes; i++) {
            for(int j=0; j<elementPerProcess; j++) {
                send_element(i, j, startElems[(i-1)*elementPerProcess+j], elem_type);
                // printf("Sent node %d to process %d\n", startElems[(i-1)*elementPerProcess+j].node, i);
                last = i*elementPerProcess;
            }
        }

        for(int h=last; h<startElems.size(); h++) {
            myQueue.push(startElems[h]);
            // printf("Rank: %d Node: %d\n", rank, startElems[h].node);
        }
    }else {
        // receive the array of QueueElem data from process 0
        for(int i=0; i<elementPerProcess; i++) {
            QueueElem myElem = recv_element(i, elem_type);
            // printf("Received node %d in process %d\n", myElem.node, rank);
            myQueue.push(myElem);
            // printf("Rank: %d Node: %d\n", rank, myElem.node);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // myQueue.print(printQueueElem);

    // calculate tsp
    pair<vector<int>, double> results = tsp(myQueue, rank);

    // printf("Rank %d\n", rank);
    // print_result(results.first, results.second);

    double costs[num_processes];
    MPI_Gather(&results.second, 1, MPI_DOUBLE,
                &costs[0], 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if(rank == 0) {
        for(int i=0; i<num_processes; i++)
            printf("Cost %d: %lf\n", i, costs[i]);
    }

    if(rank == 0)
        end_time = MPI_Wtime();


   //  print_result(best_result.first, best_result.second);

    MPI_Finalize();
    return 0;
}

void send_element(int dest, int tag, QueueElem elem, MPI_Datatype elem_type) {
    MPI_Send(&elem, 1, elem_type, dest, tag, MPI_COMM_WORLD);
    int tour_size = elem.tour.size();
    MPI_Send(&tour_size, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    MPI_Send(elem.tour.data(), elem.tour.size(), MPI_INT, dest, tag, MPI_COMM_WORLD);
}

QueueElem recv_element(int tag, MPI_Datatype elem_type) {
    QueueElem myElem;
    MPI_Recv(&myElem, 1, elem_type, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int tour_size;
    MPI_Recv(&tour_size, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    vector<int> received_tour(tour_size);
    MPI_Recv(received_tour.data(), tour_size, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    myElem.tour = received_tour;
    return myElem;
}

void parse_inputs(int argc, char *argv[]) {
    string line;
    ifstream myfile (argv[1]);

    int row, col;
    double val;

    if(argc-1 != 2)
        exit(-1);

    if (myfile.is_open()){
        getline(myfile, line);
        sscanf(line.c_str(), "%d %d", &numCities, &numRoads);
    }else
        exit(-1);

    for(int i=0; i<numCities; i++) {
        vector <double> ones(numCities, -1.0);
        distances.push_back(ones);
    }
    
    if (myfile.is_open()){
        while (getline(myfile, line)) {
            sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
            distances[row][col] = val;
            distances[col][row] = val;
        }
        myfile.close();
    }else 
        exit(-1);

    BestTourCost = atof(argv[2]);
}

void print_result(vector <int> BestTour, double BestTourCost) {
    if(BestTour.size() != numCities+1) {
        cout << "NO SOLUTION" << endl;
    } else {
        cout.precision(1);
        cout << fixed << BestTourCost << endl;
        for(int i=0; i<numCities+1; i++) {
            cout << BestTour[i] << " ";
        }
        cout << endl;
    }
}

void create_children(QueueElem &myElem, PriorityQueue<QueueElem> &myQueue, vector<pair<double,double>> &mins) {
    bool visitedCities[numCities] = {false};

    for (int city : myElem.tour) {
        visitedCities[city] = true;
    }

    for(int v=0; v<numCities; v++) {
        double dist = distances[myElem.node][v];
        if(dist>0 && !visitedCities[v]) {
            cout << "line 185" << endl;
            double newBound = calculateLB(mins, myElem.node, v, myElem.bound);                       
            if(newBound <= BestTourCost) {
                vector <int> newTour = myElem.tour;
                newTour.push_back(v);
                printQueueElem({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
                myQueue.push({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
            }
        }
    }
}

pair<vector <int>, double> tsp(PriorityQueue<QueueElem> &myQueue, int rank) {
    vector<pair<double,double>> mins = get_mins();

    vector <int> BestTour;
    BestTour.reserve(numCities+1);
    
    while(myQueue.size() > 0){
        QueueElem myElem = myQueue.pop();
        // if(rank == 0)
        //     printQueueElem(myElem);

        if(myElem.bound >= BestTourCost)
            break;

        if(myElem.length == numCities) {
            double dist = distances[myElem.node][0];
            if(dist > 0) {
                if(myElem.cost + dist <= BestTourCost) {
                    BestTour = myElem.tour;
                    BestTour.push_back(0);
                    BestTourCost = myElem.cost + dist;
                }
            }
        }else 
            create_children(myElem, myQueue, mins);
    }

    return make_pair(BestTour, BestTourCost);
}

vector<QueueElem> split_work(int num_processes) {
    vector<pair<double,double>> mins = get_mins();

    PriorityQueue<QueueElem> startQueue;
    startQueue.push({{0}, 0.0, initialLB(mins), 1, 0});

    while(startQueue.size() < num_processes) {
        QueueElem myElem = startQueue.pop();

        bool visitedCities[numCities] = {false};
        for (int city : myElem.tour) {
            visitedCities[city] = true;
        }

        for(int v=0; v<numCities; v++) {
            double dist = distances[myElem.node][v];
            if(dist>0 && !visitedCities[v]) {
                double newBound = calculateLB(mins, myElem.node, v, myElem.bound);
                vector <int> newTour = myElem.tour;
                newTour.push_back(v);
                startQueue.push({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
            }
        }
    }

    vector<QueueElem> startElems;
    startElems.reserve(startQueue.size());
    while (!startQueue.empty()){
        startElems.push_back(startQueue.pop());
    }
    
    return startElems;
}

vector<pair<double,double>> get_mins() {
    vector<pair<double,double>> mins;
    mins.reserve(numCities);

    for (int i=0; i<numCities; i++) {
        double min1 = BestTourCost;
        double min2 = BestTourCost;
        for (int j=0; j<numCities; j++) {
            double dist = distances[i][j];
            if(dist > 0) {
                if(dist <= min1) {
                    min2 = min1;
                    min1 = dist;
                }else if(dist <= min2) {
                    min2 = dist;
                }
            }
        }
        mins[i] = make_pair(min1, min2);
    }

    return mins;
}

double initialLB(vector<pair<double,double>> &mins) {
    double LB=0;

    for(int i=0; i<numCities; i++) {
        LB += (mins[i].first + mins[i].second)/2;
    }
    return LB;
}

double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB) {
    double cf, ct;
    double directCost = distances[f][t];

    if(distances[f][t] <= 0)
        exit(-1);

    if(directCost >= mins[f].second) {
        cf = mins[f].second;
    }else
        cf = mins[f].first;

    if(directCost >= mins[t].second) {
        ct = mins[t].second;
    }else
        ct = mins[t].first;

    return LB + directCost - (cf+ct)/2;
}