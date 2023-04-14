#include "tsp-mpi.h"

#define NUM_SWAPS 10
#define NUM_ITERATIONS 500

int main(int argc, char *argv[]) {
    double exec_time;
    double start_time, end_time;
    
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        if(parse_inputs(argc, argv) == -1) {
            MPI_Finalize();
            exit(-1);
        }
        start_time = MPI_Wtime();
    }

    share_inputs(rank);

    PriorityQueue<QueueElem> startElems;
    if(rank == 0)
        create_tasks(num_processes, startElems);

    // create an MPI data type for QueueElem
    MPI_Datatype elem_type = create_MPI_type();

    int elementPerProcess = startElems.size()/num_processes;
    if (num_processes > 1) {
        MPI_Bcast(&elementPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    PriorityQueue<QueueElem> myQueue;
    split_tasks(rank, startElems, myQueue, elem_type, elementPerProcess);

    pair<vector<int>, double> results = tsp(myQueue, rank, elem_type);

    pair<vector <int>, double> best = get_results(rank, results);

    if(rank == 0) {
        end_time = MPI_Wtime();

        fprintf(stderr, "%.1fs\n", end_time-start_time);

        print_result(best.first, best.second);
    }
    // printf("Process %d finished\n", rank);

    MPI_Finalize();
    return 0;
}

int parse_inputs(int argc, char *argv[]) {
    string line;
    ifstream myfile (argv[1]);

    int row, col;
    double val;

    if(argc-1 != 2)
        return -1;

    if (myfile.is_open()){
        getline(myfile, line);
        sscanf(line.c_str(), "%d %d", &numCities, &numRoads);
    }else
        return -1;

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
        return -1;

    BestTourCost = atof(argv[2]);
    return 0;
}

void share_inputs(int rank) {
    if(num_processes > 1) {
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

        for(int i=0; i<numCities; i++) {
            MPI_Bcast(&distances[i][0], numCities, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }
}

void create_tasks(int num_processes, PriorityQueue<QueueElem> &startQueue) {
    vector<pair<double,double>> mins = get_mins();

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
}

void split_tasks(int rank, PriorityQueue<QueueElem>&startElems, PriorityQueue<QueueElem> &myQueue, MPI_Datatype elem_type, int elementPerProcess) {
    QueueElem elem = {{0}, 0.0, 100.0, 1, 0};
    if (rank == 0) {
        if (num_processes > 1) {
            int last;
            for(int i=1; i<num_processes; i++) {
                for(int j=0; j<elementPerProcess; j++) {
                    send_element(i, j, startElems.pop(), elem_type);
                    last = i*elementPerProcess;
                }
            }
        }

        myQueue = startElems;
    }else {
        for(int i=0; i<elementPerProcess; i++) {
            QueueElem myElem = recv_element(0, i, elem_type);
            myQueue.push(myElem);
        }
    }
}

pair<vector <int>, double> tsp(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type) {
    double globalBestCost = BestTourCost;

    vector<pair<double,double>> mins = get_mins();

    vector <int> BestTour;
    BestTour.reserve(numCities+1);
    myQueue.print(printQueueElem);
    
    int cnt=0;
    int flag=5;
    while(flag != 0){
        // if(num_processes > 1)
        //     get_elements(myQueue, rank, elem_type);

        if(myQueue.size() > 0) {
            QueueElem myElem = myQueue.pop();

            if(num_processes > 1)
                globalBestCost = update_BestTour(rank, BestTour, globalBestCost);

            if(myElem.bound >= BestTourCost || myElem.bound >= globalBestCost) {
                myQueue.clear();
            }else {
                if(myElem.length == numCities) {
                    double dist = distances[myElem.node][0];
                    if(dist > 0) {
                        if(myElem.cost + dist <= BestTourCost) {
                            BestTour = myElem.tour;
                            BestTour.push_back(0);
                            BestTourCost = myElem.cost + dist;
                            if(num_processes > 1 && myElem.cost + dist < globalBestCost)
                                send_BestTourCost(rank);
                        }
                    }
                }else 
                    create_children(myElem, myQueue, mins);
            }

            // if(num_processes > 1) {
            //     if(cnt > NUM_ITERATIONS) {
            //         redistribute_elements(myQueue, rank, elem_type);
            //         cnt = 0;
            //     }else
            //         cnt++;
            // }
            // printf("Iteration %d of rank %d\n", cnt, rank);
            // if(num_processes > 1)
            //     redistribute_elements(myQueue, rank, elem_type);

        }
        
        int size = myQueue.size();
        MPI_Allreduce(&size, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        // cout << fixed << BestTourCost << endl;
        // for(int i=0; i<BestTour.size(); i++) {
        //     cout << BestTour[i] << " ";
        // }
        // cout << endl;
        // printf("Total number of elements in queues is %d\n", flag);
    }

    printf("Rank: %d", rank);

    return make_pair(BestTour, BestTourCost);
}

void create_children(QueueElem &myElem, PriorityQueue<QueueElem> &myQueue, vector<pair<double,double>> &mins) {
    bool visitedCities[numCities] = {false};

    for (int city : myElem.tour) {
        visitedCities[city] = true;
    }

    for(int v=0; v<numCities; v++) {
        double dist = distances[myElem.node][v];
        if(dist>0 && !visitedCities[v]) {
            double newBound = calculateLB(mins, myElem.node, v, myElem.bound);                       
            if(newBound <= BestTourCost) {
                vector <int> newTour = myElem.tour;
                newTour[myElem.length+1] = v;
                myQueue.push({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
            }
        }
    }
}

MPI_Datatype create_MPI_type() {
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
    return elem_type;
}

// void send_element(int dest, int tag, QueueElem elem, MPI_Datatype elem_type) {
//     MPI_Send(&elem, 1, elem_type, dest, tag, MPI_COMM_WORLD);
//     int tour_size = elem.tour.size();
//     MPI_Send(&tour_size, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
//     MPI_Send(elem.tour.data(), elem.tour.size(), MPI_INT, dest, tag, MPI_COMM_WORLD);
// }

// QueueElem recv_element(int source, int tag, MPI_Datatype elem_type) {
//     QueueElem myElem;
//     MPI_Recv(&myElem, 1, elem_type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     int tour_size;
//     MPI_Recv(&tour_size, 1, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     vector<int> received_tour(tour_size);
//     MPI_Recv(received_tour.data(), tour_size, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     myElem.tour = received_tour;
//     return myElem;
// }

void send_element(int dest, int tag, QueueElem elem, MPI_Datatype elem_type) {
    elem.tour.resize(numCities+1);
    int size = (numCities+1)*sizeof(int) + 2*sizeof(int) + 2*sizeof(double), pos = 0;
    char buffer[size] = {};
    for(int i=0; i<numCities+1; i++) {
        memcpy(&buffer[pos], &elem.tour[i], sizeof(int));
        pos += sizeof(int);
    }
    memcpy(&buffer[pos], &elem.cost, sizeof(double));
    pos += sizeof(double);
    memcpy(&buffer[pos], &elem.bound, sizeof(double));
    pos += sizeof(double);
    memcpy(&buffer[pos], &elem.length, sizeof(double));
    pos += sizeof(int);
    memcpy(&buffer[pos], &elem.node, sizeof(int));
    MPI_Send(&buffer[0], size, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    // printf("Sent element:\n");
    // printQueueElem(elem);
}

QueueElem recv_element(int source, int tag, MPI_Datatype elem_type) {
    QueueElem elem;
    elem.tour.resize(numCities+1);
    int size = (numCities+1)*sizeof(int) + 2*sizeof(int) + 2*sizeof(double), pos = 0;
    char buffer[size] = {};
    MPI_Recv(&buffer[0], size, MPI_CHAR, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for(int i=0; i<numCities+1; i++) {
        memcpy(&elem.tour[i], &buffer[pos], sizeof(int));
        pos += sizeof(int);
    }
    memcpy(&elem.cost, &buffer[pos], sizeof(double));
    pos += sizeof(double);
    memcpy(&elem.bound, &buffer[pos], sizeof(double));
    pos += sizeof(double);
    memcpy(&elem.length, &buffer[pos], sizeof(double));
    pos += sizeof(int);
    memcpy(&elem.node, &buffer[pos], sizeof(int));
    // printf("Received element:\n");
    // printQueueElem(elem);
    elem.tour.resize(elem.length);
    return elem;
}

void redistribute_elements(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type) {
    int dest;
    if (rank==0) {
        dest = num_processes-1;
    }else {
        dest = rank-1;
    }

    if(myQueue.size() > NUM_SWAPS) {
        // for(int i=0; i<NUM_SWAPS; i++) {
            send_element(dest, 2, myQueue.pop(), elem_type);
        // }
    }
}

void get_elements(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type) {
    int source;
    if (rank==num_processes-1) {
        source = 0;
    }else {
        source = rank+1;
    }

    int flag;
    while(1) {
        MPI_Iprobe(source, 2, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        if(flag) {
            QueueElem newElem = recv_element(source, 2, elem_type);
            myQueue.push(newElem);
            // printf("Process %d received:\n", rank);
            // printQueueElem(newElem);
        }else
            break;
    }
}

void send_BestTourCost(int rank) {
    for(int i=0; i<num_processes; i++) {
        if(i!=rank) {
            MPI_Send(&BestTourCost, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        }
    }
}

double update_BestTour(int rank, vector <int> &BestTour, double globalBestCost) {
    for(int i=0; i<num_processes; i++) {
        if(i!=rank) {
            int flag;
            MPI_Iprobe(i, 1, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if(flag) {
                double newBest = globalBestCost;
                MPI_Recv(&newBest, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(newBest < globalBestCost) {
                    return newBest;
                }
            }
        }
    }
    return BestTourCost;
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

pair<vector <int>, double> get_results(int rank, pair<vector<int>, double> results) {
    vector<int> bestTour(numCities+1);
    double bestCost;
    int bestRank;
    if (num_processes > 1) {
        MPI_Allreduce(&results.second, &bestCost, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        if(bestCost==results.second && results.first.size()==numCities+1) {
            if (rank != 0) {
                MPI_Send(results.first.data(), numCities+1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }

        double costs[num_processes];
        MPI_Gather(&results.second, 1, MPI_DOUBLE, &costs[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(rank == 0) {
            for(int i=0; i<num_processes; i++) {
                if(costs[i] == bestCost) {
                    bestRank = i;
                    break;
                }
            }

            if(bestRank != 0) {
                MPI_Recv(bestTour.data(), numCities+1, MPI_INT, bestRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else
                bestTour = results.first;
        }
    }else {
        bestTour = results.first;
        bestCost = results.second;
    }

    return make_pair(bestTour, bestCost);
}

void print_result(vector <int> BestTour, double BestCost) {
    if(BestTour.size() != numCities+1) {
        cout << "NO SOLUTION" << endl;
        cout << fixed << BestCost << endl;
        for(int i=0; i<numCities+1; i++) {
            cout << BestTour[i] << " ";
        }
    } else {
        cout.precision(1);
        cout << fixed << BestCost << endl;
        for(int i=0; i<numCities+1; i++) {
            cout << BestTour[i] << " ";
        }
        cout << endl;
    }
}