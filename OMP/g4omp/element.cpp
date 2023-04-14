#include "element.h"

void printQueueElem(QueueElem myElem) {
    cout << "Node: " << myElem.node << endl;
    cout << "Tour: ";
    for(int i=0; i<myElem.length; i++) {
        cout << myElem.tour.at(i) << " ";
    }
    cout << endl;
    cout << "Cost: " << myElem.cost << endl;
    cout << "Bound: " << myElem.bound << endl;
    cout << "Lenght: " << myElem.length << endl;
}