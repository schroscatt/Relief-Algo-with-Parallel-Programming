/*
Notes: Relief Algorithm with Parallel Programing - MPI
*/
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
#include <vector>
#include<bits/stdc++.h>

using namespace std;



// Finds the maximum and minumum values of the features among the instances of the current processor.
pair<vector<double>, vector<double>> findMaxnMinFeatures( vector<vector<double>> instances, int featureNum, int perInstance) {
    vector<double> maxFeatures(featureNum, -1/0.0);
    vector<double> minFeatures(featureNum, 1/0.0);

    // Iterate over all instances of the processor
    for(int instance = 0; instance < perInstance; instance++) {
        // Iterate over the features
        for( int feature = 0; feature < featureNum; feature++) {
            double val = instances[instance][feature];
            if( val > maxFeatures[feature]) {
                maxFeatures[feature] = val;
            }
            if( val < minFeatures[feature]) {
                minFeatures[feature] = val;
            }
        }

    }

    return make_pair(maxFeatures, minFeatures);

}

// Finds Manhattan Distance ==> MH = |x1 - x2| + |y1 - y2| ...
double manhattanDistance(vector<double> instance1, vector<double> instance2, int featureNum) {
    double value = 0.0;
    for(int index = 0; index < featureNum; index++) {

        value += abs(instance1[index] - instance2[index]);
    }
    return value;
}


// Takes differences
double diff(double val1, double val2, double maxA, double minA) {

    return abs(val1 - val2)/(maxA - minA);
}

vector<double> relief(vector<vector<double>> instances, int featureNum, int perInstance, int iterationNum) {
    vector<double> weights(featureNum, 0.0); // Initialize the weight vector

    // Get the maximum and minumum values of the features among all instances of the processor
    pair<vector<double>, vector<double>> maxNmin = findMaxnMinFeatures(instances, featureNum, perInstance);

    // Iterate over instances sequentially.
    for(int i = 0; i < iterationNum; i++) {
        int process = i % perInstance;
        double nearestHit = 1/0.0;
        double nearestMiss = 1/0.0;
        int nearHit = process;
        int nearMiss = process;

        // Take the class of current instance
        double classNum = instances[process][featureNum];

        // Iterate over instances to find nearest hit and nearest miss.
        for(int instance = 0; instance < perInstance; instance++) {
            if(instance == process) {
                continue;
            }
            if (instances[instance][featureNum] == classNum) {
                double val = manhattanDistance(instances[process], instances[instance], featureNum);
                if (val < nearestHit) {
                    nearestHit = val;
                    nearHit = instance;
                }
            }
            else {
                double val = manhattanDistance(instances[process], instances[instance], featureNum);
                if (val < nearestMiss) {
                    nearestMiss = val;
                    nearMiss = instance;
                }
            }
        }

        // For every feature of the instance calculate the weights.
        for(int feature = 0; feature < featureNum; feature++) {
            double diffHit =  diff(instances[nearHit][feature], instances[process][feature], maxNmin.first[feature], maxNmin.second[feature]);
            double diffMiss = diff(instances[nearMiss][feature], instances[process][feature], maxNmin.first[feature], maxNmin.second[feature]);
            weights[feature] = weights[feature] - diffHit / iterationNum + diffMiss / iterationNum;
        }
    }

    return weights;

}



int main(int argc, char *argv[])
{
    int rank;
    int size;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Take input file
    ifstream input(argv[1]);


    string s;
    int processorNum, instanceNum, featureNum, iterationNum, resultNum, perInstance;

    // Read parameters from the file
    input >> processorNum >> instanceNum >> featureNum >> iterationNum >> resultNum;

    // Calculate per instance for the processors.
    perInstance = instanceNum/(processorNum-1);

    // Calculate total feature number that will be sent to the each processors.
    int insSize = perInstance * (featureNum + 1);


    int features[resultNum];
    double processorInstances[insSize];

    int featureS[resultNum*processorNum];
    double instances[insSize * processorNum];


    // If master
    if(rank == 0) {

        // Broadcast necessary variables to the slaves.
        MPI_Bcast(&iterationNum, processorNum, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&resultNum, processorNum, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&processorNum, processorNum, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&perInstance, processorNum, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&featureNum, processorNum, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&instanceNum, processorNum, MPI_INT, 0, MPI_COMM_WORLD);



        int ind = 0;
        while(ind < insSize) {
            // Dummy values for master
            instances[ind] = 0.0;
            ind ++;
        }
        while (getline(input, s)) {
            // Read feature values from the file and add to the instances array.
            double feature;
            istringstream ss(s);
            while (ss >> feature) {
                instances[ind] = feature;
                ind++;
            }
        }
    }

    // sends instances from master (instances array) to processorInstances on each processor
    MPI_Scatter(instances, insSize, MPI_DOUBLE, processorInstances, insSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // If Slave
    if(rank != 0) {

        vector<vector<double>> allInstances;

        // Convert comming processorInstances array to 2D array which consists of instances.
        for(int index = 0; index < perInstance; index++) {

            vector<double> instance;
            for(int feature = 0; feature <= featureNum; feature++ ) {
                instance.push_back(processorInstances[index*(featureNum + 1) + feature]);
            }
            allInstances.push_back(instance);
        }

        //Calculate weights with relief algorithm
        vector<double> weights = relief(allInstances, featureNum, perInstance, iterationNum);


        // <Weight, Slave Id> Pair
        vector<pair<double, int>> weight;
        for(int i = 0; i < featureNum; i++) {
            weight.push_back(make_pair(weights[i], i));
        }

        // Sort features according to weight.
        sort(weight.rbegin(), weight.rend());

        // Take the top T results.
        for(int result = 0; result < resultNum; result++) {
            int feature = weight[result].second;
            features[result] = feature;
        }

        // Now, sort the results according to feature ID.
        int n = sizeof(features) / sizeof(features[0]);
        sort(features, features + n);

        // Print out the top T feature IDs.
        int last = 0;
        cout << "Slave P" << rank << " : ";
        for(int feature : features) {
            if(last != resultNum - 1) {
                cout << feature << " ";
            }
            else {
                cout << feature << endl;
            }
            last++;
        }

    }


    // Master gets the feature IDs from all processors.
    MPI_Gather(features, resultNum, MPI_INT, featureS, resultNum, MPI_INT, 0, MPI_COMM_WORLD);

    // If Master
    if(rank == 0) {

        // Put every feature ID that comes from the each processors to avoid duplicates in the ascending order.
        set<int, less<>> featureSet;
        for(int index = resultNum; index < resultNum*processorNum; index++) {
            featureSet.insert(featureS[index]);
        }

        // Print out the results
        set<int>::iterator it;
        int last = 0;
        cout << "Master P0 : ";
        for (it = featureSet.begin(); it != featureSet.end()  ; ++it) {
            if(last != featureSet.size() -1 ) {
                cout << *it << " ";
            }
            else {
                cout << *it;
            }
            last++;
        }


    }


    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return(0);
}