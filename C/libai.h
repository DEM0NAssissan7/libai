// Includes
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>

// Definitions
#define E 2.718281828

// Objects
typedef struct
    Node
{
    float value;
    float gradient;
    float bias;
    float bias_change;
    int num_weights;
    float *weights;
    float *changes;
} Node;

typedef struct
    Layout
{
    int size;
    int *data;
} Layout;

typedef Node *Layer;
typedef struct
    Network
{
    Layer *layers;
    int *layout;
    int epoch;
    float learning_rate;
    float cost;
    float average_cost;
    long runs;
    //QOL variables
    int out_index;
    int size;
} Network;

// Internal use functions
long count = 0;
float c_random()
{
    srand(time(NULL) + count);
    count++;
    return (float)rand() / (float)RAND_MAX;
}
float sigmoid(float x){
    return 1 / (1 + pow(E, -x));
}
float square(float x){
    return x * x;
}
float sum_gradient(Network *network, int node_index, int layer_index){
    float result = 0;
    for(int i = 0; i < network->layout[layer_index]; i++){
        Node *node = &(network->layers[layer_index][i]);
        result += node->weights[node_index] * node->gradient;
    }
    return result;
}

// Object Initializations
Node init_node(int previous_layer_size)
{
    Node node;
    // Variable initializations
    node.value = 0;
    node.gradient = 0;
    node.bias = 0;
    node.bias_change = 0;
    node.num_weights = previous_layer_size;
    node.weights = malloc(sizeof(float[previous_layer_size]));
    node.changes = malloc(sizeof(float[previous_layer_size]));

    for (int i = 0; i < previous_layer_size; i++)
    {
        node.weights[i] = c_random();
        node.changes[i] = 0;
    }
    return node;
}
Layout netlay(int nums, ...){
    Layout layout;
    
    va_list ptr;

    va_start(ptr, nums);
    unsigned long len = 0;
    int *data_buffer = malloc(sizeof(int));
    data_buffer[0] = nums;
    while(1){
        int num = va_arg(ptr, int);
        if(num == 0) break;
        len++;
        data_buffer = realloc(data_buffer, sizeof(int[len]));
        data_buffer[len] = num;
    }
    layout.size = len;
    layout.data = data_buffer;

    return layout;
}
Network netinit(Layout *layout, int epoch, float learning_rate){
    Network network;
    network.cost = 0;
    network.average_cost = 0;
    network.runs = 0;

    //Initialize option variables
    network.layout = layout->data;
    network.epoch = epoch;
    network.learning_rate = learning_rate;

    //Initialize layers
    int network_size = layout->size;
    network.size = network_size;
    network.out_index = network_size - 1;
    network.layers = malloc(sizeof(Layer[network_size]));

    //Initalize nodes
    for(int i = 0; i < network_size; i++){
        int layer_size = layout->data[i];
        network.layers[i] = malloc(sizeof(Node[layer_size]));
        for(int j = 0; j < layer_size; j++){
            if(i > 0)
                network.layers[i][j] = init_node(layout->data[i - 1]);
            else
                network.layers[i][j] = init_node(0);
        }
    }

    return network;
}

//Network functions
void netrun(Network *network, float *inputs){
    int *layout = network->layout;
    
    //Set first layer to inputs
    for(int i = 0; i < layout[0]; i++)
        network->layers[0][i].value = inputs[i];
    
    //Calculate the network output
    for(int i = 1; i < network->size; i++){
        for(int j = 0; j < layout[i]; j++){
            Node *node = &(network->layers[i][j]);
            node->value = 0;
            for(int k = 0; k < node->num_weights; k++)
                node->value += network->layers[i - 1][k].value * node->weights[k];
            node->value += node->bias;
            node->value = sigmoid(node->value);
        }
    }
}
void netcost(Network *network, float *targets){
    float cost = 0;
    for(int i = 0; i < network->layout[network->out_index]; i++)
        cost += square(network->layers[network->out_index][i].value - targets[i]);
    network->cost = cost;
    network->average_cost = (cost + network->epoch * network->average_cost) / (network->epoch + 1);
}
void netlearn(Network *network, float *targets){
    //Calculate gradients and changes for the output layer and the one following it
    for(int i = 0; i < network->layout[network->out_index]; i++){
        Node *node = &(network->layers[network->out_index][i]);
        node->gradient = (2 * (node->value - targets[i])) * (node->value * (1 - node->value));
        node->bias_change -= network->learning_rate * node->gradient;
        for(int j = 0; j < node->num_weights; j++)
            node->changes[j] -= network->learning_rate * (node->gradient * network->layers[network->out_index - 1][j].value);
    }

    //Get network cost
    netcost(network, targets);

    //Update the rest of the layers
    for(int i = network->out_index - 1; i > 0; i--){
        for(int j = 0; j < network->layout[i]; j++){
            Node *node = &(network->layers[i][j]);
            node->gradient = sum_gradient(network, j, i + 1) * (node->value * (1 - node->value));
            node->bias_change -= network->learning_rate * node->gradient;
            for(int k = 0; k < node->num_weights; k++)
                node->changes[k] -= network->learning_rate * (node->gradient * network->layers[i - 1][k].value);
        }
    }

    //Apply weights on each epoch
    if(network->runs % (network->epoch + 1) == network->epoch){
        for(int i = 0; i < network->size; i++){
            for(int j = 0; j < network->layout[i]; j++){
                Node *node = &(network->layers[i][j]);
                //Apply bias change
                node->bias += node->bias_change;
                node->bias_change = 0;
                for(int k = 0; k < node->num_weights; k++){
                    //Apply weight changes
                    node->weights[k] += node->changes[k];
                    node->changes[k] = 0;
                }
            }
        }
    }

    network->runs++;
}
void nettrain(Network *network, float *inputs, float *targets){
    netrun(network, inputs);
    netlearn(network, targets);
}
void netprintout(Network *network){
    for(int i = 0; i < (network->layout)[network->out_index]; i++)
        printf("%f\n", network->layers[network->out_index][i].value);
}
void netdebug(Network *network){
    printf("Cost: %f | Average: %f\n", network->cost, network->average_cost);
}