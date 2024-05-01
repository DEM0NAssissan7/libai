#include "libai.h"

#include <stdio.h>
#include <stdlib.h>

float input[2];
float target[1];
float inputs[4][2] = {
    {0, 0},
    {1, 0},
    {0, 1},
    {1, 1}
};
float targets[4] = {
    0,
    1,
    1,
    0
};

long trials = 0;

void write_array(int len, float *a, float *b){
    for(int i = 0; i < len; i++)
        a[i] = b[i];
}
void set_io(){
    int index = trials % 4;
    write_array(2, input, inputs[index]);
    target[0] = targets[index];
    trials++;
}
void test(Network *network){
    for(int i = 0; i < 4; i++){
        set_io();
        netrun(network, input);
        netprintout(network);
    }
}

int main(){
    //Inits

    //Program begin
    Layout layout = netlay(2, 2, 1);
    Network network = netinit(&layout, 4, 0.02);


    printf("Before training:\n");
    test(&network);
    
    for(int i = 0; i < 4; i++) {
        set_io();
        printf("Input: (%f, %f) | Target: (%f)\n", input[0], input[1], target[0]);
        nettrain(&network, input, target);
    }


    printf("\nAfter training:\n");
    netdebug(&network);
    test(&network);
    return 0;
}