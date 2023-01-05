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
float targets[4][1] = {
    {0},
    {1},
    {1},
    {0}
};

long trials = 0;

void r_array(int len, float *a, float *b){
    for(int i = 0; i < len; i++)
        a[i] = b[i];
}
void set_io(){
    int index = trials % 4;
    r_array(2, input, inputs[index]);
    r_array(2, target, targets[index]);
    trials++;
}


int main(){
    //Inits

    //Program begin
    Layout layout = netlay(2, 2, 1);
    Network network = netinit(&layout, 4, 0.07);

    void train(){
        set_io();
        nettrain(&network, input, target);
    }
    void test(){
        for(int i = 0; i < 4; i++){
            set_io();
            netrun(&network, input);
            netprintout(&network);
        }
    }
    printf("Before training:\n");
    test();
    
    for(int i = 0; i < 1000000; i++)
        train();


    printf("\nAfter training:\n");
    netdebug(&network);
    test();
    return 0;
}