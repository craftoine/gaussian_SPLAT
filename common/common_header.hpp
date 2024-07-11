#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <stack>

void random_unitary_quaternion(float_double* q){
    float_double d = 0;
    for(dimension_type i=0;i<4;i++){
        q[i] = 2*((float_double)rand()/(float_double)RAND_MAX)-1;
        d += q[i]*q[i];
    }
    d = sqrt(d);
    for(dimension_type i=0;i<4;i++){
        q[i] = q[i]/d;
    }
}
void random_scale(float_double* s){
    for(dimension_type i=0;i<3;i++){
        s[i] = 2*((float_double)rand()/(float_double)RAND_MAX)-1;
    }
}


