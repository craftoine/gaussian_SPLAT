//knn with gaussian like kernel
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <cmath>
#define pi 3.14159265358979323846
#define negligeable_val_when_exp (float_double)-6.
//negligeable gaussian value is so exp(negligeable_val_when_exp)
//sort
#include <algorithm>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#define Simple_heuristic false
#define Simple_heuristic2 true
#define Beter_majoration true
#define float_double float
#define dimension_type unsigned char
#define array_indexes_type unsigned int
float_double det(dimension_type N_dim, float_double* sigma){
    //std::cout<< "det" << N_dim << std::endl;
    if (N_dim == 1){
        //std::cout<< "det end" << std::endl;
        return sigma[0];
    }
    if (N_dim == 2){
        //std::cout<< "det end" << std::endl;
        return sigma[0]*sigma[3] - sigma[1]*sigma[2];
    }
    if (N_dim == 3){
        //std::cout<< "det end" << std::endl;
        return sigma[0]*(sigma[4]*sigma[8]-sigma[5]*sigma[7])-sigma[1]*(sigma[3]*sigma[8]-sigma[5]*sigma[6])+sigma[2]*(sigma[3]*sigma[7]-sigma[4]*sigma[6]);
    }
    //use the sum other the first line
    float_double dete = 0;
    for(dimension_type i=0;i<N_dim;i++){
        float_double sigma_temp[(N_dim-1)*(N_dim-1)];
        for(dimension_type j=0;j<N_dim-1;j++){
            for(dimension_type k=0;k<N_dim-1;k++){
                if(k<i){
                    sigma_temp[j*(N_dim-1)+k] = sigma[(j+1)*N_dim+k];
                }
                else{
                    sigma_temp[j*(N_dim-1)+k] = sigma[(j+1)*N_dim+k+1];
                }
            }
            dete += sigma[0*N_dim+i]*pow(-1, i)*det(N_dim-1, sigma_temp);
        }
    }
    //std::cout<< "det end" << std::endl;
    return dete;
}

void invert_matrix(dimension_type N_dim, double* sigma, double* sigma_inv){
    //create the augmented matrix
    double sigma_aug[N_dim*(2*N_dim)];
    for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=0;j<N_dim;j++){
            sigma_aug[i*(2*N_dim)+j] = sigma[i*N_dim+j];
            if (i==j){
                sigma_aug[i*(2*N_dim)+j+N_dim] = 1;
            }
            else{
                sigma_aug[i*(2*N_dim)+j+N_dim] = 0;
            }
        }
    }

    //gauss pivot method
    for(dimension_type i=0;i<N_dim;i++){
        //find the pivot
        dimension_type pivot = i;
        for(dimension_type j=i;j<N_dim;j++){
            if (abs(sigma_aug[j*(2*N_dim)+i]) > abs(sigma_aug[pivot*(2*N_dim)+i])){
                pivot = j;
            }
        }
        //swap the lines
        for(dimension_type j=0;j<2*N_dim;j++){
            double temp = sigma_aug[i*(2*N_dim)+j];
            sigma_aug[i*(2*N_dim)+j] = sigma_aug[pivot*(2*N_dim)+j];
            sigma_aug[pivot*(2*N_dim)+j] = temp;
        }
        //make the pivot 1
        double pivot_value = sigma_aug[i*(2*N_dim)+i];
        for(dimension_type j=0;j<2*N_dim;j++){
            sigma_aug[i*(2*N_dim)+j] = sigma_aug[i*(2*N_dim)+j]/pivot_value;
        }
        //make the other lines 0
        for(dimension_type j=0;j<N_dim;j++){
            if (j != i){
                double factor = sigma_aug[j*(2*N_dim)+i];
                for(dimension_type k=0;k<2*N_dim;k++){
                    sigma_aug[j*(2*N_dim)+k] = sigma_aug[j*(2*N_dim)+k] - factor*sigma_aug[i*(2*N_dim)+k];
                }
            }
        }
    }
    //copy the inverse to sigma_inv
    for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=0;j<N_dim;j++){
            sigma_inv[i*N_dim+j] = sigma_aug[i*(2*N_dim)+j+N_dim];
        }
    }
}

void invert_matrix(dimension_type N_dim, float* sigma, float* sigma_inv){
    //create the augmented matrix
    float sigma_aug[N_dim*(2*N_dim)];
    for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=0;j<N_dim;j++){
            sigma_aug[i*(2*N_dim)+j] = sigma[i*N_dim+j];
            if (i==j){
                sigma_aug[i*(2*N_dim)+j+N_dim] = 1;
            }
            else{
                sigma_aug[i*(2*N_dim)+j+N_dim] = 0;
            }
        }
    }

    //gauss pivot method
    for(dimension_type i=0;i<N_dim;i++){
        //find the pivot
        dimension_type pivot = i;
        for(dimension_type j=i;j<N_dim;j++){
            if (abs(sigma_aug[j*(2*N_dim)+i]) > abs(sigma_aug[pivot*(2*N_dim)+i])){
                pivot = j;
            }
        }
        //swap the lines
        for(dimension_type j=0;j<2*N_dim;j++){
            float temp = sigma_aug[i*(2*N_dim)+j];
            sigma_aug[i*(2*N_dim)+j] = sigma_aug[pivot*(2*N_dim)+j];
            sigma_aug[pivot*(2*N_dim)+j] = temp;
        }
        //make the pivot 1
        float pivot_value = sigma_aug[i*(2*N_dim)+i];
        for(dimension_type j=0;j<2*N_dim;j++){
            sigma_aug[i*(2*N_dim)+j] = sigma_aug[i*(2*N_dim)+j]/pivot_value;
        }
        //make the other lines 0
        for(dimension_type j=0;j<N_dim;j++){
            if (j != i){
                float factor = sigma_aug[j*(2*N_dim)+i];
                for(dimension_type k=0;k<2*N_dim;k++){
                    sigma_aug[j*(2*N_dim)+k] = sigma_aug[j*(2*N_dim)+k] - factor*sigma_aug[i*(2*N_dim)+k];
                }
            }
        }
    }
    //copy the inverse to sigma_inv
    for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=0;j<N_dim;j++){
            sigma_inv[i*N_dim+j] = sigma_aug[i*(2*N_dim)+j+N_dim];
        }
    }
}


float_double min_eigan_value(dimension_type N_dim, float_double* sigma){
    //USE power iteration method
    //we want the smallest eigan value instead of the biggest so we use the inverse of the matrix
    float_double sigma_inv[N_dim*N_dim];
    //invert the matrix
    invert_matrix(N_dim, sigma, sigma_inv);
    //compute the determinant of sigma
    float_double det_sigma = det(N_dim, sigma);
    //show det of sigma and det of sigma_inv
    //std::cout << det_sigma <<" "<< det(N_dim, sigma_inv) << " "<<det_sigma*det(N_dim, sigma_inv) << std::endl;
    //power iteration method
    float_double x[N_dim];
    for(dimension_type i=0;i<N_dim;i++){
        x[i] = ((float_double)rand()/(float_double)RAND_MAX);
    }
    //normalize x
    float_double norm = 0;
    for(dimension_type i=0;i<N_dim;i++){
        norm += x[i]*x[i];
    }
    norm = sqrt(norm);
    for(dimension_type i=0;i<N_dim;i++){
        x[i] = x[i]/norm;
    }

    float_double x_temp[N_dim];
    float_double lambda = 0;
    float_double lambda_temp = 0;
    size_t counter = 0;
    do{
        counter++;
        lambda = lambda_temp;
        //compute x_temp
        for(dimension_type i=0;i<N_dim;i++){
            x_temp[i] = 0;
            for(dimension_type j=0;j<N_dim;j++){
                x_temp[i] += sigma_inv[i*N_dim+j]*x[j];
            }
        }
        //compute lambda_temp
        lambda_temp = 0;
        for(dimension_type i=0;i<N_dim;i++){
            lambda_temp += x_temp[i]*x[i];
        }
        //normalize x_temp
        float_double norm = 0;
        for(dimension_type i=0;i<N_dim;i++){
            norm += x_temp[i]*x_temp[i];
        }
        norm = sqrt(norm);
        for(dimension_type i=0;i<N_dim;i++){
            x_temp[i] = x_temp[i]/norm;
        }
        //copy x_temp to x
        for(dimension_type i=0;i<N_dim;i++){
            x[i] = x_temp[i];
            //std::cout << x[i] << " ";
        }
        //std::cout << std::endl;
        //std::cout << lambda_temp <<" " << lambda << std::endl;
    }while(abs(lambda_temp - lambda)/abs(lambda_temp) > 0.0001);
    if(lambda_temp < 0){
        //should not happen because the matrix is semi positive definite
        std::cout << "error lambda < 0" << std::endl;
        std::cout << lambda_temp << std::endl;
        std::cout << det_sigma << std::endl;
        std::cout << counter << std::endl;
        for(dimension_type i=0;i<N_dim;i++){
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
        std::cout<<std::endl;
        //show the matrix
        for(dimension_type i=0;i<N_dim;i++){
            for(dimension_type j=0;j<N_dim;j++){
                std::cout << sigma[i*N_dim+j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        //show the inverse
        for(dimension_type i=0;i<N_dim;i++){
            for(dimension_type j=0;j<N_dim;j++){
                std::cout << sigma_inv[i*N_dim+j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;


        //show the next x
        float_double x_temp_temp[N_dim];
        for(dimension_type i=0;i<N_dim;i++){
            x_temp_temp[i] = 0;
            for(dimension_type j=0;j<N_dim;j++){
                x_temp_temp[i] += sigma_inv[i*N_dim+j]*x[j];
            }
        }
        for(dimension_type i=0;i<N_dim;i++){
            std::cout << x_temp_temp[i] << " ";
        }
        std::cout << std::endl;
        exit(1);
    }
    else{
        return 1/lambda_temp - std::min(1/(2*lambda_temp), (float_double)0.0001*lambda_temp);
    }
}


template <dimension_type N_dim>
class point {
    public:
    float_double x[N_dim];
    point(float_double x[N_dim]){
        for(dimension_type i=0;i<N_dim;i++){
            this->x[i] = x[i];
        }
    }
    point(){
        for(dimension_type i=0;i<N_dim;i++){
            this->x[i] = 0;
        }
    }
    float_double norm(){
        float_double norm = 0;
        for(dimension_type i=0;i<N_dim;i++){
            norm += x[i]*x[i];
        }
        return sqrt(norm);
    }
    float_double square_norm(){
        float_double norm = 0;
        for(dimension_type i=0;i<N_dim;i++){
            norm += x[i]*x[i];
        }
        return norm;
    }
    float_double distance(point<N_dim> p){
        float_double dist = 0;
        for(dimension_type i=0;i<N_dim;i++){
            dist += (x[i] - p.x[i])*(x[i] - p.x[i]);
        }
        return sqrt(dist);
    }
    float_double square_distance(point<N_dim> p){
        float_double dist = 0;
        for(dimension_type i=0;i<N_dim;i++){
            dist += (x[i] - p.x[i])*(x[i] - p.x[i]);
        }
        return dist;
    }
    //overload operator for + with point - with point * with float_double and * with point
    point<N_dim> operator+(point<N_dim> p){
        point<N_dim> res;
        for(dimension_type i=0;i<N_dim;i++){
            res.x[i] = x[i] + p.x[i];
        }
        return res;
    }
    point<N_dim> operator-(point<N_dim> p){
        point<N_dim> res;
        for(dimension_type i=0;i<N_dim;i++){
            res.x[i] = x[i] - p.x[i];
        }
        return res;
    }
    point<N_dim> operator*(float_double a){
        point<N_dim> res;
        for(dimension_type i=0;i<N_dim;i++){
            res.x[i] = a*x[i];
        }
        return res;
    }
    //overload the other way of * with float_double and point
    friend point<N_dim> operator*(float_double a, point<N_dim> p){
        point<N_dim> res;
        for(dimension_type i=0;i<N_dim;i++){
            res.x[i] = a*p.x[i];
        }
        return res;
    }

    float_double operator*(point<N_dim> p){
        //scalar product
        float_double res = 0;
        for(dimension_type i=0;i<N_dim;i++){
            res += x[i]*p.x[i];
        }
        return res;
    }

    friend std::ostream& operator<<(std::ostream& os, const point<N_dim>& p){
        os << "point(";
        for(dimension_type i=0;i<N_dim-1;i++){
            os << p.x[i] << ",";
        }
        os << p.x[N_dim-1] << ")";
        return os;
    }
    bool operator==(point<N_dim> p){
        for(dimension_type i=0;i<N_dim;i++){
            if (x[i] != p.x[i]){
                return false;
            }
        }
        return true;
    }
};
class point3d{
    public:
    float_double x;
    float_double y;
    float_double z;
    point3d(float_double x, float_double y, float_double z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
    point3d(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }
    float_double norm(){
        return sqrt(x*x+y*y+z*z);
    }
    float_double distance(point3d p){
        return sqrt((x-p.x)*(x-p.x)+(y-p.y)*(y-p.y)+(z-p.z)*(z-p.z));
    }
    float_double square_distance(point3d p){
        return (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y)+(z-p.z)*(z-p.z);
    }
    float_double square_norm(){
        return x*x+y*y+z*z;
    }
    //overload []
    float_double operator[](dimension_type i){
        if (i == 0){
            return x;
        }
        if (i == 1){
            return y;
        }
        if (i == 2){
            return z;
        }
        //raise an error
        //throw std::invalid_argument("i != 0,1,2");
        exit(6);
    }
    point3d operator+(point3d p){
        return point3d(x+p.x, y+p.y, z+p.z);
    }
    point3d operator-(point3d p){
        return point3d(x-p.x, y-p.y, z-p.z);
    }
    point3d operator*(float_double a){
        return point3d(x*a, y*a, z*a);
    }
    friend point3d operator*(float_double a, point3d p){
        return point3d(a*p.x, a*p.y, a*p.z);
    }
    float_double operator*(point3d p){
        return x*p.x+y*p.y+z*p.z;
    }
    friend std::ostream& operator<<(std::ostream& os, const point3d& p){
        os << "point3d(" << p.x << "," << p.y << "," << p.z << ")";
        return os;
    }
    bool operator==(point3d p){
        return x == p.x && y == p.y && z == p.z;
    }
};





template <dimension_type N_dim, dimension_type N_dim_point>
class gaussian_kernel{
    public:
    float_double sigma[N_dim*N_dim];//the covariance matrix
    point<N_dim_point> mu;
    float_double det_sigma;
    float_double log_weight;
    float_double log_coef_sigma;
    gaussian_kernel(point<N_dim_point> mu, float_double log_weight, float_double sigma[N_dim]){
        this->mu = mu;
        this->log_weight = log_weight;
        for(dimension_type i=0;i<N_dim*N_dim;i++){
            this->sigma[i] = sigma[i];
        }
        this->det_sigma = det(N_dim, this->sigma);
        if(N_dim == 1){
            log_coef_sigma = log(sigma[0]*sqrt(2*pi));
        }
        else{
            log_coef_sigma = log(sqrt(pow(pi ,N_dim)/det_sigma));
        }
    }
    gaussian_kernel(){
        this->mu = point<N_dim_point>();
        this->log_weight = 0;
        for(dimension_type i=0;i<N_dim*N_dim;i++){
            this->sigma[i] = 0;
        }
        this->det_sigma = 0;
        this->log_coef_sigma = 0;
    }
    float_double operator()(point<N_dim_point> x){
        if(N_dim==1){
            float_double d = mu.square_distance(x);
            //std::cout << mu << " " << x << " " << d << std::endl;
            //return weight*exp(-d/(2*sigma[0]*sigma[0]))/sqrt(2*pi*sigma[0]*sigma[0]);
            //return exp(log_weight - d/(2*sigma[0]*sigma[0]))/(sigma[0]*sqrt(2*pi));
            //return log_weight - d/(2*sigma[0]*sigma[0])-log(sigma[0]*sqrt(2*pi));
            return log_weight - d/(2*sigma[0]*sigma[0])-log_coef_sigma;
        }
        //assert N_dim == N_dim_point
        if (N_dim != N_dim_point){
            //throw std::invalid_argument("N_dim != N_dim_point");
            exit(2);
        }
        float_double x_mu[N_dim];
        for(dimension_type i=0;i<N_dim;i++){
            x_mu[i] = x.x[i] - mu.x[i];
        }
        float_double x_mu_transpose[N_dim];
        for(dimension_type i=0;i<N_dim;i++){
            x_mu_transpose[i] = x_mu[i];
        }
        float_double x_mu_sigma[N_dim];
        for(dimension_type i=0;i<N_dim;i++){
            x_mu_sigma[i] = 0;
            for(dimension_type j=0;j<N_dim;j++){
                x_mu_sigma[i] += x_mu_transpose[j]*sigma[j*N_dim+i];
            }
        }
        float_double res = 0;
        for(dimension_type i=0;i<N_dim;i++){
            res += x_mu[i]*x_mu_sigma[i];
        }
        if (res < 0){
            //imposible because the matrix is semi positive definite
            std::cout << "error res < 0" << std::endl;
            std::cout << res << std::endl;
            std::cout << mu << std::endl;
            std::cout << x << std::endl;
            std::cout << x-mu << std::endl;
            std::cout << log_weight << std::endl;

            for(dimension_type i=0;i<N_dim;i++){
                for(dimension_type j=0;j<N_dim;j++){
                    std::cout << sigma[i*N_dim+j] << " ";
                }
                std::cout << std::endl;
            }

            exit(4);
        }
        /*std::cout << det_sigma << std::endl;
        //cout first line of sigma
        for(dimension_type i=0;i<N_dim;i++){
            std::cout << sigma[i] << " ";
        }
        std::cout << std::endl;
        //cout second line of sigma
        for(dimension_type i=0;i<N_dim;i++){
            std::cout << sigma[i+N_dim] << " ";
        }
        std::cout << std::endl;
        */
       //std::cout << res << std::endl;
       //std::cout << det_sigma << std::endl;
        //return weight*exp(-res)/sqrt(pow(pi ,N_dim)/det_sigma);
        //return exp(log_weight - res)/sqrt(pow(pi ,N_dim)/det_sigma);
        //return log_weight - res - log(sqrt(pow(pi ,N_dim)/det_sigma));
        return log_weight - res - log_coef_sigma;
    }
    float_double distance(point<N_dim_point> x){
        //if the kernel does not have weight infinit
        if(!(std::isinf(log_weight))){
            //std::cout << this->operator()(x) << std::endl;
            return -std::max(this->operator()(x), negligeable_val_when_exp);
        }
        else{
            //return minimum value of a float_double
            return -std::numeric_limits<float_double>::max();
        }
    }
    gaussian_kernel<1,N_dim_point> convert(){
        if(N_dim == 1){
            //copy of myself
            return gaussian_kernel<1,N_dim_point>(mu, log_weight, sigma);
        }
        float_double min_eigan = min_eigan_value(N_dim, sigma);
        float_double sigma_[1] = {(float_double)sqrt(1/(2*min_eigan))};
        /*std::cout << "convert" << std::endl;
        std::cout << mu << std::endl;
        std::cout << log_weight << std::endl;
        std::cout << min_eigan << std::endl;
        std::cout << 1/(2*min_eigan) << std::endl;
        std::cout << sigma_[0] << std::endl;
        std::cout << log(sigma_[0]*sqrt(2*pi)) << std::endl;
        std::cout << log(sqrt(pow(pi ,N_dim)/det_sigma)) << std::endl;
        std::cout << log_weight+log(sigma[0]*sqrt(2*pi))-log(sqrt(pow(pi ,N_dim)/det_sigma)) << std::endl;
        */
        return gaussian_kernel<1,N_dim_point>(mu, log_weight+log(sigma_[0]*sqrt(2*pi))-log(sqrt(pow(pi ,N_dim)/det_sigma)), sigma_);
    }
    friend std::ostream& operator<<(std::ostream& os, const gaussian_kernel<N_dim, N_dim_point>& g){
        os << "gaussian_kernel(" << g.mu << "," << g.log_weight <<",[";
        for(dimension_type i=0;i<N_dim*N_dim-1;i++){
            os << g.sigma[i] << ",";
        }
        os << g.sigma[N_dim*N_dim-1] << "])";
        return os;
    }
};
class gaussian_kernel2_1D{
    public:
        float_double scale;
        point3d mu;
        float_double log_weight;
    gaussian_kernel2_1D(point3d mu, float_double log_weight, float_double scale){
        this->mu = mu;
        this->log_weight = log_weight;
        this->scale = scale;
    }
    gaussian_kernel2_1D(){
        this->mu = point3d();
        this->log_weight = 0;
        this->scale = 0;
    }
    float_double operator()(point3d x){
        float_double d = mu.square_distance(x);
        return log_weight - d/(2*scale*scale);
    }
    float_double distance(point3d x){
        //if the kernel does not have weight infinit
        if(!(std::isinf(log_weight))){
            return -std::max(this->operator()(x), negligeable_val_when_exp);
        }
        else{
            //return minimum value of a float_double
            return -std::numeric_limits<float_double>::max();
        }
    }
    gaussian_kernel2_1D convert(){
        float_double max_eigan = scale*scale;
        float_double scale_ = (float_double)sqrt(max_eigan);
        return gaussian_kernel2_1D(mu, log_weight, scale_);
    }
    friend std::ostream& operator<<(std::ostream& os, const gaussian_kernel2_1D& g){
        os << "gaussian_kernel(" << g.mu << "," << g.log_weight <<",[";
        os << g.scale << "])";
        return os;
    }
};
class gaussian_kernel2_3D{
    public:
        float_double scale[3];
        float_double quaternions[4];
        point3d mu;
        float_double log_weight;
    gaussian_kernel2_3D(point3d mu, float_double log_weight, float_double scale[3], float_double* quaternions){
        this->mu = mu;
        this->log_weight = log_weight;
        for(dimension_type i=0;i<3;i++){
            this->scale[i] = scale[i];
        }
        for(dimension_type i=0;i<4;i++){
            this->quaternions[i] = quaternions[i];
        }
    }
    gaussian_kernel2_3D(){
        this->mu = point3d();
        this->log_weight = 0;
        for(dimension_type i=0;i<3;i++){
            this->scale[i] = 0;
        }
        for(dimension_type i=0;i<4;i++){
            this->quaternions[i] = 0;
        }
    }
    float_double operator()(point3d x){
        // log_weight - (x-mu)^T (scale*R)"^-2"(x-mu)
        // R is the rotation matrix
        // (x-mu) is the vector from mu to x
        // (x-mu)^T (scale*R)"^-2"(x-mu) = (x-mu)^T(scale*R)^-1^T*(scale*R)^-1(x-mu)
        //                               = ((scale*R)^-1(x-mu))^T*((scale*R)^-1(x-mu))
        // (scale*R)^-1 = scale^-1*R^-1
        //R^-1 = R^T

        point3d x_mu = x - mu;
        float_double R[3][3];
        R[0][0] = 1-2*quaternions[2]*quaternions[2]-2*quaternions[3]*quaternions[3];
        R[0][1] = 2*quaternions[1]*quaternions[2]-2*quaternions[0]*quaternions[3];
        R[0][2] = 2*quaternions[0]*quaternions[2]+2*quaternions[1]*quaternions[3];
        R[1][0] = 2*quaternions[1]*quaternions[2]+2*quaternions[0]*quaternions[3];
        R[1][1] = 1-2*quaternions[1]*quaternions[1]-2*quaternions[3]*quaternions[3];
        R[1][2] = 2*quaternions[2]*quaternions[3]-2*quaternions[0]*quaternions[1];
        R[2][0] = 2*quaternions[1]*quaternions[3]-2*quaternions[0]*quaternions[2];
        R[2][1] = 2*quaternions[0]*quaternions[1]+2*quaternions[2]*quaternions[3];
        R[2][2] = 1-2*quaternions[1]*quaternions[1]-2*quaternions[2]*quaternions[2];

        point3d R_inv_x_mu;
        /*for(dimension_type i=0;i<3;i++){
            R_inv_x_mu[i] = 0;
            for(dimension_type j=0;j<3;j++){
                //R_inv_x_mu[i] += R_inv[i][j]*x_mu[j];
                R_inv_x_mu[i] += R[j][i]*x_mu[j];
            }
        }*/
        R_inv_x_mu.x = R[0][0]*x_mu.x + R[0][1]*x_mu.y + R[0][2]*x_mu.z;
        R_inv_x_mu.y = R[1][0]*x_mu.x + R[1][1]*x_mu.y + R[1][2]*x_mu.z;
        R_inv_x_mu.z = R[2][0]*x_mu.x + R[2][1]*x_mu.y + R[2][2]*x_mu.z;
        point3d scale_inv_R_inv_x_mu;
        /*for(dimension_type i=0;i<3;i++){
            scale_inv_R_inv_x_mu[i] = R_inv_x_mu[i]/scale[i];
        }*/
        scale_inv_R_inv_x_mu.x = R_inv_x_mu.x/scale[0];
        scale_inv_R_inv_x_mu.y = R_inv_x_mu.y/scale[1];
        scale_inv_R_inv_x_mu.z = R_inv_x_mu.z/scale[2];
        /*float_double res = 0;
        for(dimension_type i=0;i<3;i++){
            res += scale_inv_R_inv_x_mu[i]*scale_inv_R_inv_x_mu[i];
        }*/
        float_double res = scale_inv_R_inv_x_mu.square_norm();
        return log_weight - res/2;
    }
    float_double distance(point3d x){
        //if the kernel does not have weight infinit
        if(!(std::isinf(log_weight))){
            return -std::max(this->operator()(x), negligeable_val_when_exp);
        }
        else{
            //return minimum value of a float_double
            return -std::numeric_limits<float_double>::max();
        }
    }
    gaussian_kernel2_1D convert(){
        float_double max_eigan = std::max(scale[0]*scale[0], std::max(scale[1]*scale[1], scale[2]*scale[2]));
        return gaussian_kernel2_1D(mu, log_weight, (float_double)sqrt(max_eigan));
    }
    friend std::ostream& operator<<(std::ostream& os, const gaussian_kernel2_3D& g){
        os << "gaussian_kernel(" << g.mu << "," << g.log_weight <<",[";
        for(dimension_type i=0;i<3-1;i++){
            os << g.scale[i] << ",";
        }
        os << g.scale[3-1] << "],[";
        for(dimension_type i=0;i<3;i++){
            os << g.quaternions[i] << ",";
        }
        os << g.quaternions[3] << "])";
        return os;
    }
};




template<dimension_type N_dim_point>
float_double pow(point<N_dim_point> p, size_t n){
    //assert n ==2
    if (n == 2){
        return p*p;
    }
    //raise an error
    //throw std::invalid_argument("n != 2");
    exit(3);
}
float_double pow(point3d p, size_t n){
    //assert n ==2
    if (n == 2){
        return p*p;
    }
    //raise an error
    //throw std::invalid_argument("n != 2");
    exit(7);
}

template<dimension_type N_dim_point>
gaussian_kernel<1, N_dim_point> max_gaussian_kernel(gaussian_kernel<1, N_dim_point>g1, gaussian_kernel<1, N_dim_point> g2){
    //only for 1D for now
    //make the first kernel the one with the biggest sigma
    if (g1.sigma[0] < g2.sigma[0]){
        return max_gaussian_kernel(g2, g1);
    }
    //if the two kernels have the same mu
    if (g1.mu == g2.mu){
        //return a kernel with a max sigma and weight of center mu
        float_double sigma_[1] = {std::max(g1.sigma[0], g2.sigma[0])};
        //return gaussian_kernel<1,N_dim_point>(g1.mu, std::max(g1.weight, g2.weight), sigma_);
        return gaussian_kernel<1,N_dim_point>(g1.mu, std::max(g1.log_weight, g2.log_weight), sigma_);
    }
    //if if the sigma is the same then no solution
    if (g1.sigma[0] == g2.sigma[0]){
        /*//raise an error
        throw std::invalid_argument("no solution");*/
        //kill the program
        //exit(1);
        float_double sigma1_ = 1.1*g1.sigma[0];
        float_double log_weight1_ = g1.log_weight +log(1.1);
        float_double sigma1[1] = {sigma1_};
        gaussian_kernel<1,N_dim_point> g1_ = gaussian_kernel<1,N_dim_point>(g1.mu, log_weight1_, sigma1);
    }
    point<N_dim_point> mu1 = g1.mu;
    point<N_dim_point> mu2 = g2.mu;
    float_double sigma1 = g1.sigma[0];
    float_double sigma2 = g2.sigma[0];
    //float_double weight1 = g1.weight;
    //float_double weight2 = g2.weight;
    float_double log_weight1 = g1.log_weight;
    float_double log_weight2 = g2.log_weight;
    point<N_dim_point> mu = mu1;
    float_double sigma = sigma1;
    float_double sub1 = pow(sigma1*sigma1 * mu2-sigma2*sigma2 * mu1, 2)
        /
        (2*(sigma1*sigma1 - sigma2*sigma2)*sigma1*sigma1*sigma2*sigma2);
    float_double sub2 = (sigma1*sigma1 * mu2* mu2-sigma2*sigma2 * mu1* mu1)
        /
        (2*sigma1*sigma1 *sigma2*sigma2);
    //float_double sub3 = log(weight2*sigma1/(sigma2));
    float_double sub3 = log_weight2+log(sigma1/sigma2);

    //float_double weight = std::max(weight1, exp(sub1-sub2+sub3));
    float_double log_weight = std::max(log_weight1, sub1-sub2+sub3);
    float_double mu_[N_dim_point];
    for(dimension_type i=0;i<N_dim_point;i++){
        mu_[i] = mu.x[i];
    }
    float_double sigma_[1] = {sigma};
    //return gaussian_kernel<1,N_dim_point>(mu_, weight, sigma_);
    return gaussian_kernel<1,N_dim_point>(mu_, log_weight, sigma_);
}

template<dimension_type N_dim_point>
gaussian_kernel<1, N_dim_point> max_gaussian_kernel_list(std::vector<gaussian_kernel<1, N_dim_point>>* kernels,array_indexes_type start, array_indexes_type end){
    //std::cout << "max_gaussian_kernel_list " << start << " " << end << std::endl;
    array_indexes_type size = end-start+1;
    //std::cout << size << std::endl;
    if (size == 1){
        return (*kernels)[0];
    }
    if (size == 2){
        return max_gaussian_kernel((*kernels)[0], (*kernels)[1]);
    }
    //find the max sigma index
    array_indexes_type max_index = 0;
    for(array_indexes_type i=1;i<size;i++){
        if ((*kernels)[i+start].sigma[0] > (*kernels)[max_index+start].sigma[0]){
            max_index = i;
        }
    }
    point<N_dim_point> mu = (*kernels)[max_index+start].mu;
    float_double sigma = (*kernels)[max_index+start].sigma[0];
    float_double log_weight = (*kernels)[max_index+start].log_weight;
    for(array_indexes_type i=0;i<size;i++){
        if (i != max_index){
            float_double sub1 = pow(sigma*sigma * (*kernels)[i+start].mu-sigma*sigma * mu, 2)
                /
                (2*(sigma*sigma - sigma*sigma)*sigma*sigma*sigma*sigma);
            float_double sub2 = (sigma*sigma * (*kernels)[i+start].mu* (*kernels)[i+start].mu-sigma*sigma * mu* mu)
                /
                (2*sigma*sigma *sigma*sigma);
            float_double sub3 = (*kernels)[i+start].log_weight+log(sigma/sigma);
            log_weight = std::max(log_weight, sub1-sub2+sub3);
        }
    }
    float_double mu_[N_dim_point];
    for(dimension_type i=0;i<N_dim_point;i++){
        mu_[i] = mu.x[i];
    }
    float_double sigma_[1] = {sigma};
    return gaussian_kernel<1,N_dim_point>(mu_, log_weight, sigma_);  
}

gaussian_kernel2_1D max_gaussian_kernel(gaussian_kernel2_1D g1, gaussian_kernel2_1D g2){
    //only for 1D for now
    //make the first kernel the one with the biggest scale
    if (g1.scale < g2.scale){
        return max_gaussian_kernel(g2, g1);
    }
    //if the two kernels have the same mu
    if (g1.mu == g2.mu){
        //return a kernel with a max scale and weight of center mu
        float_double scale_ = std::max(g1.scale, g2.scale);
        return gaussian_kernel2_1D(g1.mu, std::max(g1.log_weight, g2.log_weight), scale_);
    }
    //if if the scale is the same then no solution
    if (g1.scale == g2.scale){
        /*//raise an error
        throw std::invalid_argument("no solution");*/
        //kill the program
        //exit(1);
        float_double scale1_ = 1.1*g1.scale;
        float_double log_weight1_ = g1.log_weight;
        float_double scale1 = scale1_;
        gaussian_kernel2_1D g1_ = gaussian_kernel2_1D(g1.mu, log_weight1_, scale1);
    }
    point3d mu1 = g1.mu;
    point3d mu2 = g2.mu;
    float_double scale1 = g1.scale;
    float_double scale2 = g2.scale;
    float_double log_weight1 = g1.log_weight;
    float_double log_weight2 = g2.log_weight;
    float_double scale = scale1;
    float_double sub1 = pow(scale1*scale1 * mu2-scale2*scale2 * mu1, 2)
        /
        (2*(scale1*scale1 - scale2*scale2)*scale1*scale1*scale2*scale2);
    float_double sub2 = (scale1*scale1 * mu2* mu2-scale2*scale2 * mu1* mu1)
        /
        (2*scale1*scale1 *scale2*scale2);
    float_double sub3 = log_weight2;
    float_double log_weight = std::max(log_weight1, sub1-sub2+sub3);
    float_double scale_ = scale;

    return gaussian_kernel2_1D(mu1, log_weight, scale_);
}

gaussian_kernel2_1D max_gaussian_kernel_list(std::vector<gaussian_kernel2_1D>* kernels,array_indexes_type start, array_indexes_type end){
    //std::cout << "max_gaussian_kernel_list " << start << " " << end << std::endl;
    array_indexes_type size = end-start+1;
    //std::cout << size << std::endl;
    if (size == 1){
        return (*kernels)[0];
    }
    if (size == 2){
        return max_gaussian_kernel((*kernels)[0], (*kernels)[1]);
    }
    //find the max scale index
    array_indexes_type max_index = 0;
    for(array_indexes_type i=1;i<size;i++){
        if ((*kernels)[i+start].scale > (*kernels)[max_index+start].scale){
            max_index = i;
        }
    }
    point3d mu = (*kernels)[max_index+start].mu;
    float_double scale = (*kernels)[max_index+start].scale;
    float_double log_weight = (*kernels)[max_index+start].log_weight;
    for(array_indexes_type i=0;i<size;i++){
        if (i != max_index){
            float_double sub1 = pow(scale*scale * (*kernels)[i+start].mu-scale*scale * mu, 2)
                /
                (2*(scale*scale - scale*scale)*scale*scale*scale*scale*2);
            float_double sub2 = (scale*scale * (*kernels)[i+start].mu* (*kernels)[i+start].mu-scale*scale * mu* mu)
                /
                (2*scale*scale *scale*scale);
            float_double sub3 = (*kernels)[i+start].log_weight;
            log_weight = std::max(log_weight, sub1-sub2+sub3);
        }
    }
    float_double scale_ = scale;
    return gaussian_kernel2_1D(mu, log_weight, scale_);  
}



template <dimension_type N_dim, dimension_type N_dim_point>
class kdtree_node{
    public:
    std::vector<gaussian_kernel<N_dim,N_dim_point>>* kernels;
    array_indexes_type start, end;
    gaussian_kernel<1,N_dim_point> kernel_max;
    dimension_type split_dim;
    kdtree_node<N_dim,N_dim_point>* left;
    kdtree_node<N_dim,N_dim_point>* right;
    float_double range_point_min[N_dim_point];
    float_double range_point_max[N_dim_point];
    //float_double range_weight_max;
    float_double range_log_weight_max;
    float_double range_sigma_min;
    float_double range_sigma_max;
    kdtree_node(std::vector<gaussian_kernel<N_dim,N_dim_point>>* ks, array_indexes_type start, array_indexes_type end, dimension_type split_dim){
        this->kernels = ks;
        this->start = start;
        this->end = end;
        this->split_dim = split_dim;
        if (start < end){
            build();
            for(dimension_type i=0;i<N_dim_point;i++){
                //compare left and right to find the min and max of the range of the point
                range_point_min[i] = std::min(left->range_point_min[i], right->range_point_min[i]);
                range_point_max[i] = std::max(left->range_point_max[i], right->range_point_max[i]);
            }
            //range_weight_max = std::max(left->range_weight_max, right->range_weight_max);
            range_log_weight_max = std::max(left->range_log_weight_max, right->range_log_weight_max);
            range_sigma_min = std::min(left->range_sigma_min, right->range_sigma_min);
            range_sigma_max = std::max(left->range_sigma_max, right->range_sigma_max);
        }
        else{
            kernel_max = (*ks)[start].convert();
            for(dimension_type i=0;i<N_dim_point;i++){
                range_point_min[i] = kernel_max.mu.x[i];
                range_point_max[i] = kernel_max.mu.x[i];
            }
            //range_weight_max = kernel_max.weight;
            range_log_weight_max = kernel_max.log_weight;
            range_sigma_min = kernel_max.sigma[0];
            range_sigma_max = kernel_max.sigma[0];
        }
        //std::cout <<"a"<< start << " " << end << std::endl;
    }
    kdtree_node(std::vector<gaussian_kernel<N_dim,N_dim_point>>* ks){
        this->kernels = ks;
        this->start = 0;
        this->end = ks->size()-1;
        this->split_dim = 0;
        if (start < end){
            build();
            for(dimension_type i=0;i<N_dim_point;i++){
                //compare left and right to find the min and max of the range of the point
                range_point_min[i] = std::min(left->range_point_min[i], right->range_point_min[i]);
                range_point_max[i] = std::max(left->range_point_max[i], right->range_point_max[i]);
            }
            //range_weight_max = std::max(left->range_weight_max, right->range_weight_max);
            range_log_weight_max = std::max(left->range_log_weight_max, right->range_log_weight_max);
            range_sigma_min = std::min(left->range_sigma_min, right->range_sigma_min);
            range_sigma_max = std::max(left->range_sigma_max, right->range_sigma_max);
        }
        else{
            kernel_max = (*ks)[start].convert();
            for(dimension_type i=0;i<N_dim_point;i++){
                range_point_min[i] = kernel_max.mu.x[i];
                range_point_max[i] = kernel_max.mu.x[i];
            }
            //range_weight_max = kernel_max.weight;
            range_log_weight_max = kernel_max.log_weight;
            range_sigma_min = kernel_max.sigma[0];
            range_sigma_max = kernel_max.sigma[0];
        }
        //std::cout <<"b"<< start << " " << end << std::endl;
    }
    void build(){
        //std::cout << start << " " << end << std::endl;
        std::sort(kernels->begin()+start, kernels->begin()+end+1, [this](gaussian_kernel<N_dim,N_dim_point> g1, gaussian_kernel<N_dim,N_dim_point> g2){
            return g1.mu.x[split_dim] < g2.mu.x[split_dim];
        });
        std::vector<gaussian_kernel<1,N_dim_point>> kernels_;
        if(Beter_majoration or (!Simple_heuristic && !Simple_heuristic2)){
            for (array_indexes_type i=start;i<=end;i++){
                kernels_.push_back((*kernels)[i].convert());
            }
        }
        if(Simple_heuristic){
            array_indexes_type new_size = (end-start)/2;
            array_indexes_type left_end = start + new_size;
            array_indexes_type right_start = left_end + 1;
            //std::cout << start << " " << left_end << " " << right_start << " " << end << std::endl;
            left = new kdtree_node<N_dim,N_dim_point>(kernels, start, left_end, (split_dim+1)%N_dim);
            right = new kdtree_node<N_dim,N_dim_point>(kernels, right_start, end, (split_dim+1)%N_dim);
            //kernel_max = max_gaussian_kernel(left->kernel_max, right->kernel_max);
        }
        else{
            point<N_dim_point> midle;
            float_double max_dim[N_dim_point];
            float_double min_dim[N_dim_point];
            for(dimension_type i=0;i<N_dim_point;i++){
                max_dim[i] = std::numeric_limits<float_double>::min();
                min_dim[i] = std::numeric_limits<float_double>::max();
            }
            
            for(array_indexes_type j=start;j<=end;j++){
                for(dimension_type i=0;i<N_dim_point;i++){
                    max_dim[i] = std::max(max_dim[i], (*kernels)[j].mu.x[i]);
                    min_dim[i] = std::min(min_dim[i], (*kernels)[j].mu.x[i]);
                }
            }
            for(dimension_type i=0;i<N_dim_point;i++){
                midle.x[i] = (max_dim[i]+min_dim[i])/2;
            }
            //search by dichotomy the split
            //the split is the index where the distance to the left becomes bigger than the distance to the right
            array_indexes_type split = 0;
            array_indexes_type st = 0;
            array_indexes_type en = end-start;
            if(Simple_heuristic2){
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = std::numeric_limits<float_double>::max();
                    float_double distance_right = std::numeric_limits<float_double>::max();
                    for(array_indexes_type i=0;i<split;i++){
                        distance_left = std::min(distance_left, (*kernels)[i].distance(midle));
                    }
                    for(array_indexes_type i=split;i<end-start;i++){
                        distance_right = std::min(distance_right, (*kernels)[i].distance(midle));
                    }
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                    //std::cout << distance_left << " " << distance_right << std::endl;
                }
            }
            else{
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = max_gaussian_kernel_list(&kernels_, 0, split).distance(midle);
                    float_double distance_right = max_gaussian_kernel_list(&kernels_, split+1, kernels_.size()-1).distance(midle);
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                    //std::cout << distance_left << " " << distance_right << std::endl;
                }
            }
            split = std::max(split, (array_indexes_type)1);
            split = std::min(split, end-start);
            //std::cout << end-start << std::endl;
            //std::cout << split<<" " << start<<" " << split+start<<" " << split+start+1<<" " << end << std::endl;
            left = new kdtree_node<N_dim,N_dim_point>(kernels, start, split+start-1, (split_dim+1)%N_dim);
            right = new kdtree_node<N_dim,N_dim_point>(kernels, split+start, end, (split_dim+1)%N_dim);
        }

        if(Beter_majoration){
            kernel_max = max_gaussian_kernel_list(&kernels_, 0, kernels_.size()-1);
        }
        else{
            kernel_max = max_gaussian_kernel(left->kernel_max, right->kernel_max);
        }
    }
    void search_1nn(point<N_dim_point> x, gaussian_kernel<N_dim,N_dim_point>* res,float_double* min_dist,size_t* debug_counter = nullptr){
        //std::cout << start << " " << end << std::endl;
        if (debug_counter != nullptr){
                (*debug_counter)++;
        }
        if (start == end){
            //std::cout << *res << " "<< *min_dist << std::endl;
            if ((*kernels)[start].distance(x) < *min_dist){
                *res = (*kernels)[start];
                *min_dist = (*kernels)[start].distance(x);
                //std::cout << "new min" << std::endl;
            }
            return;
        }
        float_double dist_left = left->distance(x);
        float_double dist_right = right->distance(x);
        if (dist_left < dist_right){
            //std::cout << "starting left " << dist_left << " " << dist_right << std::endl; 
            if (dist_left < *min_dist){
                //std::cout << "left" << std::endl;
                left->search_1nn(x, res, min_dist, debug_counter);
                if (dist_right < *min_dist){
                    //std::cout << "right" << std::endl;
                    right->search_1nn(x, res, min_dist, debug_counter);
                }
                else{
                    //std::cout << "skip right" << std::endl;
                    //std::cout << "because" << dist_right << " " << *min_dist << std::endl;
                }
            }
            else{
                //std::cout << "skip" << std::endl;
                //std::cout << "because" << dist_left << " " << *min_dist << std::endl;
            }
        }else{
            //std::cout << "starting right " << dist_left << " " << dist_right << std::endl;
            if (dist_right < *min_dist){
                //std::cout << "right" << std::endl;
                right->search_1nn(x, res, min_dist, debug_counter);
                if (dist_left < *min_dist){
                    //std::cout << "left" << std::endl;967
                    left->search_1nn(x, res, min_dist, debug_counter);
                }
                else{
                    //std::cout << "skip left" << std::endl;
                    //std::cout << "because" << dist_left << " " << *min_dist << std::endl;
                }
            }
            else{
                //std::cout << "skip" << std::endl;
                //std::cout << "because" << dist_right << " " << *min_dist << std::endl;
            }
        }
    }
    void search_knn(point<N_dim_point> x, float_double* min_dist,gaussian_kernel<N_dim,N_dim_point>** found, array_indexes_type k, array_indexes_type* number_found){
        //std::cout << start << " " << end << std::endl;
        if (start == end){
            if ((*kernels)[start].distance(x) < *min_dist){
                //std::cout << *number_found << std::endl;
                if (*number_found == 0){
                    found[0] = new gaussian_kernel<N_dim,N_dim_point>((*kernels)[start]);
                }
                else{
                    float_double dist = (*kernels)[start].distance(x);
                    /*std::cout << dist << " " << *min_dist << std::endl;
                    for(array_indexes_type i=0;i<*number_found;i++){
                        std::cout << found[i]->distance(x) << " ";
                    }
                    std::cout << std::endl;
                    */
                    //find by dichotomy the place of the new kernel
                    array_indexes_type i = 0;
                    array_indexes_type j = *number_found;
                    while(j > i){
                        array_indexes_type m = (i+j)/2;
                        if (found[m]->distance(x) < dist){
                            i=m+1;
                        }
                        else{
                            j=m;
                        }
                    }
                    //insert the new kernel
                    //std::cout << i << std::endl;
                    if(*number_found == k){
                        free(found[k-1]);
                    }
                    for(array_indexes_type l=*number_found;l>i;l--){
                        found[l] = found[l-1];
                    }
                    /*for (int l=0;l<N_dim_point;l++){
                        std::cout << found[i]->distance(x) << " ";
                    }
                    std::cout << std::endl;
                    */
                    
                    found[i] = new gaussian_kernel<N_dim,N_dim_point>((*kernels)[start]);
                    /*for(dimension_type l=0;l<N_dim_point;l++){
                        std::cout << found[i]->distance(x) << " ";
                    }
                    std::cout << std::endl;
                    */
                }
                *number_found = std::min(k, *number_found+1);
                if(*number_found == k){
                    *min_dist = found[k-1]->distance(x);
                }
            }
            return;
        }
        float_double dist_left = left->distance(x);
        float_double dist_right = right->distance(x);
        if (dist_left < dist_right){
            if (dist_left < *min_dist && dist_left < -negligeable_val_when_exp){
                left->search_knn(x, min_dist, found, k, number_found);
                if (dist_right < *min_dist && dist_right < -negligeable_val_when_exp){
                    right->search_knn(x, min_dist, found, k, number_found);
                }
            }
        }else{
            if (dist_right < *min_dist && dist_right < -negligeable_val_when_exp){
                right->search_knn(x, min_dist, found, k, number_found);
                if (dist_left < *min_dist && dist_left < -negligeable_val_when_exp){
                    left->search_knn(x, min_dist, found, k, number_found);
                }
            }
        }
    }
    float_double distance(point<N_dim_point> x){
        if (start == end){
            return (*kernels)[start].distance(x);
        }    
        float_double projection_x[N_dim_point];
        for (dimension_type i=0;i<N_dim_point;i++){
            projection_x[i] = x.x[i];
            if (x.x[i] < range_point_min[i]){
                projection_x[i] = range_point_min[i];
            }
            if (x.x[i] > range_point_max[i]){
                projection_x[i] = range_point_max[i];
            }
        }
        point<N_dim_point> projection(projection_x);
        float_double dist = x.distance(projection);
        //float_double d1 =  - (1/(range_sigma_min*sqrt(2*pi)))*exp(range_log_weight_max-dist*dist/(2*range_sigma_max*range_sigma_max));
        float_double d1_ = (range_log_weight_max-dist*dist/(2*range_sigma_max*range_sigma_max) - log(range_sigma_min*sqrt(2*pi)));
        float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
        //std::cout << d1_ << " " << d1 << std::endl;
        
        float_double d2 = kernel_max.distance(x);
        //std::cout << d1 << " " << d2 << std::endl;
        return std::max(d1, d2);
    }
    friend std::ostream& operator<<(std::ostream& os, const kdtree_node<N_dim,N_dim_point> &k){
        if (k.start == k.end){
            os << "kdtree_node(" << k.start << "," << k.end << "," << k.split_dim << "," << k.kernel_max << ")";
            return os;
        }
        os << "kdtree_node(" << k.start << "," << k.end << "," << k.split_dim << ",";
        for(dimension_type i=0;i<N_dim_point;i++){
            os << "[" << k.range_point_min[i] << "," << k.range_point_max[i] << "],";
        }
        os<< k.kernel_max << *(k.left) << *(k.right) << ")";
        return os;
    }
};

class kdtree_node2{
    public:
    //array_indexes_type start, end;
    //gaussian_kernel2_1D kernel_max;
    //dimension_type split_dim;
    kdtree_node2* left;
    kdtree_node2* right;
    float_double range_point_min[3];
    float_double range_point_max[3];
    float_double range_log_weight_max;
    float_double range_scale_max;
    //array_indexes_type split_index;
    kdtree_node2(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type start, array_indexes_type end, dimension_type split_dim){
        //std::cout << start << " " << end << std::endl;
        //this->start = start;
        //this->end = end;
        //this->split_dim = split_dim;
        if (start < end){
            build(ks,split_dim, start, end);
            for(dimension_type i=0;i<3;i++){
                range_point_min[i] = std::min(left->range_point_min[i], right->range_point_min[i]);
                range_point_max[i] = std::max(left->range_point_max[i], right->range_point_max[i]);
            }
            range_log_weight_max = std::max(left->range_log_weight_max, right->range_log_weight_max);
            range_scale_max = std::max(left->range_scale_max, right->range_scale_max);
        }
        else{
            gaussian_kernel2_1D kernel_max = (*ks)[start].convert();
            for(dimension_type i=0;i<3;i++){
                range_point_min[i] = kernel_max.mu[i];
                range_point_max[i] = kernel_max.mu[i];
            }
            range_log_weight_max = kernel_max.log_weight;
            range_scale_max = kernel_max.scale;
            left = nullptr;
            //save the start index in right pointer to save memory, to do so convert the index into a pointer
            right = (kdtree_node2*)((size_t) start);
        }
    }
    kdtree_node2(std::vector<gaussian_kernel2_3D>* ks){
        //std::cout << "a" << std::endl;
        //this->start = 0;
        //this->end = ks->size()-1;
        //this->split_dim = 0;
        //std::cout << "b" << start << " " << end << std::endl;
        if (ks->size() > 1){
            build(ks,0, 0, ks->size()-1);
            for(dimension_type i=0;i<3;i++){
                range_point_min[i] = std::min(left->range_point_min[i], right->range_point_min[i]);
                range_point_max[i] = std::max(left->range_point_max[i], right->range_point_max[i]);
            }
            range_log_weight_max = std::max(left->range_log_weight_max, right->range_log_weight_max);
            range_scale_max = std::max(left->range_scale_max, right->range_scale_max);
        }
        else{
            gaussian_kernel2_1D kernel_max = (*ks)[0].convert();
            for(dimension_type i=0;i<3;i++){
                range_point_min[i] = kernel_max.mu[i];
                range_point_max[i] = kernel_max.mu[i];
            }
            range_log_weight_max = kernel_max.log_weight;
            range_scale_max = kernel_max.scale;
            left = nullptr;
            //save the start index in right pointer to save memory, to do so convert the index into a pointer
            right = (kdtree_node2*)0;
        }
    }
    void build(std::vector<gaussian_kernel2_3D>* kernels, dimension_type split_dim, array_indexes_type start, array_indexes_type end){
        //std::cout << "build" << std::endl;
        std::sort(kernels->begin()+start, kernels->begin()+end+1, [&split_dim](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
            return g1.mu[split_dim] < g2.mu[split_dim];
        });
        //std::cout << "sort" << std::endl;
        std::vector<gaussian_kernel2_1D> kernels_;
        if(Beter_majoration or (!Simple_heuristic && !Simple_heuristic2)){
            for (array_indexes_type i=start;i<=end;i++){
                kernels_.push_back((*kernels)[i].convert());
            }
            //std::cout << "convert" << std::endl;
        }
        if(Simple_heuristic){
            array_indexes_type new_size = (end-start)/2;
            array_indexes_type left_end = start + new_size;
            array_indexes_type right_start = left_end + 1;
            left = new kdtree_node2(kernels, start, left_end, (split_dim+1)%3);
            right = new kdtree_node2(kernels, right_start, end, (split_dim+1)%3);
        }
        else{
            point3d midle;
            float_double max_dim[3];
            float_double min_dim[3];
            for(dimension_type i=0;i<3;i++){
                max_dim[i] = std::numeric_limits<float_double>::min();
                min_dim[i] = std::numeric_limits<float_double>::max();
            }
            
            for(array_indexes_type j=start;j<=end;j++){
                for(dimension_type i=0;i<3;i++){
                    max_dim[i] = std::max(max_dim[i], (*kernels)[j].mu[i]);
                    min_dim[i] = std::min(min_dim[i], (*kernels)[j].mu[i]);
                }
            }
            /*for(dimension_type i=0;i<3;i++){
                midle.x[i] = (max_dim[i]+min_dim[i])/2;
            }*/
            midle.x = (max_dim[0]+min_dim[0])/2;
            midle.y = (max_dim[1]+min_dim[1])/2;
            midle.z = (max_dim[2]+min_dim[2])/2;
            array_indexes_type split = 0;
            array_indexes_type st = 0;
            array_indexes_type en = end-start;
            if(Simple_heuristic2){
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = std::numeric_limits<float_double>::max();
                    float_double distance_right = std::numeric_limits<float_double>::max();
                    for(array_indexes_type i=0;i<split;i++){
                        distance_left = std::min(distance_left, (*kernels)[i].distance(midle));
                    }
                    for(array_indexes_type i=split;i<end-start;i++){
                        distance_right = std::min(distance_right, (*kernels)[i].distance(midle));
                    }
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            else{
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = max_gaussian_kernel_list(&kernels_, 0, split).distance(midle);
                    float_double distance_right = max_gaussian_kernel_list(&kernels_, split+1, kernels_.size()-1).distance(midle);
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            split = std::max(split, (array_indexes_type)1);
            split = std::min(split, end-start);
            left = new kdtree_node2(kernels, start, split+start-1, (split_dim+1)%3);
            right = new kdtree_node2(kernels, split+start, end, (split_dim+1)%3);
        }
        /*if(Beter_majoration){
            kernel_max = max_gaussian_kernel_list(&kernels_, 0, kernels_.size()-1);
        }
        else{
            kernel_max = max_gaussian_kernel(left->kernel_max, right->kernel_max);
        }*/
    }
    void search_1nn(point3d x, gaussian_kernel2_3D* res,float_double* min_dist, std::vector<gaussian_kernel2_3D>* kernels, size_t* debug_counter = nullptr){
        if (debug_counter != nullptr){
                (*debug_counter)++;
        }
        if (left == nullptr){
            array_indexes_type start = (array_indexes_type)((size_t)right);
            if ((*kernels)[start].distance(x) < *min_dist){
                *res = (*kernels)[start];
                *min_dist = (*kernels)[start].distance(x);
            }
            //test to convert back the right pointer into an index
            //std::cout << (array_indexes_type)right << " " << end << std::endl;
            return;
        }
        float_double dist_left = left->distance(x, kernels);
        float_double dist_right = right->distance(x, kernels);
        kdtree_node2 *first;
        kdtree_node2 *second;
        float_double dist_first;
        float_double dist_second;
        /*if (dist_left < dist_right){
            if (dist_left < *min_dist){
                left->search_1nn(x, res, min_dist,kernels, debug_counter);
                if (dist_right < *min_dist){
                    right->search_1nn(x, res, min_dist, kernels, debug_counter);
                }
            }
        }else{
            if (dist_right < *min_dist){
                right->search_1nn(x, res, min_dist, kernels, debug_counter);
                if (dist_left < *min_dist){
                    left->search_1nn(x, res, min_dist, kernels, debug_counter);
                }
            }
        }*/

        if (dist_left < dist_right){
            first = left;
            second = right;
            dist_first = dist_left;
            dist_second = dist_right;
        }else{
            first = right;
            second = left;
            dist_first = dist_right;
            dist_second = dist_left;
        }
        if (dist_first < *min_dist && dist_first < -negligeable_val_when_exp){
            first->search_1nn(x, res, min_dist, kernels, debug_counter);
            if (dist_second < *min_dist && dist_second < -negligeable_val_when_exp){
                second->search_1nn(x, res, min_dist, kernels, debug_counter);
            }
        }
    }
    void search_knn(point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found, std::vector<gaussian_kernel2_3D>* kernels){
        if (left == nullptr){
            array_indexes_type start = (array_indexes_type)((size_t)right);
            if ((*kernels)[start].distance(x) < *min_dist){
                if (*number_found == 0){
                    found[0] = new gaussian_kernel2_3D((*kernels)[start]);
                }
                else{
                    float_double dist = (*kernels)[start].distance(x);
                    array_indexes_type i = 0;
                    array_indexes_type j = *number_found;
                    while(j > i){
                        array_indexes_type m = (i+j)/2;
                        if (found[m]->distance(x) < dist){
                            i=m+1;
                        }
                        else{
                            j=m;
                        }
                    }
                    if(*number_found == k){
                        free(found[k-1]);
                    }
                    for(array_indexes_type l=*number_found;l>i;l--){
                        found[l] = found[l-1];
                    }
                    found[i] = new gaussian_kernel2_3D((*kernels)[start]);
                }
                *number_found = std::min(k, *number_found+1);
                if(*number_found == k){
                    *min_dist = found[k-1]->distance(x);
                }
            }
            return;
        }
        float_double dist_left = left->distance(x, kernels);
        float_double dist_right = right->distance(x, kernels);
        if (dist_left < dist_right){
            if (dist_left < *min_dist){
                left->search_knn(x, min_dist, found, k, number_found, kernels);
                if (dist_right < *min_dist){
                    right->search_knn(x, min_dist, found, k, number_found, kernels);
                }
            }
        }else{
            if (dist_right < *min_dist){
                right->search_knn(x, min_dist, found, k, number_found, kernels);
                if (dist_left < *min_dist){
                    left->search_knn(x, min_dist, found, k, number_found, kernels);
                }
            }
        }
    }
    float_double distance(point3d x, std::vector<gaussian_kernel2_3D>* kernels){
        if (left == nullptr){
            array_indexes_type start = (array_indexes_type)((size_t)right);
            return (*kernels)[start].distance(x);
        }
        /*float_double projection_x[3];
        for (int i=0;i<3;i++){
            projection_x[i] = x.x[i];
            if (x.x[i] < range_point_min[i]){
                projection_x[i] = range_point_min[i];
            }
            if (x.x[i] > range_point_max[i]){
                projection_x[i] = range_point_max[i];
            }
        }
        point3d projection(projection_x);*/
        point3d projection;
        projection.x = x.x;
        projection.y = x.y;
        projection.z = x.z;
        if(projection.x < range_point_min[0]){
            projection.x = range_point_min[0];
        }
        if(projection.x > range_point_max[0]){
            projection.x = range_point_max[0];
        }
        if(projection.y < range_point_min[1]){
            projection.y = range_point_min[1];
        }
        if(projection.y > range_point_max[1]){
            projection.y = range_point_max[1];
        }
        if(projection.z < range_point_min[2]){
            projection.z = range_point_min[2];
        }
        if(projection.z > range_point_max[2]){
            projection.z = range_point_max[2];
        }
        float_double dist = x.square_distance(projection);
        float_double d1_ = (range_log_weight_max - dist/(2*range_scale_max*range_scale_max));
        float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
        /*float_double d2 = kernel_max.distance(x);
        std::cout << (d1 >= d2)<<"   " << (d1 - d2) << std::endl;
        return std::max(d1, d2);*/
        return d1;
    }
    friend std::ostream& operator<<(std::ostream& os, const kdtree_node2 &k){
        /*if (k.start == k.end){
            os << "kdtree_node(" << k.start << "," << k.end << "," << k.split_dim << "," << k.kernel_max << ")";
            return os;
        }
        os << "kdtree_node(" << k.start << "," << k.end << "," << k.split_dim << ",";
        for(dimension_type i=0;i<3;i++){
            os << "[" << k.range_point_min[i] << "," << k.range_point_max[i] << "],";
        }
        os<< k.kernel_max << *(k.left) << *(k.right) << ")";*/
        os << "kdtree_node(" << "," << ",";
        for(dimension_type i=0;i<3;i++){
            os << "[" << k.range_point_min[i] << "," << k.range_point_max[i] << "],";
        }
        os<< k.range_log_weight_max << "," << k.range_scale_max << ",";
        if (k.left != nullptr){
            os << *(k.left);
        }
        os << ",";
        if (k.right != nullptr){
            os << *(k.right);
        }
        os << ")";
        return os;
    }
};

/*

    kdtree_node2* left;
    kdtree_node2* right;
    float_double range_point_min[3];
    float_double range_point_max[3];
    float_double range_log_weight_max;
    float_double range_scale_max;

*/



class kdtree_node3{
    public:
    array_indexes_type size;
    char* data;
    //float_double* ranges;//(of size 3+3+1+1)
    float_double range0;
    float_double range1;
    float_double range2;
    float_double range3;
    float_double range4;
    float_double range5;
    float_double range6;
    float_double range7;

    kdtree_node3(size_t size, char* data, float_double ranges0, float_double ranges1, float_double ranges2, float_double ranges3, float_double ranges4, float_double ranges5, float_double ranges6, float_double ranges7){
        this->size = size;
        this->data = data;
        this->range0 = ranges0;
        this->range1 = ranges1;
        this->range2 = ranges2;
        this->range3 = ranges3;
        this->range4 = ranges4;
        this->range5 = ranges5;
        this->range6 = ranges6;
        this->range7 = ranges7;
    }


    //first values are left index and it's ranges

    kdtree_node3* left(){
        const unsigned char size_index = sizeof(array_indexes_type)/sizeof(unsigned char);
        array_indexes_type left_index = ((array_indexes_type*)data)[0];
        const unsigned char range_size = (3+3+1+1)*sizeof(float_double)/sizeof(unsigned char);
        const unsigned char ofset = size_index + range_size;
        array_indexes_type right_index = ((array_indexes_type*)(data+ofset))[0];
        float_double* ranges = (float_double*)(data+size_index);
        return new kdtree_node3(right_index-left_index,data+left_index, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7]);
    }

    //next values are right index and it's ranges

    kdtree_node3* right(){
        const unsigned char size_index = sizeof(array_indexes_type)/sizeof(unsigned char);
        const unsigned char range_size = (3+3+1+1)*sizeof(float_double)/sizeof(unsigned char);
        const unsigned char ofset = size_index + range_size;
        array_indexes_type right_index = ((array_indexes_type*)(data+ofset))[0];
        float_double* ranges = (float_double*)(data+ofset+size_index);
        return new kdtree_node3(size-right_index,data+right_index, ranges[0], ranges[1], ranges[2], ranges[3], ranges[4], ranges[5], ranges[6], ranges[7]);
    }
    kdtree_node3(){
        size = 0;
        data = nullptr;
        range0 = 0;
        range1 = 0;
        range2 = 0;
        range3 = 0;
        range4 = 0;
        range5 = 0;
        range6 = 0;
        range7 = 0;
    
    }
    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type start, array_indexes_type end, dimension_type split_dim){
        //std::cout << start << " " << end << std::endl;
        if(start<end){
            build(ks, start, end, split_dim);
        }
        else{
            //std::cout << "a" << std::endl;
            array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char);
            //char* data = (char*)malloc(size);
            char* data = new char[size];
            //std::cout << "b" << std::endl;
            ((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = start;
            //std::cout << "c" << std::endl;
            //float_double* ranges = (float_double*)malloc((3+3+1+1)*sizeof(float_double));
            //std::cout << "d" << std::endl;
            gaussian_kernel2_1D kernel_max = (*ks)[start].convert();
            //std::cout << "e" << std::endl;
            /*for(dimension_type i=0;i<3;i++){
                ranges[i] = kernel_max.mu.x[i];
                ranges[i+3] = kernel_max.mu.x[i];
            }
            ranges[6] = kernel_max.log_weight;
            ranges[7] = kernel_max.scale;*/
            range0 = kernel_max.mu.x;
            range1 = kernel_max.mu.y;
            range2 = kernel_max.mu.z;
            range3 = kernel_max.mu.x;
            range4 = kernel_max.mu.y;
            range5 = kernel_max.mu.z;
            range6 = kernel_max.log_weight;
            range7 = kernel_max.scale;
            this->size = size;
            this->data = data;
        }            
    }

    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks){
        if(ks->size()>1){
            build(ks, 0, ks->size()-1, 0);
        }
        else{
            array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char);
            //char* data = (char*)malloc(size);
            char* data = new char[size];
            ((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = 0;
            //float_double* ranges = (float_double*)malloc((3+3+1+1)*sizeof(float_double));
            gaussian_kernel2_1D kernel_max = (*ks)[0].convert();
            /*for(dimension_type i=0;i<3;i++){
                ranges[i] = kernel_max.mu.x[i];
                ranges[i+3] = kernel_max.mu.x[i];
            }
            ranges[6] = kernel_max.log_weight;
            ranges[7] = kernel_max.scale;*/
            range0 = kernel_max.mu.x;
            range1 = kernel_max.mu.y;
            range2 = kernel_max.mu.z;
            range3 = kernel_max.mu.x;
            range4 = kernel_max.mu.y;
            range5 = kernel_max.mu.z;
            range6 = kernel_max.log_weight;
            range7 = kernel_max.scale;
            this->size = size;
            this->data = data;
        }
    }

    void build(std::vector<gaussian_kernel2_3D>* kernels, array_indexes_type start, array_indexes_type end, dimension_type split_dim){
        std::sort(kernels->begin()+start, kernels->begin()+end+1, [&split_dim](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
            return g1.mu[split_dim] < g2.mu[split_dim];
        });
        std::vector<gaussian_kernel2_1D> kernels_;
        kdtree_node3 left;
        kdtree_node3 right;
        if(Beter_majoration or (!Simple_heuristic && !Simple_heuristic2)){
            for (array_indexes_type i=start;i<=end;i++){
                kernels_.push_back((*kernels)[i].convert());
            }
        }
        if(Simple_heuristic){
            array_indexes_type new_size = (end-start)/2;
            array_indexes_type left_end = start + new_size;
            array_indexes_type right_start = left_end + 1;
            left = kdtree_node3(kernels, start, left_end, (split_dim+1)%3);
            right = kdtree_node3(kernels, right_start, end, (split_dim+1)%3);
        }
        else{
            point3d midle;
            float_double max_dim[3];
            float_double min_dim[3];
            for(dimension_type i=0;i<3;i++){
                max_dim[i] = std::numeric_limits<float_double>::min();
                min_dim[i] = std::numeric_limits<float_double>::max();
            }
            
            for(array_indexes_type j=start;j<=end;j++){
                for(dimension_type i=0;i<3;i++){
                    max_dim[i] = std::max(max_dim[i], (*kernels)[j].mu[i]);
                    min_dim[i] = std::min(min_dim[i], (*kernels)[j].mu[i]);
                }
            }
            /*for(dimension_type i=0;i<3;i++){
                midle.x[i] = (max_dim[i]+min_dim[i])/2;
            }*/
            midle.x = (max_dim[0]+min_dim[0])/2;
            midle.y = (max_dim[1]+min_dim[1])/2;
            midle.z = (max_dim[2]+min_dim[2])/2;
            array_indexes_type split = 0;
            array_indexes_type st = 0;
            array_indexes_type en = end-start;
            if(Simple_heuristic2){
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = std::numeric_limits<float_double>::max();
                    float_double distance_right = std::numeric_limits<float_double>::max();
                    for(array_indexes_type i=0;i<split;i++){
                        distance_left = std::min(distance_left, (*kernels)[i].distance(midle));
                    }
                    for(array_indexes_type i=split;i<end-start;i++){
                        distance_right = std::min(distance_right, (*kernels)[i].distance(midle));
                    }
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            else{
                while(st < en){
                    split = (st+en)/2;
                    float_double distance_left = max_gaussian_kernel_list(&kernels_, 0, split).distance(midle);
                    float_double distance_right = max_gaussian_kernel_list(&kernels_, split+1, kernels_.size()-1).distance(midle);
                    if (distance_left < distance_right){
                        en = split;
                    }
                    else{
                        if (distance_left > distance_right){
                            st = split+1;
                        }
                        else{
                            break;
                        }
                    }
                }
            }
            split = std::max(split, (array_indexes_type)1);
            split = std::min(split, end-start);
            left = kdtree_node3(kernels, start, split+start-1, (split_dim+1)%3);
            right = kdtree_node3(kernels, split+start, end, (split_dim+1)%3);
        }
        //std::cout << "a  " << start << " " << end << std::endl;
        array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char)+2*(3+3+1+1)*sizeof(float_double)/sizeof(char) + left.size + right.size;
        //std::cout << size << std::endl;
        //char* data = (char*)malloc(size);
        char* data = new char[size];
        //std::cout << "a" << std::endl;
        const unsigned char size_local = 2*sizeof(array_indexes_type)/sizeof(unsigned char) + 2*(3+3+1+1)*sizeof(float_double)/sizeof(unsigned char);
        ((array_indexes_type*)data)[0] = size_local;
        //std::cout << "b" << std::endl;
        const unsigned char ofset1 = sizeof(array_indexes_type)/sizeof(char);
        //std::cout << (array_indexes_type)ofset1 << std::endl;
        /*for(dimension_type i=0;i<(3+3+1+1);i++){
            //std::cout << (size_t)i << std::endl;
            //std::cout << ofset1+i*sizeof(float_double)/sizeof(char) <<" "<< ofset1+(i+1)*sizeof(float_double)/sizeof(char) << std::endl;
            //std::cout << left.ranges[i] << std::endl;
            ((float_double*)(data+ofset1))[i] = left.ranges[i];
        }*/
        ((float_double*)(data+ofset1))[0] = left.range0;
        ((float_double*)(data+ofset1))[1] = left.range1;
        ((float_double*)(data+ofset1))[2] = left.range2;
        ((float_double*)(data+ofset1))[3] = left.range3;
        ((float_double*)(data+ofset1))[4] = left.range4;
        ((float_double*)(data+ofset1))[5] = left.range5;
        ((float_double*)(data+ofset1))[6] = left.range6;
        ((float_double*)(data+ofset1))[7] = left.range7;
        //std::cout << "c" << std::endl;
        const unsigned char ofset2 = ofset1 +(3+3+1+1)*sizeof(float_double)/sizeof(char);
        ((array_indexes_type*)(data+ofset2))[0] = size_local+left.size;
        //std::cout << "d" << std::endl;
        const unsigned char ofset3 = ofset2 + sizeof(array_indexes_type)/sizeof(char);
        /*for (dimension_type i=0;i<(3+3+1+1);i++){
            ((float_double*)(data+ofset3))[i] = right.ranges[i];
        }*/
        ((float_double*)(data+ofset3))[0] = right.range0;
        ((float_double*)(data+ofset3))[1] = right.range1;
        ((float_double*)(data+ofset3))[2] = right.range2;
        ((float_double*)(data+ofset3))[3] = right.range3;
        ((float_double*)(data+ofset3))[4] = right.range4;
        ((float_double*)(data+ofset3))[5] = right.range5;
        ((float_double*)(data+ofset3))[6] = right.range6;
        ((float_double*)(data+ofset3))[7] = right.range7;
        //std::cout << "e" << std::endl;
        for(array_indexes_type i=0;i<left.size;i++){
            data[size_local+i] = left.data[i];
        }
        //std::cout << "f" << std::endl;
        const array_indexes_type ofset4 = size_local + left.size;
        for(array_indexes_type i=0;i<right.size;i++){
            data[ofset4+i] = right.data[i];
        }
        //std::cout << "g" << std::endl;
        this->size = size;
        this->data = data;
        //float_double* ranges = (float_double*)malloc((3+3+1+1)*sizeof(float_double));
        /*for(dimension_type i=0;i<3;i++){
            ranges[i] = std::min(left.ranges[i], right.ranges[i]);
            ranges[i+3] = std::max(left.ranges[i+3], right.ranges[i+3]);
        }
        ranges[6] = std::max(left.ranges[6], right.ranges[6]);
        ranges[7] = std::max(left.ranges[7], right.ranges[7]);*/
        range0 = std::min(left.range0, right.range0);
        range1 = std::min(left.range1, right.range1);
        range2 = std::min(left.range2, right.range2);
        range3 = std::max(left.range3, right.range3);
        range4 = std::max(left.range4, right.range4);
        range5 = std::max(left.range5, right.range5);
        range6 = std::max(left.range6, right.range6);
        range7 = std::max(left.range7, right.range7);
        if(left.data != nullptr){
            free(left.data);
            left.data = nullptr;
        }
        if(right.data != nullptr){
            free(right.data);
            right.data = nullptr;
        }
    }

    void search_1nn(point3d x, gaussian_kernel2_3D* res,float_double* min_dist, std::vector<gaussian_kernel2_3D>* kernels, size_t* debug_counter = nullptr){
        if (debug_counter != nullptr){
                (*debug_counter)++;
        }
        if (((array_indexes_type*)data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)data)[1];
            if ((*kernels)[start].distance(x) < *min_dist){
                *res = (*kernels)[start];
                *min_dist = (*kernels)[start].distance(x);
            }
            return;
        }
        kdtree_node3* left = this->left();
        kdtree_node3* right = this->right();
        float_double dist_left = left->distance(x, kernels);
        float_double dist_right = right->distance(x, kernels);
        kdtree_node3 *first;
        kdtree_node3 *second;
        float_double dist_first;
        float_double dist_second;
        if (dist_left < dist_right){
            first = left;
            second = right;
            dist_first = dist_left;
            dist_second = dist_right;
        }else{
            first = right;
            second = left;
            dist_first = dist_right;
            dist_second = dist_left;
        }
        if (dist_first < *min_dist && dist_first < -negligeable_val_when_exp){
            first->search_1nn(x, res, min_dist, kernels, debug_counter);
            if (dist_second < *min_dist && dist_second < -negligeable_val_when_exp){
                second->search_1nn(x, res, min_dist, kernels, debug_counter);
            }
        }
        free(left);
        free(right);
    }

    void search_knn(point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found, std::vector<gaussian_kernel2_3D>* kernels){
        if (((array_indexes_type*)data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)data)[1];
            if ((*kernels)[start].distance(x) < *min_dist){
                if (*number_found == 0){
                    found[0] = new gaussian_kernel2_3D((*kernels)[start]);
                }
                else{
                    float_double dist = (*kernels)[start].distance(x);
                    array_indexes_type i = 0;
                    array_indexes_type j = *number_found;
                    while(j > i){
                        array_indexes_type m = (i+j)/2;
                        if (found[m]->distance(x) < dist){
                            i=m+1;
                        }
                        else{
                            j=m;
                        }
                    }
                    if(*number_found == k){
                        free(found[k-1]);
                    }
                    for(array_indexes_type l=*number_found;l>i;l--){
                        found[l] = found[l-1];
                    }
                    found[i] = new gaussian_kernel2_3D((*kernels)[start]);
                }
                *number_found = std::min(k, *number_found+1);
                if(*number_found == k){
                    *min_dist = found[k-1]->distance(x);
                }
            }
            return;
        }

        kdtree_node3* left = this->left();
        kdtree_node3* right = this->right();
        float_double dist_left = left->distance(x, kernels);
        float_double dist_right = right->distance(x, kernels);
        kdtree_node3 *first;
        kdtree_node3 *second;
        float_double dist_first;
        float_double dist_second;
        /*if (dist_left < dist_right){
            if (dist_left < *min_dist){
                this->left()->search_knn(x, min_dist, found, k, number_found, kernels);
                if (dist_right < *min_dist){
                    this->right()->search_knn(x, min_dist, found, k, number_found, kernels);
                }
            }
        }else{
            if (dist_right < *min_dist){
                this->right()->search_knn(x, min_dist, found, k, number_found, kernels);
                if (dist_left < *min_dist){
                    this->left()->search_knn(x, min_dist, found, k, number_found, kernels);
                }
            }
        }*/

        if (dist_left < dist_right){
            first = left;
            second = right;
            dist_first = dist_left;
            dist_second = dist_right;
        }else{
            first = right;
            second = left;
            dist_first = dist_right;
            dist_second = dist_left;
        }
        if (dist_first < *min_dist && dist_first < -negligeable_val_when_exp){
            first->search_knn(x, min_dist, found, k, number_found, kernels);
            if (dist_second < *min_dist && dist_second < -negligeable_val_when_exp){
                second->search_knn(x, min_dist, found, k, number_found, kernels);
            }
        }
        free(left);
        free(right);
    }

    float_double distance(point3d x, std::vector<gaussian_kernel2_3D>* kernels){
        if (((array_indexes_type*)data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)data)[1];
            return (*kernels)[start].distance(x);
        }
        //float_double projection_x[3];
        /*for (int i=0;i<3;i++){
            projection_x[i] = x.x[i];
            if (x.x[i] < ranges[i]){
                projection_x[i] = ranges[i];
            }
            if (x.x[i] > ranges[i+3]){
                projection_x[i] = ranges[i+3];
            }
        }*/
        /*projection_x[0] = x.x[0];
        projection_x[1] = x.x[1];
        projection_x[2] = x.x[2];
        if (x.x[0] < range0){
            projection_x[0] = range0;
        }
        else if (x.x[0] > range3){
            projection_x[0] = range3;
        }
        if (x.x[1] < range1){
            projection_x[1] = range1;
        }
        else if (x.x[1] > range4){
            projection_x[1] = range4;
        }
        if (x.x[2] < range2){
            projection_x[2] = range2;
        }
        else if (x.x[2] > range5){
            projection_x[2] = range5;
        }*/
        point3d projection;
        projection.x = x.x;
        projection.y = x.y;
        projection.z = x.z;
        if (x.x < range0){
            projection.x = range0;
        }
        else if (x.x > range3){
            projection.x = range3;
        }
        if (x.y < range1){
            projection.y = range1;
        }
        else if (x.y > range4){
            projection.y = range4;
        }
        if (x.z < range2){
            projection.z = range2;
        }
        else if (x.z > range5){
            projection.z = range5;
        }
        float_double dist = x.square_distance(projection);
        //float_double d1_ = (ranges[6] - dist/(2*ranges[7]*ranges[7]));
        float_double d1_ = (range6 - dist/(2*range7*range7));
        float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
        return d1;
    }

    friend std::ostream& operator<<(std::ostream& os, const kdtree_node3 &k){
        if(((array_indexes_type*)k.data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)k.data)[1];
            os << "kdtree_node(" << start << "," << start << ",";
            /*for(dimension_type i=0;i<3;i++){
                os << "[" << k.ranges[i] << "," << k.ranges[i+3] << "],";
            }
            os<< k.ranges[6] << "," << k.ranges[7] << ")";*/
            os << "[" << k.range0 << "," << k.range3 << "],";
            os << "[" << k.range1 << "," << k.range4 << "],";
            os << "[" << k.range2 << "," << k.range5 << "],";
            os << k.range6 << "," << k.range7 << ")";
            return os;
        }
        os << "kdtree_node(";
        /*for(dimension_type i=0;i<3;i++){
            os << "[" << k.ranges[i] << "," << k.ranges[i+3] << "],";
        }
        os<< k.ranges[6] << "," << k.ranges[7] << ",";*/
        os << "[" << k.range0 << "," << k.range3 << "],";
        os << "[" << k.range1 << "," << k.range4 << "],";
        os << "[" << k.range2 << "," << k.range5 << "],";
        os << k.range6 << "," << k.range7 << ",";
        kdtree_node3* left = const_cast<kdtree_node3*>(&k)->left();
        kdtree_node3* right = const_cast<kdtree_node3*>(&k)->right();
        os << *left << "," << *right << ")";
        return os;
    }
};
float_double distance_arr(char* data, float_double range0, float_double range1, float_double range2, float_double range3, float_double range4, float_double range5, float_double range6, float_double range7, point3d x, std::vector<gaussian_kernel2_3D>* kernels){
    if (((array_indexes_type*)data)[0] == 0){
        array_indexes_type start = ((array_indexes_type*)data)[1];
        return (*kernels)[start].distance(x);
    }
    /*float_double projection_x[3];
    projection_x[0] = x.x[0];
    projection_x[1] = x.x[1];
    projection_x[2] = x.x[2];
    if (x.x[0] < range0){
        projection_x[0] = range0;
    }
    else if (x.x[0] > range3){
        projection_x[0] = range3;
    }
    if (x.x[1] < range1){
        projection_x[1] = range1;
    }
    else if (x.x[1] > range4){
        projection_x[1] = range4;
    }
    if (x.x[2] < range2){
        projection_x[2] = range2;
    }
    else if (x.x[2] > range5){
        projection_x[2] = range5;
    }
    point<3> projection(projection_x);*/
    point3d projection;
    projection.x = x.x;
    projection.y = x.y;
    projection.z = x.z;
    if (x.x < range0){
        projection.x = range0;
    }
    else if (x.x > range3){
        projection.x = range3;
    }
    if (x.y < range1){
        projection.y = range1;
    }
    else if (x.y > range4){
        projection.y = range4;
    }
    if (x.z < range2){
        projection.z = range2;
    }
    else if (x.z > range5){
        projection.z = range5;
    }

    float_double dist = x.square_distance(projection);
    float_double d1_ = (range6 - dist/(2*range7*range7));
    float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
    return d1;
}

float_double distance_arr_2(char* data, char* rangedata, point3d x, std::vector<gaussian_kernel2_3D>* kernels){
    if (((array_indexes_type*)data)[0] == 0){
        array_indexes_type start = ((array_indexes_type*)data)[1];
        return (*kernels)[start].distance(x);
    }
    point3d projection;
    projection.x = x.x;
    projection.y = x.y;
    projection.z = x.z;
    float_double range0 = ((float_double*)rangedata)[0];
    if (x.x < range0){
        projection.x = range0;
    }
    else{float_double range3 = ((float_double*)rangedata)[3];
        if (x.x > range3){
            projection.x = range3;
        }
    }
    float_double range1 = ((float_double*)rangedata)[1];
    if (x.y < range1){
        projection.y = range1;
    }
    
    else{float_double range4 = ((float_double*)rangedata)[4];
        if (x.y > range4){
            projection.y = range4;
        }
    }
    float_double range2 = ((float_double*)rangedata)[2];
    if (x.z < range2){
        projection.z = range2;
    }
    else{float_double range5 = ((float_double*)rangedata)[5];
        if (x.z > range5){
            projection.z = range5;
        }
    }
    float_double dist = x.square_distance(projection);
    float_double range6 = ((float_double*)rangedata)[6];
    float_double range7 = ((float_double*)rangedata)[7];
    float_double d1_ = (range6 - dist/(2*range7*range7));
    float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
    return d1;
}



void search_1nn_arr(char*data, point3d x, gaussian_kernel2_3D* res,float_double* min_dist, std::vector<gaussian_kernel2_3D>* kernels, size_t* debug_counter = nullptr){
    if (debug_counter != nullptr){
            (*debug_counter)++;
    }
    if (((array_indexes_type*)data)[0] == 0){
        array_indexes_type start = ((array_indexes_type*)data)[1];
        if ((*kernels)[start].distance(x) < *min_dist){
            *res = (*kernels)[start];
            *min_dist = (*kernels)[start].distance(x);
        }
        return;
    }
    char* left_data = data + ((array_indexes_type*)data)[0];
    char* right_data = data + ((array_indexes_type*)(data+sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
    /*float_double range0_left = ((float_double*)(data+sizeof(array_indexes_type)))[0];
    float_double range1_left = ((float_double*)(data+sizeof(array_indexes_type)))[1];
    float_double range2_left = ((float_double*)(data+sizeof(array_indexes_type)))[2];
    float_double range3_left = ((float_double*)(data+sizeof(array_indexes_type)))[3];
    float_double range4_left = ((float_double*)(data+sizeof(array_indexes_type)))[4];
    float_double range5_left = ((float_double*)(data+sizeof(array_indexes_type)))[5];
    float_double range6_left = ((float_double*)(data+sizeof(array_indexes_type)))[6];
    float_double range7_left = ((float_double*)(data+sizeof(array_indexes_type)))[7];
    float_double range0_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
    float_double range1_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[1];
    float_double range2_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[2];
    float_double range3_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[3];
    float_double range4_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[4];
    float_double range5_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[5];
    float_double range6_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[6];
    float_double range7_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[7];
    float_double dist_left = distance_arr(left_data,range0_left, range1_left, range2_left, range3_left, range4_left, range5_left, range6_left, range7_left, x, kernels);
    float_double dist_right = distance_arr(right_data,range0_right, range1_right, range2_right, range3_right, range4_right, range5_right, range6_right, range7_right, x, kernels);
    */
    /*float_double dist_left = distance_arr(
                                            left_data,
                                            ((float_double*)(data+sizeof(array_indexes_type)))[0],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[1],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[2],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[3],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[4],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[5],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[6],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[7],
                                            x, kernels
                                        );
    float_double dist_right = distance_arr(
                                            right_data,
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[1],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[2],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[3],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[4],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[5],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[6],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[7],
                                            x, kernels
                                        );*/
    float_double dist_left = distance_arr_2(left_data, data+sizeof(array_indexes_type), x, kernels);
    float_double dist_right = distance_arr_2(right_data, data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), x, kernels);
    char* first;
    char* second;
    float_double dist_first;
    float_double dist_second;
    if (dist_left < dist_right){
        first = left_data;
        second = right_data;
        dist_first = dist_left;
        dist_second = dist_right;
    }else{
        first = right_data;
        second = left_data;
        dist_first = dist_right;
        dist_second = dist_left;
    }
    if (dist_first < *min_dist && dist_first <-negligeable_val_when_exp){
        search_1nn_arr(first, x, res, min_dist, kernels, debug_counter);
        if (dist_second < *min_dist && dist_second <-negligeable_val_when_exp){
            search_1nn_arr(second, x, res, min_dist, kernels, debug_counter);
        }
    }
}

void search_knn_arr(char*data, point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found, std::vector<gaussian_kernel2_3D>* kernels){
    if (((array_indexes_type*)data)[0] == 0){
        array_indexes_type start = ((array_indexes_type*)data)[1];
        if ((*kernels)[start].distance(x) < *min_dist){
            if (*number_found == 0){
                found[0] = new gaussian_kernel2_3D((*kernels)[start]);
            }
            else{
                float_double dist = (*kernels)[start].distance(x);
                array_indexes_type i = 0;
                array_indexes_type j = *number_found;
                while(j > i){
                    array_indexes_type m = (i+j)/2;
                    if (found[m]->distance(x) < dist){
                        i=m+1;
                    }
                    else{
                        j=m;
                    }
                }
                if(*number_found == k){
                    free(found[k-1]);
                }
                for(array_indexes_type l=*number_found;l>i;l--){
                    found[l] = found[l-1];
                }
                found[i] = new gaussian_kernel2_3D((*kernels)[start]);
            }
            *number_found = std::min(k, *number_found+1);
            if(*number_found == k){
                *min_dist = found[k-1]->distance(x);
            }
        }
        return;
    }
    char* left_data = data + ((array_indexes_type*)data)[0];
    char* right_data = data + ((array_indexes_type*)(data+sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
    /*float_double range0_left = ((float_double*)(data+sizeof(array_indexes_type)))[0];
    float_double range1_left = ((float_double*)(data+sizeof(array_indexes_type)))[1];
    float_double range2_left = ((float_double*)(data+sizeof(array_indexes_type)))[2];
    float_double range3_left = ((float_double*)(data+sizeof(array_indexes_type)))[3];
    float_double range4_left = ((float_double*)(data+sizeof(array_indexes_type)))[4];
    float_double range5_left = ((float_double*)(data+sizeof(array_indexes_type)))[5];
    float_double range6_left = ((float_double*)(data+sizeof(array_indexes_type)))[6];
    float_double range7_left = ((float_double*)(data+sizeof(array_indexes_type)))[7];
    float_double dist_left = distance_arr(left_data,range0_left, range1_left, range2_left, range3_left, range4_left, range5_left, range6_left, range7_left, x, kernels);
    float_double range0_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
    float_double range1_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[1];
    float_double range2_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[2];
    float_double range3_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[3];
    float_double range4_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[4];
    float_double range5_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[5];
    float_double range6_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[6];
    float_double range7_right = ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[7];
    float_double dist_right = distance_arr(right_data,range0_right, range1_right, range2_right, range3_right, range4_right, range5_right, range6_right, range7_right, x, kernels);
    */
    float_double dist_left = distance_arr(
                                            left_data,
                                            ((float_double*)(data+sizeof(array_indexes_type)))[0],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[1],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[2],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[3],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[4],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[5],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[6],
                                            ((float_double*)(data+sizeof(array_indexes_type)))[7],
                                            x,
                                            kernels
                                        );
    float_double dist_right = distance_arr(
                                            right_data,
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[1],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[2],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[3],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[4],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[5],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[6],
                                            ((float_double*)(data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[7],
                                            x,
                                            kernels
                                        );
    char* first;
    char* second;
    float_double dist_first;
    float_double dist_second;
    if (dist_left < dist_right){
        first = left_data;
        second = right_data;
        dist_first = dist_left;
        dist_second = dist_right;
    }else{
        first = right_data;
        second = left_data;
        dist_first = dist_right;
        dist_second = dist_left;
    }
    if (dist_first < *min_dist && dist_first <-negligeable_val_when_exp){
        search_knn_arr(first, x, min_dist, found, k, number_found, kernels);
        if (dist_second < *min_dist && dist_second <-negligeable_val_when_exp){
            search_knn_arr(second, x, min_dist, found, k, number_found, kernels);
        }
    }
}



template <dimension_type N_dim,dimension_type N_dim_point>
class kd_tree{
    public:
    kdtree_node<N_dim,N_dim_point> * root;
    kd_tree(){
        root = nullptr;
    }
    void search_1nn(gaussian_kernel<N_dim,N_dim_point>* res,point<N_dim_point> x,float_double* min_dist, size_t* debug_counter = nullptr){
        root->search_1nn(x, res, min_dist, debug_counter);
    }
    void search_knn(point<N_dim_point> x, float_double* min_dist,gaussian_kernel<N_dim,N_dim_point>** found, array_indexes_type k, array_indexes_type* number_found){
        root->search_knn(x, min_dist, found, k, number_found);
    }


    friend std::ostream& operator<<(std::ostream& os, const kd_tree<N_dim,N_dim_point>& k){
        os << "kd_tree(" << *k.root << ")";
        return os;
    }
};

class kd_tree2{
    public:
    kdtree_node2 * root;
    std::vector<gaussian_kernel2_3D>* kernels;
    kd_tree2(std::vector<gaussian_kernel2_3D>* ks){
        root = nullptr;
        kernels = ks;
    }
    void search_1nn(gaussian_kernel2_3D* res,point3d x,float_double* min_dist, size_t* debug_counter = nullptr){
        root->search_1nn(x, res, min_dist, kernels ,debug_counter);
    }
    void search_knn(point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found){
        root->search_knn(x, min_dist, found, k, number_found, kernels);
    }
    friend std::ostream& operator<<(std::ostream& os, const kd_tree2& k){
        os << "kd_tree(" << *k.root << ")";
        return os;
    }
};

class kd_tree3{
    public:
    kdtree_node3 * root;
    std::vector<gaussian_kernel2_3D>* kernels;
    kd_tree3(std::vector<gaussian_kernel2_3D>* ks){
        root = new kdtree_node3(ks);
        kernels = ks;
    }
    void search_1nn(gaussian_kernel2_3D* res,point3d x,float_double* min_dist, size_t* debug_counter = nullptr){
        //root->search_1nn(x, res, min_dist, kernels ,debug_counter);
        search_1nn_arr(root->data, x, res, min_dist, kernels, debug_counter);
    }
    void search_knn(point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found){
        //root->search_knn(x, min_dist, found, k, number_found, kernels);
        search_knn_arr(root->data, x, min_dist, found, k, number_found, kernels);
    }
    friend std::ostream& operator<<(std::ostream& os, const kd_tree3& k){
        os << "kd_tree(" << *k.root << ")";
        return os;
    }
};





template<dimension_type N_dim>
void random_semipositive_definite_matrix(float_double * sigma){
    //std::cout<< "a" << std::endl;
    //generate a random matrix and check if it is semi positive definite
    for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=i;j<N_dim;j++){
            sigma[i*N_dim+j] = 2*((float_double)rand()/(float_double)RAND_MAX)-1;
            sigma[j*N_dim+i] = sigma[i*N_dim+j];
            if(i==j){
                sigma[i*N_dim+j] = abs(sigma[i*N_dim+j]);
            }
            //std::cout << sigma[i*N_dim+j] << " ";
        }
        //std::cout << std::endl;
    }
    //check if it is semi positive definite
    //Sylvester's criterion
    for(dimension_type i=0;i<N_dim;i++){
        float_double sigma_temp[(i+1)*(i+1)];
        for(dimension_type j=0;j<i+1;j++){
            for(dimension_type k=0;k<i+1;k++){
                sigma_temp[j*(i+1)+k] = sigma[j*N_dim+k];
            }
        }
        float_double det_temp = det(i+1, sigma_temp);
        //std::cout << det_temp << std::endl;
        if (det_temp <= 0){
            random_semipositive_definite_matrix<N_dim>(sigma);
            return;
        }
    }
    /*for(dimension_type i=0;i<N_dim;i++){
        for(dimension_type j=0;j<N_dim;j++){
            std::cout << sigma[i*N_dim+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << det(N_dim, sigma) << std::endl;
    std::cout << std::endl;
    */
        
}

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
    

int main(){
    /*const dimension_type N_dim = 3;
    const dimension_type N_dim_point = 3;*/
    array_indexes_type N = 1774942;
    /*std::cout << "start" << std::endl;
    std::vector<gaussian_kernel<N_dim,N_dim_point>> kernels;
    //take random kernels
    std::cout << "kernels start creating" << std::endl;
    for(array_indexes_type i=0;i<N;i++){
        point<N_dim_point> mu;
        for(dimension_type j=0;j<N_dim_point;j++){
            mu.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }
        float_double* sigma_ = new float_double[N_dim*N_dim];
        random_semipositive_definite_matrix<N_dim>(sigma_);
        float_double sigma[N_dim*N_dim];
        for (int j=0;j<N_dim_point*N_dim_point;j++){
            sigma[j] = sigma_[j]*10;
        }
        //weight between 0 and 1
        float_double weight = ((float_double)rand()/(float_double)RAND_MAX);
        kernels.push_back(gaussian_kernel<N_dim,N_dim_point>(mu, log(weight), sigma));
    }
    std::cout << "kernels created" << std::endl;
    kd_tree<N_dim,N_dim_point> tree;
    kdtree_node<N_dim,N_dim_point> root(&kernels, 0, N-1, 0);
    std::cout << "root created" << std::endl;
    tree.root = &root;
    std::cout << "tree created" << std::endl;
    //std::cout << tree << std::endl;
    //make random points
    //save current time to measure the performance to do that : use the function time*/
    clock_t start_uniform_1, end_uniform_1, start_uniform_k, end_uniform_k, start_true_1, end_true_1, start_true_k, end_true_k;
    start_uniform_1 = clock();
    /*for(array_indexes_type i=0;i<1000000;i++){
        std::cout << i << std::endl;
        point<N_dim_point> x;
        for(dimension_type j=0;j<N_dim_point;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }
        
        //size_t debug_counter = 0;
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel<N_dim,N_dim_point>* res = new gaussian_kernel<N_dim,N_dim_point>();
        //tree.search_1nn(res, x, &min_dist_kd, &debug_counter);
        //std::cout << debug_counter << std::endl;
        tree.search_1nn(res, x, &min_dist_kd);
        //print the root gaussian kernel
        //std::cout << root.kernel_max << std::endl;
        //std::cout << *res << std::endl;
        //std::cout << "tree explored" << std::endl;*/
        /*
        //compute the distance by brute force to check
        float_double min_dist = kernels[0].distance(x);
        array_indexes_type min_dist_ind = std::numeric_limits<array_indexes_type>::max();
        for(array_indexes_type j=1;j<N;j++){
            float_double dist = kernels[j].distance(x);
            if (dist < min_dist){
                min_dist = dist;
                min_dist_ind = j;
            }
        }
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist_kd << " " << min_dist << std::endl;
        if (abs(min_dist_kd - min_dist) > 0){
            std::cout << "error" << std::endl;
            std::cout << min_dist_kd << " " << min_dist << std::endl;
            std::cout << *res << kernels[min_dist_ind] << std::endl;
            std::cout << x << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels[j] << std::endl;
                std::cout << kernels[j].distance(x) << std::endl;
            }
            break;
        }
        */
    /*}*/
    end_uniform_1 = clock();
    /*std::cout << "end 1nn" << std::endl;
    //search knn
    const array_indexes_type k = 20;*/
    start_uniform_k = clock();
    /*for(array_indexes_type i=0;i<1000000;i++){
        std::cout << i << std::endl;
        point<N_dim_point> x;
        for(dimension_type j=0;j<N_dim_point;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel<N_dim,N_dim_point>* found[k];
        array_indexes_type number_found = 0;
        tree.search_knn(x, &min_dist, found, k, &number_found);
        */
        /*
        //compute the distance by brute force to check
        //std::cout << "brute force" << std::endl;
        std::vector<float_double> min_dist_brute;
        for (array_indexes_type j=0;j<k;j++){
            min_dist_brute.push_back(kernels[j].distance(x));
        }
        //sort the 2 vectors
        std::sort(min_dist_brute.begin(), min_dist_brute.end());
        for(array_indexes_type j=k;j<N;j++){
            //put the new distance in the list
            float_double dist = kernels[j].distance(x);
            //push the new distance
            min_dist_brute.push_back(dist);
            std::sort(min_dist_brute.begin(), min_dist_brute.end());
            min_dist_brute.pop_back();
        }
        
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist << " " << min_dist_brute[k-1] << std::endl;
        //for(array_indexes_type j=0;j<k;j++){
        //    std::cout << min_dist_brute[j] << std::endl;
        //}
        bool error = false;
        for(array_indexes_type j=0;j<number_found;j++){
            if (abs(found[j]->distance(x) - min_dist_brute[j]) > 0){
                error = true;
                std::cout << "error at " << j << std::endl;
                std::cout << found[j]->distance(x) << " " << min_dist_brute[j] << std::endl;
            }
        }
        for(array_indexes_type j=number_found;j<k;j++){
            if(min_dist_brute[j] != std::min(std::numeric_limits<float_double>::max(),-negligeable_val_when_exp)){
                error = true;
                std::cout << "error at " << j << std::endl;
            }
        }
        if (error){
            std::cout << "error" << std::endl;
            std::cout << "min_dist" << " " << "min_dist_brute" << std::endl;
            std::cout << min_dist << " " << min_dist_brute[k-1] << std::endl;
            std::cout << "x"<<std::endl;
            std::cout << x << std::endl;
            std::cout << "number_found" << std::endl;
            std::cout << number_found << std::endl;
            std::cout << std::endl;
            std::cout <<  " " << "found[j]" << std::endl;
            std::cout << "min_dist_brute[j]" << " " << "found[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<k;j++){
                std::cout  << " "<<found[j] << std::endl;
                std::cout << min_dist_brute[j] <<" "<< (*found)[j].distance(x) << std::endl;
            }
            std::cout << std::endl;
            std::cout << "kernels[j]" << " " << "kernels[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels[j] << std::endl;
                std::cout << kernels[j].distance(x) << std::endl;
            }
            break;
        }
        */
        
    /*}*/
    end_uniform_k = clock();
    /*std::cout << "end knn" << std::endl;*/

    //load true gaussian values from a ply file
    //typical header:
    /*
    ply
    format binary_little_endian 1.0
    element vertex 281498
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float f_dc_0
    property float f_dc_1
    property float f_dc_2
    property float f_rest_0
    property float f_rest_1
    property float f_rest_2
    property float f_rest_3
    property float f_rest_4
    property float f_rest_5
    property float f_rest_6
    property float f_rest_7
    property float f_rest_8
    property float f_rest_9
    property float f_rest_10
    property float f_rest_11
    property float f_rest_12
    property float f_rest_13
    property float f_rest_14
    property float f_rest_15
    property float f_rest_16
    property float f_rest_17
    property float f_rest_18
    property float f_rest_19
    property float f_rest_20
    property float f_rest_21
    property float f_rest_22
    property float f_rest_23
    property float f_rest_24
    property float f_rest_25
    property float f_rest_26
    property float f_rest_27
    property float f_rest_28
    property float f_rest_29
    property float f_rest_30
    property float f_rest_31
    property float f_rest_32
    property float f_rest_33
    property float f_rest_34
    property float f_rest_35
    property float f_rest_36
    property float f_rest_37
    property float f_rest_38
    property float f_rest_39
    property float f_rest_40
    property float f_rest_41
    property float f_rest_42
    property float f_rest_43
    property float f_rest_44
    property float opacity
    property float scale_0
    property float scale_1
    property float scale_2
    property float rot_0
    property float rot_1
    property float rot_2
    property float rot_3
    end_header
    */
    /*const dimension_type N_dim_point_ = 3;
    const dimension_type N_dim_ = 3;
    std::vector<gaussian_kernel<N_dim_,N_dim_point_>> kernels_;
    

    std::ifstream infile("input.ply", std::ios_base::binary);

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	// Output number of Gaussians contained
	std::cout << "Loading " << count << " Gaussian splats" << std::endl;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians
    array_indexes_type i = 0;
    while (infile.peek() != EOF){
        float x, y, z;
        float nx, ny, nz;
        float opacity;
        float scale_0, scale_1, scale_2;
        float rot_0, rot_1, rot_2, rot_3;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&nx), sizeof(float));
        infile.read(reinterpret_cast<char*>(&ny), sizeof(float));
        infile.read(reinterpret_cast<char*>(&nz), sizeof(float));
        for(int j=0;j< 3+45;j++){
            float dummy;
            infile.read(reinterpret_cast<char*>(&dummy), sizeof(float));
        }
        infile.read(reinterpret_cast<char*>(&opacity), sizeof(float));
        infile.read(reinterpret_cast<char*>(&scale_0), sizeof(float));
        infile.read(reinterpret_cast<char*>(&scale_1), sizeof(float));
        infile.read(reinterpret_cast<char*>(&scale_2), sizeof(float));
        infile.read(reinterpret_cast<char*>(&rot_0), sizeof(float));
        infile.read(reinterpret_cast<char*>(&rot_1), sizeof(float));
        infile.read(reinterpret_cast<char*>(&rot_2), sizeof(float));
        infile.read(reinterpret_cast<char*>(&rot_3), sizeof(float));

        //normalize the quaternion
        float norm = sqrt(rot_0*rot_0+rot_1*rot_1+rot_2*rot_2+rot_3*rot_3);
        rot_0 /= norm;
        rot_1 /= norm;
        rot_2 /= norm;
        rot_3 /= norm;

        //exponentiate the scale
        scale_0 = exp(scale_0);
        scale_1 = exp(scale_1);
        scale_2 = exp(scale_2);

        //opacity pass throught sigmoid
        opacity =( 1/(1+exp(-opacity)));

        //create the rotation matrix from the quaternion
        float_double rot[3][3];
        rot[0][0] = (float_double) (1-2*rot_2*rot_2-2*rot_3*rot_3);
        rot[0][1] = (float_double) (2*rot_1*rot_2-2*rot_0*rot_3);
        rot[0][2] = (float_double) (2*rot_1*rot_3+2*rot_0*rot_2);
        rot[1][0] = (float_double) (2*rot_1*rot_2+2*rot_0*rot_3);
        rot[1][1] = (float_double) (1-2*rot_1*rot_1-2*rot_3*rot_3);
        rot[1][2] = (float_double) (2*rot_2*rot_3-2*rot_0*rot_1);
        rot[2][0] = (float_double) (2*rot_1*rot_3-2*rot_0*rot_2);
        rot[2][1] = (float_double) (2*rot_2*rot_3+2*rot_0*rot_1);
        rot[2][2] = (float_double) (1-2*rot_1*rot_1-2*rot_2*rot_2);

        //create the scale matrix
        float_double scale[3][3];
        scale[0][0] = (float_double) scale_0;
        scale[0][1] = 0;
        scale[0][2] = 0;
        scale[1][0] = 0;
        scale[1][1] = (float_double) scale_1;
        scale[1][2] = 0;
        scale[2][0] = 0;
        scale[2][1] = 0;
        scale[2][2] = (float_double) scale_2;

        //create the covariance matrix
        //sigma__ = rot * scale * scale^T * rot^T
        //sigma__ = (rot * scale) * (rot * scale)^T
        //sigma_ = 1/2 sigma__^-1

        double rot_scale[3][3];
        for(dimension_type j=0;j<3;j++){
            for(dimension_type k=0;k<3;k++){
                rot_scale[j][k] = 0;
                for(dimension_type l=0;l<3;l++){
                    rot_scale[j][k] += rot[j][l]*scale[l][k];
                }
            }
        }

        double sigma[3][3];
        for(dimension_type j=0;j<3;j++){
            for(dimension_type k=0;k<3;k++){
                sigma[j][k] = 0;
                for(dimension_type l=0;l<3;l++){
                    sigma[j][k] += rot_scale[j][l]*rot_scale[k][l];
                }
            }
        }



        double sigma__[3*3];
        for(dimension_type j=0;j<3;j++){
            for(dimension_type k=0;k<3;k++){
                sigma__[j*3+k] = sigma[j][k];
            }
        }

        //sigma is 1/2 sigma__^-1
        double sigma_d[3*3];
        invert_matrix(3, sigma__, sigma_d);
        for(dimension_type j=0;j<3;j++){
            for(dimension_type k=0;k<3;k++){
                sigma_d[j*3+k] /= 2;
            }
        }
        float_double sigma_[3*3];
        for(dimension_type j=0;j<3;j++){
            for(dimension_type k=0;k<3;k++){
                sigma_[j*3+k] = (float_double) sigma_d[j*3+k];
            }
        }

        float_double mu_[3];
        mu_[0] = (float_double)x;
        mu_[1] = (float_double)y;
        mu_[2] = (float_double)z;

        //create the kernel
        kernels_.push_back(gaussian_kernel<N_dim_,N_dim_point_>(mu_, log(opacity)+log(sqrt(pow(pi,3)/det(3,sigma_))), sigma_));
        //ensure the sigma is semi positive definite (because computation i do to obtain it can make it not semi positive definite if we use float_double = float)\
        //if not we drop the kernel*/
        /*if (det(3, sigma_) <= 0 || det(2, sigma_) <= 0 || det(1, sigma_) <= 0){
            kernels_.pop_back();
            std::cout << "kernel dropped due to numerical instability" << std::endl;
        }*/
        
        
        
        
        
        //std::cout << kernels_[i] << std::endl;
        /*i++;
    }*/
    /*std::cout << "kernels loaded" << std::endl;  
    std::cout << kernels_.size() << std::endl;
    //create the tree
    kd_tree<N_dim_,N_dim_point_> tree_;
    kdtree_node<N_dim_,N_dim_point_> root_(&kernels_);
    tree_.root = &root_;
    float_double min_coord[3] = {std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max()};
    float_double max_coord[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    for(dimension_type i=0;i<3;i++){
        min_coord[i] = root_.range_point_min[i];
        max_coord[i] = root_.range_point_max[i];
    }
    */
    //search the 1nn
    /* float_double min_dist_ever_seen = std::numeric_limits<float_double>::max();
    point<N_dim_point_> x_ever_seen;*/
    start_true_1 = clock();
    /*for(array_indexes_type i=0;i<1000000;i++){
        std::cout << i << std::endl;
        point<N_dim_point_> x;
        for(dimension_type j=0;j<N_dim_point_;j++){
            x.x[j] = min_coord[j] + (max_coord[j]-min_coord[j])*((float_double)rand()/(float_double)RAND_MAX);
        }
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel<N_dim_,N_dim_point_>* res = new gaussian_kernel<N_dim_,N_dim_point_>();
        tree_.search_1nn(res, x, &min_dist_kd);*/
        /*std::cout << min_dist_kd << std::endl;
        std::cout << *res << std::endl;
        std::cout << x << std::endl;
        std::cout << (*res).distance(x) << std::endl;
        if (min_dist_kd < min_dist_ever_seen){
            min_dist_ever_seen = min_dist_kd;
            x_ever_seen = x;
        }*/
    /*}*/
    end_true_1 = clock();
    /*std::cout << "end 1nn" << std::endl;*/
    /*std::cout << min_dist_ever_seen << std::endl;
    std::cout << x_ever_seen << std::endl;*/
    //search the knn
    /*const array_indexes_type k_ = 20;*/
    start_true_k = clock();
    /*for(array_indexes_type i=0;i<1000000;i++){
        std::cout << i << std::endl;
        point<N_dim_point_> x;
        for(dimension_type j=0;j<N_dim_point_;j++){
            x.x[j] = min_coord[j] + (max_coord[j]-min_coord[j])*((float_double)rand()/(float_double)RAND_MAX);
        }
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel<N_dim_,N_dim_point_>* found[k_];
        array_indexes_type number_found = 0;
        tree_.search_knn(x, &min_dist, found, k_, &number_found);
    }*/
    end_true_k = clock();

    /*
    //output centroids coordinates in a txt file to visualize them
    std::ofstream outfile("output.txt");
    for(array_indexes_type i=0;i<kernels_.size();i++){
        for(array_indexes_type j=0;j<N_dim_point_;j++){
            outfile << kernels_[i].mu.x[j] << " ";
        }
        outfile << std::endl;
    }
    outfile.close();

    //output the centroids and the invert of the minimum eigenvalue of the covariance matrix and the weight in a txt file
    std::ofstream outfile2("output2.txt");
    for(array_indexes_type i=0;i<kernels_.size();i++){
        outfile2 << kernels_[i].mu.x[0] << " " << kernels_[i].mu.x[1] << " " << kernels_[i].mu.x[2] << " " << 1/(min_eigan_value(3, kernels_[i].sigma)) << " " << exp(kernels_[i].log_weight-kernels_[i].log_coef_sigma) << std::endl;
    }
    outfile2.close();
    
    */

    /*infile.close();*/


    //test the kdtree2

    std::vector<gaussian_kernel2_3D> kernels2;
    for(array_indexes_type i=0;i<N;i++){
        point3d mu;
        /*for(dimension_type j=0;j<3;j++){
            mu.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        mu.x= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.y= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.z= 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double* scale_ = new float_double[3];
        random_scale(scale_);
        for(dimension_type j=0;j<3;j++){
            scale_[j] = 4*scale_[j];
        }
        float_double* q = new float_double[4];
        random_unitary_quaternion(q);
        float_double weight = ((float_double)rand()/(float_double)RAND_MAX);
        kernels2.push_back(gaussian_kernel2_3D(mu, weight , scale_, q));
        free(scale_);
        free(q);
    }
    std::cout << "kernels2 created" << std::endl;
    kd_tree2 tree2(&kernels2);
    kdtree_node2 root2(&kernels2);
    tree2.root = &root2;
    std::cout << "tree2 created" << std::endl;
    //make random points
    clock_t start_uniform_1_2, end_uniform_1_2, start_uniform_k_2, end_uniform_k_2, start_true_1_2, end_true_1_2, start_true_k_2, end_true_k_2;
    start_uniform_1_2 = clock();
    for (int i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 10*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D res;
        tree2.search_1nn(&res, x, &min_dist_kd);
        //std::cout << min_dist_kd << std::endl;
        /*
        //compute the distance by brute force to check
        float_double min_dist = kernels2[0].distance(x);
        array_indexes_type min_dist_ind = std::numeric_limits<array_indexes_type>::max();
        for(array_indexes_type j=1;j<N;j++){
            float_double dist = kernels2[j].distance(x);
            if (dist < min_dist){
                min_dist = dist;
                min_dist_ind = j;
            }
        }
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist_kd << " " << min_dist << std::endl;
        if (abs(min_dist_kd - min_dist) > 0){
            std::cout << "error" << std::endl;
            std::cout << min_dist_kd << " " << min_dist << std::endl;
            std::cout << res << kernels2[min_dist_ind] << std::endl;
            std::cout << x << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels2[j] << std::endl;
                std::cout << kernels2[j].distance(x) << std::endl;
            }
            break;
        }
        else{
            if(std::isnan(min_dist_kd) || std::isnan(min_dist)){
                std::cout << "error" << std::endl;
                std::cout << min_dist_kd << " " << min_dist << std::endl;
                std::cout << res << kernels2[min_dist_ind] << std::endl;
                std::cout << x << std::endl;
                for(array_indexes_type j=0;j<N;j++){
                    std::cout << kernels2[j] << std::endl;
                    std::cout << kernels2[j].distance(x) << std::endl;
                }
                break;
            }
        }
        */
    }
    end_uniform_1_2 = clock();
    
    //search knn
    const array_indexes_type k2 = 20;
    start_uniform_k_2 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D* found[k2];
        array_indexes_type number_found = 0;
        tree2.search_knn(x, &min_dist, found, k2, &number_found);
        
        /*
        //compute the distance by brute force to check
        //std::cout << "brute force" << std::endl;
        std::vector<float_double> min_dist_brute;
        for (array_indexes_type j=0;j<k2;j++){
            min_dist_brute.push_back(kernels2[j].distance(x));
        }
        //sort the 2 vectors
        std::sort(min_dist_brute.begin(), min_dist_brute.end());
        for(array_indexes_type j=k2;j<N;j++){
            //put the new distance in the list
            float_double dist = kernels2[j].distance(x);
            //push the new distance
            min_dist_brute.push_back(dist);
            std::sort(min_dist_brute.begin(), min_dist_brute.end());
            min_dist_brute.pop_back();
        }
        
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist << " " << min_dist_brute[k2-1] << std::endl;
        //for(array_indexes_type j=0;j<k2;j++){
        //    std::cout << min_dist_brute[j] << std::endl;
        //}
        bool error = false;
        for(array_indexes_type j=0;j<number_found;j++){
            if (abs(found[j]->distance(x) - min_dist_brute[j]) > 0){
                error = true;
                std::cout << "error at " << j << std::endl;
                std::cout << found[j]->distance(x) << " " << min_dist_brute[j] << std::endl;
            }
        }
        for(array_indexes_type j=number_found;j<k2;j++){
            if(min_dist_brute[j] != std::min(std::numeric_limits<float_double>::max(),-negligeable_val_when_exp)){
                error = true;
                std::cout << "error at " << j << std::endl;
            }
        }
        if (error){
            std::cout << "error" << std::endl;
            std::cout << "min_dist" << " " << "min_dist_brute" << std::endl;
            std::cout << min_dist << " " << min_dist_brute[k2-1] << std::endl;
            std::cout << "x"<<std::endl;
            std::cout << x << std::endl;
            std::cout << "number_found" << std::endl;
            std::cout << number_found << std::endl;
            std::cout << std::endl;
            std::cout <<  " " << "found[j]" << std::endl;
            std::cout << "min_dist_brute[j]" << " " << "found[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<k2;j++){
                std::cout  << " "<<found[j] << std::endl;
                std::cout << min_dist_brute[j] <<" "<< (*found)[j].distance(x) << std::endl;
            }
            std::cout << std::endl;
            std::cout << "kernels[j]" << " " << "kernels[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels2[j] << std::endl;
                std::cout << kernels2[j].distance(x) << std::endl;
            }
            break;
        }
        */
    }
    end_uniform_k_2 = clock();

    //load true gaussian values from a ply file
    //typical header:
    /*
    ...
    */
    std::vector<gaussian_kernel2_3D> kernels2_;
    std::ifstream infile2("input.ply", std::ios_base::binary);

    // "Parse" header (it has to be a specific format anyway)
    std::string buff2;
    std::getline(infile2, buff2);
    std::getline(infile2, buff2);

    std::string dummy2;
    std::getline(infile2, buff2);
    std::stringstream ss2(buff2);
    int count2;
    ss2 >> dummy2 >> dummy2 >> count2;

    // Output number of Gaussians contained
    std::cout << "Loading " << count2 << " Gaussian splats" << std::endl;

    while (std::getline(infile2, buff2))
        if (buff2.compare("end_header") == 0)
            break;

    // Read all Gaussians
    array_indexes_type i2 = 0;
    while (infile2.peek() != EOF){
        float x, y, z;
        float nx, ny, nz;
        float opacity;
        float scale_0, scale_1, scale_2;
        float rot_0, rot_1, rot_2, rot_3;
        infile2.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&nx), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&ny), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&nz), sizeof(float));
        for(int j=0;j< 3+45;j++){
            float dummy;
            infile2.read(reinterpret_cast<char*>(&dummy), sizeof(float));
        }
        infile2.read(reinterpret_cast<char*>(&opacity), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&scale_0), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&scale_1), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&scale_2), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&rot_0), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&rot_1), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&rot_2), sizeof(float));
        infile2.read(reinterpret_cast<char*>(&rot_3), sizeof(float));

        //normalize the quaternion
        float norm = sqrt(rot_0*rot_0+rot_1*rot_1+rot_2*rot_2+rot_3*rot_3);
        rot_0 /= norm;
        rot_1 /= norm;
        rot_2 /= norm;
        rot_3 /= norm;

        //exponentiate the scale
        scale_0 = exp(scale_0);
        scale_1 = exp(scale_1);
        scale_2 = exp(scale_2);

        //opacity pass throught sigmoid
        opacity =( 1/(1+exp(-opacity)));

        //create scale and quaternion vectors
        float_double scale[3];
        scale[0] = (float_double) scale_0;
        scale[1] = (float_double) scale_1;
        scale[2] = (float_double) scale_2;
        float_double q[4];
        q[0] = (float_double) rot_0;
        q[1] = (float_double) rot_1;
        q[2] = (float_double) rot_2;
        q[3] = (float_double) rot_3;

        /*float_double mu[3];
        mu[0] = (float_double)x;
        mu[1] = (float_double)y;
        mu[2] = (float_double)z;
        */
        point3d mu;
        mu.x = (float_double)x;
        mu.y = (float_double)y;
        mu.z = (float_double)z;

        kernels2_.push_back(gaussian_kernel2_3D(mu, opacity, scale, q));
        i2++;
    }
    std::cout << "kernels2 loaded" << std::endl;

    //create the tree
    kd_tree2 tree2_(&kernels2_);
    kdtree_node2 root2_(&kernels2_);
    tree2_.root = &root2_;
    float_double min_coord2[3] = {std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max()};
    float_double max_coord2[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    for(dimension_type i=0;i<3;i++){
        min_coord2[i] = root2_.range_point_min[i];
        max_coord2[i] = root2_.range_point_max[i];
    }
    //std::cout << min_coord2[0] << " " << min_coord2[1] << " " << min_coord2[2] << std::endl;
    //std::cout << max_coord2[0] << " " << max_coord2[1] << " " << max_coord2[2] << std::endl;

    //search the 1nn
    float_double min_dist_ever_seen2 = std::numeric_limits<float_double>::max();
    point3d x_ever_seen2;
    start_true_1_2 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord2[j] + (max_coord2[j]-min_coord2[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord2[0] + (max_coord2[0]-min_coord2[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord2[1] + (max_coord2[1]-min_coord2[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord2[2] + (max_coord2[2]-min_coord2[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D res;
        tree2_.search_1nn(&res, x, &min_dist_kd);
        /*//compare the distance with the one given by the other tree on the same point
        double min_dist = res.distance(x);
        gaussian_kernel<3,3> res_;
        tree_.search_1nn(&res_, x, &min_dist);
        std::cout << min_dist_kd << " " << min_dist << std::endl;
        */



        /*std::cout << min_dist_kd << std::endl;
        std::cout << res << std::endl;
        std::cout << x << std::endl;
        std::cout << res.distance(x) << std::endl;
        if (min_dist_kd < min_dist_ever_seen2){
            min_dist_ever_seen2 = min_dist_kd;
            x_ever_seen2 = x;
        }*/
    }
    end_true_1_2 = clock();
    std::cout << "end 1nn" << std::endl;
    /*std::cout << min_dist_ever_seen2 << std::endl;
    std::cout << x_ever_seen2 << std::endl;*/
    //search the knn
    const array_indexes_type k2_ = 20;
    start_true_k_2 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord2[j] + (max_coord2[j]-min_coord2[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord2[0] + (max_coord2[0]-min_coord2[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord2[1] + (max_coord2[1]-min_coord2[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord2[2] + (max_coord2[2]-min_coord2[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D* found[k2_];
        array_indexes_type number_found = 0;
        tree2_.search_knn(x, &min_dist, found, k2_, &number_found);
    }
    end_true_k_2 = clock();


    //test the kdtree3

    std::vector<gaussian_kernel2_3D> kernels3;
    for(array_indexes_type i=0;i<N;i++){
        point3d mu;
        /*for(dimension_type j=0;j<3;j++){
            mu.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        mu.x= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.y= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.z= 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double* scale_ = new float_double[3];
        random_scale(scale_);
        for(dimension_type j=0;j<3;j++){
            scale_[j] = 4*scale_[j];
        }
        float_double* q = new float_double[4];
        random_unitary_quaternion(q);
        float_double weight = ((float_double)rand()/(float_double)RAND_MAX);
        kernels3.push_back(gaussian_kernel2_3D(mu, weight , scale_, q));
        free(q);
        free(scale_);
    }
    std::cout << "kernels3 created" << std::endl;
    kd_tree3 tree3(&kernels3);
    kdtree_node3 root3(&kernels3);
    tree3.root = &root3;
    std::cout << "tree3 created" << std::endl;
    //std::cout << tree3 <<std::endl;
    //make random points
    clock_t start_uniform_1_3, end_uniform_1_3, start_uniform_k_3, end_uniform_k_3, start_true_1_3, end_true_1_3, start_true_k_3, end_true_k_3;
    start_uniform_1_3 = clock();
    for (int i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D res;
        tree3.search_1nn(&res, x, &min_dist_kd);
        //std::cout << min_dist_kd << std::endl;
        /*
        //compute the distance by brute force to check
        float_double min_dist = kernels3[0].distance(x);
        array_indexes_type min_dist_ind = std::numeric_limits<array_indexes_type>::max();
        for(array_indexes_type j=1;j<N;j++){
            float_double dist = kernels3[j].distance(x);
            if (dist < min_dist){
                min_dist = dist;
                min_dist_ind = j;
            }
        }
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist_kd << " " << min_dist << std::endl;
        if (abs(min_dist_kd - min_dist) > 0){
            std::cout << "error" << std::endl;
            std::cout << min_dist_kd << " " << min_dist << std::endl;
            std::cout << res << kernels3[min_dist_ind] << std::endl;
            std::cout << x << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels3[j] << std::endl;
                std::cout << kernels3[j].distance(x) << std::endl;
            }
            break;
        }
        else{
            if(std::isnan(min_dist_kd) || std::isnan(min_dist)){
                std::cout << "error" << std::endl;
                std::cout << min_dist_kd << " " << min_dist << std::endl;
                std::cout << res << kernels3[min_dist_ind] << std::endl;
                std::cout << x << std::endl;
                for(array_indexes_type j=0;j<N;j++){
                    std::cout << kernels3[j] << std::endl;
                    std::cout << kernels3[j].distance(x) << std::endl;
                }
                break;
            }
        }
        */
    }
    end_uniform_1_3 = clock();

    //search knn
    const array_indexes_type k3 = 20;
    start_uniform_k_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D* found[k3];
        array_indexes_type number_found = 0;
        tree3.search_knn(x, &min_dist, found, k3, &number_found);
        /*
        //compute the distance by brute force to check
        //std::cout << "brute force" << std::endl;
        std::vector<float_double> min_dist_brute;
        for (array_indexes_type j=0;j<k3;j++){
            min_dist_brute.push_back(kernels3[j].distance(x));
        }
        //sort the 2 vectors
        std::sort(min_dist_brute.begin(), min_dist_brute.end());
        for(array_indexes_type j=k3;j<N;j++){
            //put the new distance in the list
            float_double dist = kernels3[j].distance(x);
            //push the new distance
            min_dist_brute.push_back(dist);
            std::sort(min_dist_brute.begin(), min_dist_brute.end());
            min_dist_brute.pop_back();
        }
        
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist << " " << min_dist_brute[k3-1] << std::endl;
        //for(array_indexes_type j=0;j<k3;j++){
        //    std::cout << min_dist_brute[j] << std::endl;
        //}
        bool error = false;
        for(array_indexes_type j=0;j<number_found;j++){
            if (abs(found[j]->distance(x) - min_dist_brute[j]) > 0){
                error = true;
                std::cout << "error at " << j << std::endl;
                std::cout << found[j]->distance(x) << " " << min_dist_brute[j] << std::endl;
            }
        }
        for(array_indexes_type j=number_found;j<k3;j++){
            if(min_dist_brute[j] != std::min(std::numeric_limits<float_double>::max(),-negligeable_val_when_exp)){
                error = true;
                std::cout << "error at " << j << std::endl;
            }
        }
        if (error){
            std::cout << "error" << std::endl;
            std::cout << "min_dist" << " " << "min_dist_brute" << std::endl;
            std::cout << min_dist << " " << min_dist_brute[k3-1] << std::endl;
            std::cout << "x"<<std::endl;
            std::cout << x << std::endl;
            std::cout << "number_found" << std::endl;
            std::cout << number_found << std::endl;
            std::cout << std::endl;
            std::cout <<  " " << "found[j]" << std::endl;
            std::cout << "min_dist_brute[j]" << " " << "found[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<k3;j++){
                std::cout  << " "<<found[j] << std::endl;
                std::cout << min_dist_brute[j] <<" "<< (*found)[j].distance(x) << std::endl;
            }
            std::cout << std::endl;
            std::cout << "kernels[j]" << " " << "kernels[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels3[j] << std::endl;
                std::cout << kernels3[j].distance(x) << std::endl;
            }
            break;
        }
        */
    }
    end_uniform_k_3 = clock();

    //load true gaussian values from a ply file
    //typical header:
    /*
    ...
    */
    std::vector<gaussian_kernel2_3D> kernels3_;
    std::ifstream infile3("input.ply", std::ios_base::binary);

    // "Parse" header (it has to be a specific format anyway)
    std::string buff3;
    std::getline(infile3, buff3);
    std::getline(infile3, buff3);

    std::string dummy3;
    std::getline(infile3, buff3);
    std::stringstream ss3(buff3);
    int count3;
    ss3 >> dummy3 >> dummy3 >> count3;

    // Output number of Gaussians contained
    std::cout << "Loading " << count3 << " Gaussian splats" << std::endl;

    while (std::getline(infile3, buff3))
        if (buff3.compare("end_header") == 0)
            break;
    
    // Read all Gaussians
    array_indexes_type i3 = 0;

    while (infile3.peek() != EOF){
        float x, y, z;
        float nx, ny, nz;
        float opacity;
        float scale_0, scale_1, scale_2;
        float rot_0, rot_1, rot_2, rot_3;
        infile3.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&nx), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&ny), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&nz), sizeof(float));
        for(int j=0;j< 3+45;j++){
            float dummy;
            infile3.read(reinterpret_cast<char*>(&dummy), sizeof(float));
        }
        infile3.read(reinterpret_cast<char*>(&opacity), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&scale_0), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&scale_1), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&scale_2), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&rot_0), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&rot_1), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&rot_2), sizeof(float));
        infile3.read(reinterpret_cast<char*>(&rot_3), sizeof(float));

        //normalize the quaternion
        float norm = sqrt(rot_0*rot_0+rot_1*rot_1+rot_2*rot_2+rot_3*rot_3);
        rot_0 /= norm;
        rot_1 /= norm;
        rot_2 /= norm;
        rot_3 /= norm;

        //exponentiate the scale
        scale_0 = exp(scale_0);
        scale_1 = exp(scale_1);
        scale_2 = exp(scale_2);

        //opacity pass throught sigmoid
        opacity =( 1/(1+exp(-opacity)));

        //create scale and quaternion vectors
        float_double scale[3];
        scale[0] = (float_double) scale_0;
        scale[1] = (float_double) scale_1;
        scale[2] = (float_double) scale_2;
        float_double q[4];
        q[0] = (float_double) rot_0;
        q[1] = (float_double) rot_1;
        q[2] = (float_double) rot_2;
        q[3] = (float_double) rot_3;

        /*float_double mu[3];
        mu[0] = (float_double)x;
        mu[1] = (float_double)y;
        mu[2] = (float_double)z;*/
        point3d mu;
        mu.x = (float_double)x;
        mu.y = (float_double)y;
        mu.z = (float_double)z;

        kernels3_.push_back(gaussian_kernel2_3D(mu, opacity, scale, q));
        i3++;
    }
    std::cout << "kernels3 loaded" << std::endl;

    //create the tree
    kd_tree3 tree3_(&kernels3_);
    kdtree_node3 root3_(&kernels3_);
    tree3_.root = &root3_;
    float_double min_coord3[3] = {std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max(),std::numeric_limits<float_double>::max()};
    float_double max_coord3[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    /*for(dimension_type i=0;i<3;i++){
        min_coord3[i] = root3_.ranges[i];
        max_coord3[i] = root3_.ranges[i+3];
    }*/
    min_coord3[0] = root3_.range0;
    min_coord3[1] = root3_.range1;
    min_coord3[2] = root3_.range2;
    max_coord3[0] = root3_.range3;
    max_coord3[1] = root3_.range4;
    max_coord3[2] = root3_.range5;
    //std::cout << min_coord3[0] << " " << min_coord3[1] << " " << min_coord3[2] << std::endl;
    //std::cout << max_coord3[0] << " " << max_coord3[1] << " " << max_coord3[2] << std::endl;
    
    //search the 1nn
    float_double min_dist_ever_seen3 = std::numeric_limits<float_double>::max();
    point3d x_ever_seen3;
    start_true_1_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord3[j] + (max_coord3[j]-min_coord3[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D res;
        tree3_.search_1nn(&res, x, &min_dist_kd);
        /*//compare the distance with the one given by the other tree on the same point
        double min_dist = res.distance(x);
        gaussian_kernel<3,3> res_;
        tree_.search_1nn(&res_, x, &min_dist);
        std::cout << min_dist_kd << " " << min_dist << std::endl;
        */
    }
    end_true_1_3 = clock();
    std::cout << "end 1nn" << std::endl;
    /*std::cout << min_dist_ever_seen3 << std::endl;
    std::cout << x_ever_seen3 << std::endl;*/
    //search the knn
    const array_indexes_type k3_ = 20;
    start_true_k_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord3[j] + (max_coord3[j]-min_coord3[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = std::numeric_limits<float_double>::max();
        gaussian_kernel2_3D* found[k3_];
        array_indexes_type number_found = 0;
        tree3_.search_knn(x, &min_dist, found, k3_, &number_found);
    }
    end_true_k_3 = clock();

    
    std::cout << "end" << std::endl;
    std::cout << " ignored evaluations of gaussian kernel less than " << exp(negligeable_val_when_exp) << std::endl;
    std::cout <<"time statistics" << std::endl;
    /*std::cout << "uniform 1nn 1 : " << (float_double)(end_uniform_1-start_uniform_1)/CLOCKS_PER_SEC << std::endl;
    std::cout << "uniform knn 1 : " << (float_double)(end_uniform_k-start_uniform_k)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true 1nn 1 : " << (float_double)(end_true_1-start_true_1)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true knn 1 : " << (float_double)(end_true_k-start_true_k)/CLOCKS_PER_SEC << std::endl;*/
    std::cout << "uniform 1nn 2 : " << (float_double)(end_uniform_1_2-start_uniform_1_2)/CLOCKS_PER_SEC << std::endl;
    std::cout << "uniform knn 2 : " << (float_double)(end_uniform_k_2-start_uniform_k_2)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true 1nn 2 : " << (float_double)(end_true_1_2-start_true_1_2)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true knn 2 : " << (float_double)(end_true_k_2-start_true_k_2)/CLOCKS_PER_SEC << std::endl;
    std::cout << "uniform 1nn 3 : " << (float_double)(end_uniform_1_3-start_uniform_1_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "uniform knn 3 : " << (float_double)(end_uniform_k_3-start_uniform_k_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true 1nn 3 : " << (float_double)(end_true_1_3-start_true_1_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true knn 3 : " << (float_double)(end_true_k_3-start_true_k_3)/CLOCKS_PER_SEC << std::endl;


    return 0;
}