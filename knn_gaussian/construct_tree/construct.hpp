#include <vector>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <iostream>


#define dimension_type unsigned char

#define array_indexes_type unsigned int
#define max_array_indexes_type 4294967295

#define float_double float
#define max_float_double 1e+37


#define Simple_heuristic true
#define sigma_heuristic true
#define sigma_split_midle false

#define max_sigma_factor 150.
#define max_depth 27
#define max_leaf_size 1
#define min_node_size 1

#define negligeable_val_when_exp (float_double)-6.
//negligeable gaussian value is so exp(negligeable_val_when_exp)
#define K_knn 20

#define Size_point_block 2


#ifdef _USE_CUDA_ 
__host__ __device__
#endif
inline float_double max_(float_double a, float_double b){
    if (a > b){
        return a;
    }
    return b;
}

#ifdef _USE_CUDA_ 
__host__ __device__
#endif
inline float_double min_(float_double a, float_double b){
    if (a < b){
        return a;
    }
    return b;
}


struct
#ifdef _USE_CUDA_
__align__(16)
#endif
float_double3{
    float_double x;
    float_double y;
    float_double z;
};

struct
#ifdef _USE_CUDA_
__align__(16)
#endif
float_double4{
    float_double x;
    float_double y;
    float_double z;
    float_double w;
};


class 
#ifdef _USE_CUDA_
__align__(16)
#endif
point3d{

    public:
    float_double x;
    float_double y;
    float_double z;
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    point3d(float_double x, float_double y, float_double z){
        this->x = x;
        this->y = y;
        this->z = z;
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    point3d(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double norm(){
        return sqrt(x*x+y*y+z*z);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double distance(point3d p){
        #ifdef __CUDA_ARCH__
            return norm3d(x-p.x, y-p.y, z-p.z);
        #else
            return sqrt((x-p.x)*(x-p.x)+(y-p.y)*(y-p.y)+(z-p.z)*(z-p.z));
        #endif
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double square_distance(point3d p){
        #ifdef __CUDA_ARCH__
            float_double res = norm3d(x-p.x, y-p.y, z-p.z);
            return res*res;
        #else
            return (x-p.x)*(x-p.x)+(y-p.y)*(y-p.y)+(z-p.z)*(z-p.z);
        #endif
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double square_norm(){
        #ifdef __CUDA_ARCH__
            float_double res = norm3d(x, y, z);
            return res*res;
        #else
            return x*x+y*y+z*z;
        #endif
    }
    //overload []
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double operator[](dimension_type i){
        if (i == 0){
            return x;
        }
        if (i == 1){
            return y;
        }
        if (i == 2){
            return z;
        }
        return 0.0;
        //raise an error
        //throw std::invalid_argument("i != 0,1,2");
        //exit(6);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline point3d operator+(point3d p){
        return point3d(x+p.x, y+p.y, z+p.z);
    }
    #ifdef _USE_CUDA_
    __host__ __device__
    #endif
    inline point3d operator+(float_double a){
        return point3d(x+a, y+a, z+a);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline point3d operator-(point3d p){
        return point3d(x-p.x, y-p.y, z-p.z);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline point3d operator*(float_double a){
        return point3d(x*a, y*a, z*a);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline point3d operator/(float_double a){
        return point3d(x/a, y/a, z/a);
    }
    
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline friend point3d operator*(float_double a, point3d p){
        return point3d(a*p.x, a*p.y, a*p.z);
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double operator*(point3d p){
        return x*p.x+y*p.y+z*p.z;
    }
    friend std::ostream& operator<<(std::ostream& os, const point3d& p){
        os << "point3d(" << p.x << "," << p.y << "," << p.z << ")";
        return os;
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline bool operator==(point3d p){
        return x == p.x && y == p.y && z == p.z;
    }

   /*def hilbert_curve_cord_3d(p):
    c = 1/2
    d = 0
    for s in range(20):
        #we enter from the botomm back left corner and exit on the top back left corner
        if p[0] < 0.5 and p[1] < 0.5 and p[2] < 0.5:
            p[0] = 2*p[0]
            p[1] = 2*p[1]
            p[2] = 2*p[2]
            #we are in the bottom back left corner
            #we enter it from the bottom back left corner
            #we exit it on the bottom front left corner
            #we rotate the coordinates
            p[0], p[1], p[2] = p[0], p[2], p[1]
        
        elif p[0] < 0.5 and p[1] > 0.5 and p[2] < 0.5:
            p[0] = 2*p[0]
            p[1] = 2*(p[1]-0.5)
            p[2] = 2*p[2]
            #we are in the botom front left corner
            #we enter it from the bottom back left corner
            #we exit it on the bottom back right corner
            #we rotate the coordinates
            p[0], p[1], p[2] = p[2], p[1], p[0]
            d += c/4
        
        elif p[0] > 0.5 and p[1] > 0.5 and p[2] < 0.5:
            p[0] = 2*(p[0]-0.5)
            p[1] = 2*(p[1]-0.5)
            p[2] = 2*p[2]
            #we are in the bottom front right corner
            #we enter it from the bottom back left corner
            #we exit it on the top back left corner
            #we do not rotate the coordinates
            d+= c/2
        
        elif p[0] > 0.5 and p[1] < 0.5 and p[2] < 0.5:
            p[0] = 2*(p[0]-0.5)
            p[1] = 2*p[1]
            p[2] = 2*p[2]
            #we are in the bottom back right corner
            #we enter it from the top front left corner
            #we exit it on the top front right corner
            #we rotate the coordinates
            p[0], p[1], p[2] = 1-p[1], 1-p[2], p[0]
            d+= 3*c/4

        elif p[0] > 0.5 and p[1] < 0.5 and p[2] > 0.5:
            p[0] = 2*(p[0]-0.5)
            p[1] = 2*p[1]
            p[2] = 2*(p[2]-0.5)
            #we are in the top back right corner
            #we enter it from the bottom right front corner
            #we exit it on the bottom front left corner
            #we rotate the coordinates
            p[0], p[1], p[2] = 1-p[1], p[2], 1-p[0]
            d+= c

        elif p[0] > 0.5 and p[1] > 0.5 and p[2] > 0.5:
            p[0] = 2*(p[0]-0.5)
            p[1] = 2*(p[1]-0.5)
            p[2] = 2*(p[2]-0.5)
            #we are in the top front right corner
            #we enter it from the bottom back right corner
            #we exit it on the top back left corner
            #we do not rotate the coordinates
            d+= 5*c/4
        elif p[0] < 0.5 and p[1] > 0.5 and p[2] > 0.5:
            p[0] = 2*p[0]
            p[1] = 2*(p[1]-0.5)
            p[2] = 2*(p[2]-0.5)
            #we are in the top front left corner
            #we enter it from the top back right corner
            #we exit it on the top back left corner
            #we rotate the coordinates
            p[0], p[1], p[2] = 1-p[2], p[1], 1-p[0]
            d+= 6*c/4
        else:
            p[0] = 2*p[0]
            p[1] = 2*p[1]
            p[2] = 2*(p[2]-0.5)
            #we are in the top back left corner
            #we enter it from the top front left corner
            #we exit it on the top back left corner
            #we rotate the coordinates
            p[0], p[1], p[2] = p[0], 1-p[2], 1-p[1]
            d+= 7*c/4
        c = c/8
    #return max(0,min((d-3/8)*8/2,1))
    return d*/
    double hilbert_curve_cord(point3d mins_cord, point3d max_cord, bool debug=false){
        double c = 0.5;
        float_double x__ = (this->x - mins_cord.x)/(max_cord.x - mins_cord.x);
        float_double y__ = (this->y - mins_cord.y)/(max_cord.y - mins_cord.y);
        float_double z__ = (this->z - mins_cord.z)/(max_cord.z - mins_cord.z);
        double d = 0;
        for(int s=0;s<15;s++){
            if (debug){
                std::cout << "STEP " << s << " " << d << " " << c << std::endl;
                std::cout << x__ << " " << y__ << " " << z__ << std::endl;
            }            
            float_double x_;
            float_double y_;
            float_double z_;
            if (x__ < 0.5 && y__ < 0.5 && z__ < 0.5){
                if(debug){
                    std::cout << "case 0" << std::endl;
                }
                x__= 2*x__;
                y__= 2*y__;
                z__= 2*z__;
                x_ = x__;
                y_ = z__;
                z_ = y__;
            }
            else if (x__ < 0.5 && y__ > 0.5 && z__ < 0.5){
                if(debug){
                    std::cout << "case 1" << std::endl;
                }
                x__= 2*x__;
                y__= 2*(y__-0.5);
                z__= 2*z__;
                x_ = z__;
                y_ = y__;
                z_ = x__;
                d += c/4;
            }
            else if (x__ > 0.5 && y__ > 0.5 && z__ < 0.5){
                if(debug){
                    std::cout << "case 2" << std::endl;
                }
                x__= 2*(x__-0.5);
                y__= 2*(y__-0.5);
                z__= 2*z__;
                x_ = x__;
                y_ = y__;
                z_ = z__;
                d+= c/2;
            }
            else if (x__ > 0.5 && y__ < 0.5 && z__ < 0.5){
                if(debug){
                    std::cout << "case 3" << std::endl;
                }
                x__= 2*(x__-0.5);
                y__= 2*y__;
                z__= 2*z__;
                x_ = 1-y__;
                y_ = 1-z__;
                z_ = x__;
                d+= 3*c/4;
                if(debug){
                    std::cout << c << " " << c/4 << " " << 3*c/4 << " " << d << std::endl;
                }
            }
            else if (x__ > 0.5 && y__ < 0.5 && z__ > 0.5){
                if(debug){
                    std::cout << "case 4" << std::endl;
                }
                x__= 2*(x__-0.5);
                y__= 2*y__;
                z__= 2*(z__-0.5);
                x_ = 1-y__;
                y_ = z__;
                z_ = 1-x__;
                d+= c;
            }
            else if (x__ > 0.5 && y__ > 0.5 && z__ > 0.5){
                if(debug){
                    std::cout << "case 5" << std::endl;
                }
                x__= 2*(x__-0.5);
                y__= 2*(y__-0.5);
                z__= 2*(z__-0.5);
                x_ = x__;
                y_ = y__;
                z_ = z__;
                d+= 5*c/4;
            }
            else if (x__ < 0.5 && y__ > 0.5 && z__ > 0.5){
                if(debug){
                    std::cout << "case 6" << std::endl;
                }
                x__= 2*x__;
                y__= 2*(y__-0.5);
                z__= 2*(z__-0.5);
                x_ = 1-z__;
                y_ = y__;
                z_ = 1-x__;
                d+= 6*c/4;
            }
            else{
                if(debug){
                    std::cout << "case 7" << std::endl;
                }
                x__= 2*x__;
                y__= 2*y__;
                z__= 2*(z__-0.5);
                x_ = x__;
                y_ = 1-z__;
                z_ = 1-y__;
                d+= 7*c/4;
            }
            c = c/8;
            x__ = x_;
            y__ = y_;
            z__ = z_;
        }
        return d;
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
            return -max_float_double;
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
class 
#ifdef _USE_CUDA_
//__align__(16)
#endif
gaussian_kernel2_3D{
    public:
        /*float_double scale0;
        float_double scale1;
        float_double scale2;*/
        float_double3 scales3;
        /*float_double quaternions0;
        float_double quaternions1;
        float_double quaternions2;
        float_double quaternions3;*/
        float_double4 quaternions4;
        point3d mu;
        float_double log_weight;
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    gaussian_kernel2_3D(point3d mu, float_double log_weight, float_double scale[3], float_double* quaternions){
        this->mu = mu;
        this->log_weight = log_weight;
        this->scales3.x = scale[0];
        this->scales3.y = scale[1];
        this->scales3.z = scale[2];
        this->quaternions4.x = quaternions[0];
        this->quaternions4.y = quaternions[1];
        this->quaternions4.z = quaternions[2];
        this->quaternions4.w = quaternions[3];
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    gaussian_kernel2_3D(){
        this->mu = point3d();
        this->log_weight = 0;
        this->scales3.x = 0;
        this->scales3.y = 0;
        this->scales3.z = 0;
        this->quaternions4.x = 0;
        this->quaternions4.y = 0;
        this->quaternions4.z = 0;
        this->quaternions4.w = 0;
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double operator()(point3d x){
        // log_weight - (x-mu)^T (scale*R)"^-2"(x-mu)
        // R is the rotation matrix
        // (x-mu) is the vector from mu to x
        // (x-mu)^T (scale*R)"^-2"(x-mu) = (x-mu)^T(scale*R)^-1^T*(scale*R)^-1(x-mu)
        //                               = ((scale*R)^-1(x-mu))^T*((scale*R)^-1(x-mu))
        // (scale*R)^-1 = scale^-1*R^-1
        //R^-1 = R^T

        point3d x_mu = x - mu;
        //point3d R_inv_x_mu;
        /*float_double R00 = 1-2*quaternions4.z*quaternions4.z-2*quaternions4.w*quaternions4.w;
        float_double R01 = 2*quaternions4.y*quaternions4.z-2*quaternions4.x*quaternions4.w;
        float_double R02 = 2*quaternions4.x*quaternions4.z+2*quaternions4.y*quaternions4.w;
        R_inv_x_mu.x = R00*x_mu.x + R01*x_mu.y + R02*x_mu.z;
        float_double R10 = 2*quaternions4.y*quaternions4.z+2*quaternions4.x*quaternions4.w;
        float_double R11 = 1-2*quaternions4.y*quaternions4.y-2*quaternions4.w*quaternions4.w;
        float_double R12 = 2*quaternions4.z*quaternions4.w-2*quaternions4.x*quaternions4.y;
        R_inv_x_mu.y = R10*x_mu.x + R11*x_mu.y + R12*x_mu.z;
        float_double R20 = 2*quaternions4.y*quaternions4.w-2*quaternions4.x*quaternions4.z;
        float_double R21 = 2*quaternions4.x*quaternions4.y+2*quaternions4.z*quaternions4.w;
        float_double R22 = 1-2*quaternions4.y*quaternions4.y-2*quaternions4.z*quaternions4.z;
        R_inv_x_mu.z = R20*x_mu.x + R21*x_mu.y + R22*x_mu.z;*/
        point3d scale_inv_R_inv_x_mu;
        
        scale_inv_R_inv_x_mu.x = ((1-2*quaternions4.z*quaternions4.z-2*quaternions4.w*quaternions4.w)*x_mu.x + (2*quaternions4.y*quaternions4.z-2*quaternions4.x*quaternions4.w)*x_mu.y + (2*quaternions4.x*quaternions4.z+2*quaternions4.y*quaternions4.w)*x_mu.z)/scales3.x;
        scale_inv_R_inv_x_mu.y = ((2*quaternions4.y*quaternions4.z+2*quaternions4.x*quaternions4.w)*x_mu.x + (1-2*quaternions4.y*quaternions4.y-2*quaternions4.w*quaternions4.w)*x_mu.y + (2*quaternions4.z*quaternions4.w-2*quaternions4.x*quaternions4.y)*x_mu.z)/scales3.y;
        scale_inv_R_inv_x_mu.z = ((2*quaternions4.y*quaternions4.w-2*quaternions4.x*quaternions4.z)*x_mu.x + (2*quaternions4.x*quaternions4.y+2*quaternions4.z*quaternions4.w)*x_mu.y + (1-2*quaternions4.y*quaternions4.y-2*quaternions4.z*quaternions4.z)*x_mu.z)/scales3.z;
        float_double res = scale_inv_R_inv_x_mu.square_norm();
        //scale_inv_R_inv_x_mu.x = R_inv_x_mu.x/scales3.x;
        //scale_inv_R_inv_x_mu.y = R_inv_x_mu.y/scales3.y;
        //scale_inv_R_inv_x_mu.z = R_inv_x_mu.z/scales3.z;
        return log_weight - /*scale_inv_R_inv_x_mu.square_norm()/2;*/res/2;
    }
    #ifdef _USE_CUDA_ 
    __host__ __device__
    #endif
    inline float_double distance(point3d x){
        //if the kernel does not have weight infinit
        //if(!(std::isinf(log_weight))){
        #ifdef __CUDA_ARCH__
            return -max(this->operator()(x), negligeable_val_when_exp);
        #else
            return -std::max(this->operator()(x), negligeable_val_when_exp);
        #endif
        //}
        //else{
            //return minimum value of a float_double
            //return -max_float_double;
        //}
    }
    gaussian_kernel2_1D convert(){
        float_double max_eigan = std::max(scales3.x*scales3.x, std::max(scales3.y*scales3.y, scales3.z*scales3.z));
        return gaussian_kernel2_1D(mu, log_weight, (float_double)sqrt(max_eigan));
    }
    friend std::ostream& operator<<(std::ostream& os, const gaussian_kernel2_3D& g){
        os << "gaussian_kernel(" << g.mu << "," << g.log_weight <<",[";
        os << g.scales3.x << ",";
        os << g.scales3.y << ",";
        os << g.scales3.z << "],[";
        os << g.quaternions4.x << ",";
        os << g.quaternions4.y << ",";
        os << g.quaternions4.z << ",";
        os << g.quaternions4.w << "])";
        return os;
    }
};


#if max_leaf_size > 1
    struct leaf{
        array_indexes_type is_leaf;
        array_indexes_type start;
        array_indexes_type end;
    };
#else
    struct __align__(16) leaf{
        array_indexes_type is_leaf;
        array_indexes_type start;
    };
#endif
struct range_data{
    float_double range0;
    float_double range1;
    float_double range2;
    float_double range3;
    float_double range4;
    float_double range5;
    float_double range6;
    float_double range7;
};

struct __align__(16) node{
    array_indexes_type right;
    range_data left_range;
    range_data right_range;
};
    








class kdtree_node3{
    public:
    array_indexes_type size;
    char* data;
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
/* BROKEN    kdtree_node3* left(){
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
    }*/
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
    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type start, array_indexes_type end, dimension_type split_dim, array_indexes_type depth=0){
        //std::cout << start << " " << end << std::endl;
        if(start<end){
            build(ks, start, end, split_dim, depth);
        }
        else{
            build_leaf(start, end, ks);
        }          
    }

    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type depth=0){
        if(ks->size()>1){
            build(ks, 0, ks->size()-1, 0, depth);
        }
        else{
            build_leaf(0, 0, ks);
        }
    }

    void build(std::vector<gaussian_kernel2_3D>* kernels, array_indexes_type start, array_indexes_type end, dimension_type split_dim, array_indexes_type depth=0){
        if(depth >= max_depth || (1+end)-start <= min_node_size){
            //std::cout << "leaf" << std::endl;
            //std::cout << (1+end)-start << " " << min_node_size << std::endl;
            //std::cout << depth << " " << max_depth << std::endl;
            build_leaf(start, end, kernels);
            return;
        }
        if(sigma_heuristic){
            float_double max_sigma = std::numeric_limits<float_double>::min();
            float_double min_sigma = max_float_double;
            
            for(array_indexes_type i=start;i<=end;i++){
                float_double sigma_i = abs((*kernels)[i].scales3.x);
                sigma_i = std::max(sigma_i, abs((*kernels)[i].scales3.y));
                sigma_i = std::max(sigma_i, abs((*kernels)[i].scales3.z));
                max_sigma = std::max(max_sigma, sigma_i);
                min_sigma = std::min(min_sigma, sigma_i);                
            }
            if (max_sigma/min_sigma > max_sigma_factor){
                float_double sigma_midle = (max_sigma+min_sigma)/2;
                //sort the kernels by sigma and plit in two according to the sigma_midle
                std::sort(kernels->begin()+start, kernels->begin()+end+1, [](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
                    float_double sigma1 = std::max(std::max(abs(g1.scales3.x), abs(g1.scales3.y)), abs(g1.scales3.z));
                    float_double sigma2 = std::max(std::max(abs(g2.scales3.x), abs(g2.scales3.y)), abs(g2.scales3.z));
                    return sigma1 < sigma2;
                });
                #if sigma_split_midle
                    array_indexes_type new_size = (end-start)/2;
                    array_indexes_type left_end = start + new_size;
                    array_indexes_type split = left_end + 1;
                #else
                    array_indexes_type split = start;
                    for(array_indexes_type i=start;i<=end;i++){
                        float_double sigma_i = std::max(std::max(abs((*kernels)[i].scales3.x), abs((*kernels)[i].scales3.y)), abs((*kernels)[i].scales3.z));
                        if (sigma_i > sigma_midle){
                            split = i;
                            break;
                        }
                    }
                #endif
                kdtree_node3 left = kdtree_node3(kernels, start, split-1, (split_dim+1)%3, depth+1);
                kdtree_node3 right = kdtree_node3(kernels, split, end, (split_dim+1)%3, depth+1);
                //array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char) + 2*(3+3+1+1)*sizeof(float_double)/sizeof(char) + left.size + right.size;
                array_indexes_type size = sizeof(node) + left.size + right.size;
                //char* data = (char*)malloc(size);
                char* data = new char[size];
                /*const unsigned char size_local = 2*sizeof(array_indexes_type)/sizeof(unsigned char) + 2*(3+3+1+1)*sizeof(float_double)/sizeof(unsigned char);
                ((array_indexes_type*)data)[0] = size_local;
                const unsigned char ofset1 = sizeof(array_indexes_type)/sizeof(char);
                ((float_double*)(data+ofset1))[0] = left.range0;
                ((float_double*)(data+ofset1))[1] = left.range1;
                ((float_double*)(data+ofset1))[2] = left.range2;
                ((float_double*)(data+ofset1))[3] = left.range3;
                ((float_double*)(data+ofset1))[4] = left.range4;
                ((float_double*)(data+ofset1))[5] = left.range5;
                ((float_double*)(data+ofset1))[6] = left.range6;
                ((float_double*)(data+ofset1))[7] = left.range7;
                const unsigned char ofset2 = ofset1 +(3+3+1+1)*sizeof(float_double)/sizeof(char);
                ((array_indexes_type*)(data+ofset2))[0] = size_local+left.size;
                const unsigned char ofset3 = ofset2 + sizeof(array_indexes_type)/sizeof(char);
                ((float_double*)(data+ofset3))[0] = right.range0;
                ((float_double*)(data+ofset3))[1] = right.range1;
                ((float_double*)(data+ofset3))[2] = right.range2;
                ((float_double*)(data+ofset3))[3] = right.range3;
                ((float_double*)(data+ofset3))[4] = right.range4;
                ((float_double*)(data+ofset3))[5] = right.range5;
                ((float_double*)(data+ofset3))[6] = right.range6;
                ((float_double*)(data+ofset3))[7] = right.range7;*/
                const unsigned char size_local = sizeof(node);
                ((node *) data)->left_range.range0 = left.range0;
                ((node *) data)->left_range.range1 = left.range1;
                ((node *) data)->left_range.range2 = left.range2;
                ((node *) data)->left_range.range3 = left.range3;
                ((node *) data)->left_range.range4 = left.range4;
                ((node *) data)->left_range.range5 = left.range5;
                ((node *) data)->left_range.range6 = left.range6;
                ((node *) data)->left_range.range7 = left.range7;
                ((node *) data)->right = size_local+left.size;
                ((node *) data)->right_range.range0 = right.range0;
                ((node *) data)->right_range.range1 = right.range1;
                ((node *) data)->right_range.range2 = right.range2;
                ((node *) data)->right_range.range3 = right.range3;
                ((node *) data)->right_range.range4 = right.range4;
                ((node *) data)->right_range.range5 = right.range5;
                ((node *) data)->right_range.range6 = right.range6;
                ((node *) data)->right_range.range7 = right.range7;
                for(array_indexes_type i=0;i<left.size;i++){
                    data[size_local+i] = left.data[i];
                }
                const array_indexes_type ofset4 = size_local + left.size;
                for(array_indexes_type i=0;i<right.size;i++){
                    data[ofset4+i] = right.data[i];
                }
                this->size = size;
                this->data = data;
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
                return;
            }
        }
        std::sort(kernels->begin()+start, kernels->begin()+end+1, [&split_dim](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
            return g1.mu[split_dim] < g2.mu[split_dim];
        });
        kdtree_node3 left;
        kdtree_node3 right;
        if(Simple_heuristic){
            array_indexes_type new_size = (end-start)/2;
            array_indexes_type left_end = start + new_size;
            array_indexes_type right_start = left_end + 1;
            left = kdtree_node3(kernels, start, left_end, (split_dim+1)%3, depth+1);
            right = kdtree_node3(kernels, right_start, end, (split_dim+1)%3, depth+1);
        }
        else{
            point3d midle;
            float_double max_dim[3];
            float_double min_dim[3];
            for(dimension_type i=0;i<3;i++){
                max_dim[i] = std::numeric_limits<float_double>::min();
                min_dim[i] = max_float_double;
            }
            
            for(array_indexes_type j=start;j<=end;j++){
                for(dimension_type i=0;i<3;i++){
                    max_dim[i] = std::max(max_dim[i], (*kernels)[j].mu[i]);
                    min_dim[i] = std::min(min_dim[i], (*kernels)[j].mu[i]);
                }
            }
            midle.x = (max_dim[0]+min_dim[0])/2;
            midle.y = (max_dim[1]+min_dim[1])/2;
            midle.z = (max_dim[2]+min_dim[2])/2;
            array_indexes_type split = 0;
            array_indexes_type st = 0;
            array_indexes_type en = end-start;
            while(st < en){
                split = (st+en)/2;
                float_double distance_left = max_float_double;
                float_double distance_right = max_float_double;
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
            
            split = std::max(split, (array_indexes_type)1);
            split = std::min(split, end-start);
            left = kdtree_node3(kernels, start, split+start-1, (split_dim+1)%3, depth+1);
            right = kdtree_node3(kernels, split+start, end, (split_dim+1)%3, depth+1);
        }
        //array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char)+2*(3+3+1+1)*sizeof(float_double)/sizeof(char) + left.size + right.size;
        array_indexes_type size = sizeof(node) + left.size + right.size;
        char* data = new char[size];
        /*const unsigned char size_local = 2*sizeof(array_indexes_type)/sizeof(unsigned char) + 2*(3+3+1+1)*sizeof(float_double)/sizeof(unsigned char);
        ((array_indexes_type*)data)[0] = size_local;
        const unsigned char ofset1 = sizeof(array_indexes_type)/sizeof(char);
        ((float_double*)(data+ofset1))[0] = left.range0;
        ((float_double*)(data+ofset1))[1] = left.range1;
        ((float_double*)(data+ofset1))[2] = left.range2;
        ((float_double*)(data+ofset1))[3] = left.range3;
        ((float_double*)(data+ofset1))[4] = left.range4;
        ((float_double*)(data+ofset1))[5] = left.range5;
        ((float_double*)(data+ofset1))[6] = left.range6;
        ((float_double*)(data+ofset1))[7] = left.range7;
        const unsigned char ofset2 = ofset1 +(3+3+1+1)*sizeof(float_double)/sizeof(char);
        ((array_indexes_type*)(data+ofset2))[0] = size_local+left.size;
        const unsigned char ofset3 = ofset2 + sizeof(array_indexes_type)/sizeof(char);
        ((float_double*)(data+ofset3))[0] = right.range0;
        ((float_double*)(data+ofset3))[1] = right.range1;
        ((float_double*)(data+ofset3))[2] = right.range2;
        ((float_double*)(data+ofset3))[3] = right.range3;
        ((float_double*)(data+ofset3))[4] = right.range4;
        ((float_double*)(data+ofset3))[5] = right.range5;
        ((float_double*)(data+ofset3))[6] = right.range6;
        ((float_double*)(data+ofset3))[7] = right.range7;*/
        const unsigned char size_local = sizeof(node);
        ((node *) data)->left_range.range0 = left.range0;
        ((node *) data)->left_range.range1 = left.range1;
        ((node *) data)->left_range.range2 = left.range2;
        ((node *) data)->left_range.range3 = left.range3;
        ((node *) data)->left_range.range4 = left.range4;
        ((node *) data)->left_range.range5 = left.range5;
        ((node *) data)->left_range.range6 = left.range6;
        ((node *) data)->left_range.range7 = left.range7;
        ((node *) data)->right = size_local+left.size;
        ((node *) data)->right_range.range0 = right.range0;
        ((node *) data)->right_range.range1 = right.range1;
        ((node *) data)->right_range.range2 = right.range2;
        ((node *) data)->right_range.range3 = right.range3;
        ((node *) data)->right_range.range4 = right.range4;
        ((node *) data)->right_range.range5 = right.range5;
        ((node *) data)->right_range.range6 = right.range6;
        ((node *) data)->right_range.range7 = right.range7;
        for(array_indexes_type i=0;i<left.size;i++){
            data[size_local+i] = left.data[i];
        }
        const array_indexes_type ofset4 = size_local + left.size;
        for(array_indexes_type i=0;i<right.size;i++){
            data[ofset4+i] = right.data[i];
        }
        this->size = size;
        this->data = data;
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
    void build_leaf(array_indexes_type start, array_indexes_type end, std::vector<gaussian_kernel2_3D>* ks){
        if((1+end)-start > max_leaf_size){
            std::cout << "error: leaf size is too big" << std::endl;
            std::cout << (1+end)-start << " " << max_leaf_size << std::endl;
            exit(8);
        }
        #if max_leaf_size > 1
            array_indexes_type size = 3*sizeof(array_indexes_type)/sizeof(char);
            char* data = new char[size];
            /*((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = start;
            ((array_indexes_type*)data)[2] = end;*/
            ((leaf*)data)[0].is_leaf = 0;
            ((leaf*)data)[0].start = start;
            ((leaf*)data)[0].end = end;
        #else
            array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char);
            char* data = new char[size];
            /*((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = start;*/
            ((leaf*)data)[0].is_leaf = 0;
            ((leaf*)data)[0].start = start;
        #endif          
            
            
        gaussian_kernel2_1D kernel_max = (*ks)[start].convert();
        range0 = kernel_max.mu.x;
        range1 = kernel_max.mu.y;
        range2 = kernel_max.mu.z;
        range3 = kernel_max.mu.x;
        range4 = kernel_max.mu.y;
        range5 = kernel_max.mu.z;
        range6 = kernel_max.log_weight;
        range7 = kernel_max.scale;
        for (array_indexes_type i=start+1;i<=end;i++){
            gaussian_kernel2_1D kernel = (*ks)[i].convert();
            range0 = std::min(range0, kernel.mu.x);
            range1 = std::min(range1, kernel.mu.y);
            range2 = std::min(range2, kernel.mu.z);
            range3 = std::max(range3, kernel.mu.x);
            range4 = std::max(range4, kernel.mu.y);
            range5 = std::max(range5, kernel.mu.z);
            range6 = std::max(range6, kernel.log_weight);
            range7 = std::max(range7, kernel.scale);
        }
        this->size = size;
        this->data = data;
    }
    friend std::ostream& operator<<(std::ostream& os, const kdtree_node3 &k){
        if(((array_indexes_type*)k.data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)k.data)[1];
            os << "kdtree_node(" << start << "," << start << ",";
            os << "[" << k.range0 << "," << k.range3 << "],";
            os << "[" << k.range1 << "," << k.range4 << "],";
            os << "[" << k.range2 << "," << k.range5 << "],";
            os << k.range6 << "," << k.range7 << ")";
            return os;
        }
        os << "kdtree_node(";
        os << "[" << k.range0 << "," << k.range3 << "],";
        os << "[" << k.range1 << "," << k.range4 << "],";
        os << "[" << k.range2 << "," << k.range5 << "],";
        os << k.range6 << "," << k.range7 << ",";
        //kdtree_node3* left = const_cast<kdtree_node3*>(&k)->left();
        //kdtree_node3* right = const_cast<kdtree_node3*>(&k)->right();
        //os << *left << "," << *right << ")";
        return os;
    }
};