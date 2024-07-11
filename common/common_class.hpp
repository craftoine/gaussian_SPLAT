#ifdef _USE_CUDA_ 

#else
   template <typename T>
   inline T max(T a, T b){
       return std::max(a, b);
   }
    template <typename T>
    inline T min(T a, T b){
         return std::min(a, b);
    }
#endif

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
