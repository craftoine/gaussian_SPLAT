#define _USE_CUDA_
#define M_PI 3.14159265358979323846
#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f
#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f
#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

#include "../knn_gaussian/construct_tree/construct.hpp"
#include "../knn_gaussian/common_header.hpp"
#define stop_T 0.0001
#define k_steps 10
#define k_steps_2 (10*k_steps)
#define min_sum_eval 0.0001
#define rendering_block_size 16
#define reordering_window_size 30
#define max_overlap 7
#define background_color point3d(0, 0, 0)
#define fov 60.
#define x_scaling 30
#define y_scaling 30
#define orthogonal false
#define angle_1 false
#define screen_dist 1.
//#define rescaling_integral_1
//#define rescaling_integral_2
#define min_integral_v 0.001
//#define can_buffer_sigma
//#define can_buffer_color
class __align__(16) float_double6{
    public:
    float_double f[6];
    __host__ __device__ inline float_double6(float_double f0, float_double f1, float_double f2, float_double f3, float_double f4, float_double f5){
        f[0] = f0;
        f[1] = f1;
        f[2] = f2;
        f[3] = f3;
        f[4] = f4;
        f[5] = f5;
    }
    __host__ __device__ inline float_double6(){
        f[0] = 0;
        f[1] = 0;
        f[2] = 0;
        f[3] = 0;
        f[4] = 0;
        f[5] = 0;
    }
    __host__ __device__ inline float_double6(float_double *f){
        this->f[0] = f[0];
        this->f[1] = f[1];
        this->f[2] = f[2];
        this->f[3] = f[3];
        this->f[4] = f[4];
        this->f[5] = f[5];
    }
    __host__ __device__ inline float_double& operator[](int i){
        return f[i];
    }
};
class rotation_matrix{
    //stored as quaternion
     public:
        float_double4 quaternions4;
        __host__ __device__ inline rotation_matrix(float_double4 quaternions4){
            this->quaternions4 = quaternions4;
        }
        __host__ __device__ inline rotation_matrix(){
            this->quaternions4 = {0, 0, 0, 0};
        }
        __host__ __device__ inline rotation_matrix(float_double ** R){
            float_double trace = R[0][0] + R[1][1] + R[2][2];
            if (trace > 0) {
                float_double s = 0.5 / sqrt(trace + 1.0);
                quaternions4.x = 0.25 / s;
                quaternions4.y = (R[2][1] - R[1][2]) * s;
                quaternions4.z = (R[0][2] - R[2][0]) * s;
                quaternions4.w = (R[1][0] - R[0][1]) * s;
            } else {
                if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
                    float_double s = 2.0 * sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]);
                    quaternions4.x = (R[2][1] - R[1][2]) / s;
                    quaternions4.y = 0.25 * s;
                    quaternions4.z = (R[0][1] + R[1][0]) / s;
                    quaternions4.w = (R[0][2] + R[2][0]) / s;
                } else if (R[1][1] > R[2][2]) {
                    float_double s = 2.0 * sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]);
                    quaternions4.x = (R[0][2] - R[2][0]) / s;
                    quaternions4.y = (R[0][1] + R[1][0]) / s;
                    quaternions4.z = 0.25 * s;
                    quaternions4.w = (R[1][2] + R[2][1]) / s;
                } else {
                    float_double s = 2.0 * sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]);
                    quaternions4.x = (R[1][0] - R[0][1]) / s;
                    quaternions4.y = (R[0][2] + R[2][0]) / s;
                    quaternions4.z = (R[1][2] + R[2][1]) / s;
                    quaternions4.w = 0.25 * s;
                }
            }
        }
        __host__ __device__ inline rotation_matrix(float_double*R){
            float_double trace = R[0] + R[4] + R[8];
            if (trace > 0) {
                float_double s = 0.5 / sqrt(trace + 1.0);
                quaternions4.x = 0.25 / s;
                quaternions4.y = (R[7] - R[5]) * s;
                quaternions4.z = (R[2] - R[6]) * s;
                quaternions4.w = (R[3] - R[1]) * s;
            } else {
                if (R[0] > R[4] && R[0] > R[8]) {
                    float_double s = 2.0 * sqrt(1.0 + R[0] - R[4] - R[8]);
                    quaternions4.x = (R[7] - R[5]) / s;
                    quaternions4.y = 0.25 * s;
                    quaternions4.z = (R[1] + R[3]) / s;
                    quaternions4.w = (R[2] + R[6]) / s;
                } else if (R[4] > R[8]) {
                    float_double s = 2.0 * sqrt(1.0 + R[4] - R[0] - R[8]);
                    quaternions4.x = (R[2] - R[6]) / s;
                    quaternions4.y = (R[1] + R[3]) / s;
                    quaternions4.z = 0.25 * s;
                    quaternions4.w = (R[5] + R[7]) / s;
                } else {
                    float_double s = 2.0 * sqrt(1.0 + R[8] - R[0] - R[4]);
                    quaternions4.x = (R[3] - R[1]) / s;
                    quaternions4.y = (R[2] + R[6]) / s;
                    quaternions4.z = (R[5] + R[7]) / s;
                    quaternions4.w = 0.25 * s;
                }
            }
        }
        //overload [] operator
        __host__ __device__ inline float_double operator[](int i){
            //return R[i/3][i%3];
            /*float_double r = kernel.quaternions4.x;
            float_double x = kernel.quaternions4.y;
            float_double y = kernel.quaternions4.z;
            float_double z = kernel.quaternions4.w;
            R[0][0] = 1 - 2*y*y - 2*z*z;
            R[0][1] = 2*x*y - 2*z*r;
            R[0][2] = 2*x*z + 2*y*r;
            R[1][0] = 2*x*y + 2*z*r;
            R[1][1] = 1 - 2*x*x - 2*z*z;
            R[1][2] = 2*y*z - 2*x*r;
            R[2][0] = 2*x*z - 2*y*r;
            R[2][1] = 2*y*z + 2*x*r;
            R[2][2] = 1 - 2*x*x - 2*y*y;*/
            if (i == 0)
            {
                //R[0][0]
                return 1 - 2*quaternions4.z*quaternions4.z - 2*quaternions4.w*quaternions4.w;
            }
            if (i == 1)
            {
                //R[0][1]
                return 2*quaternions4.y*quaternions4.z - 2*quaternions4.x*quaternions4.w;
            }
            if (i == 2)
            {
                //R[0][2]
                return 2*quaternions4.x*quaternions4.z + 2*quaternions4.y*quaternions4.w;
            }
            if (i == 3)
            {
                //R[1][0]
                return 2*quaternions4.y*quaternions4.z + 2*quaternions4.x*quaternions4.w;
            }
            if (i == 4)
            {
                //R[1][1]
                return 1 - 2*quaternions4.y*quaternions4.y - 2*quaternions4.w*quaternions4.w;
            }
            if (i == 5)
            {
                //R[1][2]
                return 2*quaternions4.z*quaternions4.w - 2*quaternions4.y*quaternions4.x;
            }
            if (i == 6)
            {
                //R[2][0]
                return 2*quaternions4.y*quaternions4.w - 2*quaternions4.z*quaternions4.x;
            }
            if (i == 7)
            {
                //R[2][1]
                return 2*quaternions4.z*quaternions4.w + 2*quaternions4.y*quaternions4.x;
            }
            if (i == 8)
            {
                //R[2][2]
                return 1 - 2*quaternions4.y*quaternions4.y - 2*quaternions4.z*quaternions4.z;
            }
            return 0;
        }
    //overload * operator
    __host__ __device__ inline rotation_matrix operator*(rotation_matrix R){
        float_double r1 = quaternions4.x;
        float_double x1 = quaternions4.y;
        float_double y1 = quaternions4.z;
        float_double z1 = quaternions4.w;
        float_double r2 = R.quaternions4.x;
        float_double x2 = R.quaternions4.y;
        float_double y2 = R.quaternions4.z;
        float_double z2 = R.quaternions4.w;
        rotation_matrix result;
        result.quaternions4.x = r1*r2 - x1*x2 - y1*y2 - z1*z2;
        result.quaternions4.y = r1*x2 + x1*r2 + y1*z2 - z1*y2;
        result.quaternions4.z = r1*y2 - x1*z2 + y1*r2 + z1*x2;
        result.quaternions4.w = r1*z2 + x1*y2 - y1*x2 + z1*r2;
        return result;
    }
    //overload * operator
    __host__ __device__ inline point3d operator*(point3d p){
        float_double r = quaternions4.x;
        float_double x = quaternions4.y;
        float_double y = quaternions4.z;
        float_double z = quaternions4.w;
        point3d result;
        result.x = (1-2*y*y-2*z*z)*p.x + (2*x*y-2*z*r)*p.y + (2*x*z+2*y*r)*p.z;
        result.y = (2*x*y+2*z*r)*p.x + (1-2*x*x-2*z*z)*p.y + (2*y*z-2*x*r)*p.z;
        result.z = (2*x*z-2*y*r)*p.x + (2*y*z+2*x*r)*p.y + (1-2*x*x-2*y*y)*p.z;
        return result;
    }
    //transpose
    __host__ __device__ inline rotation_matrix transpose(){
        float_double r = quaternions4.x;
        float_double x = quaternions4.y;
        float_double y = quaternions4.z;
        float_double z = quaternions4.w;
        float_double new_r = r;
        float_double new_x = -x;
        float_double new_y = -y;
        float_double new_z = -z;
        return rotation_matrix({new_r, new_x, new_y, new_z});
    }
    __host__ __device__ inline bool is_initialized(){
        return quaternions4.x != 0 || quaternions4.y != 0 || quaternions4.z != 0 || quaternions4.w != 0;
    }
};
// class __align__(16) rotation_matrix{
//     //store the matrix
//     public:
//     float_double R[9];
//     __host__ __device__ inline rotation_matrix(float_double4 quaternions4){
//         R[0] = 1 - 2*quaternions4.z*quaternions4.z - 2*quaternions4.w*quaternions4.w;
//         R[1] = 2*quaternions4.y*quaternions4.z - 2*quaternions4.x*quaternions4.w;
//         R[2] = 2*quaternions4.x*quaternions4.z + 2*quaternions4.y*quaternions4.w;
//         R[3] = 2*quaternions4.y*quaternions4.z + 2*quaternions4.x*quaternions4.w;
//         R[4] = 1 - 2*quaternions4.y*quaternions4.y - 2*quaternions4.w*quaternions4.w;
//         R[5] = 2*quaternions4.z*quaternions4.w - 2*quaternions4.y*quaternions4.x;
//         R[6] = 2*quaternions4.y*quaternions4.w - 2*quaternions4.z*quaternions4.x;
//         R[7] = 2*quaternions4.z*quaternions4.w + 2*quaternions4.y*quaternions4.x;
//         R[8] = 1 - 2*quaternions4.y*quaternions4.y - 2*quaternions4.z*quaternions4.z;
//     }
//     __host__ __device__ inline rotation_matrix(float_double ** R){
//         this->R[0] = R[0][0];
//         this->R[1] = R[0][1];
//         this->R[2] = R[0][2];
//         this->R[3] = R[1][0];
//         this->R[4] = R[1][1];
//         this->R[5] = R[1][2];
//         this->R[6] = R[2][0];
//         this->R[7] = R[2][1];
//         this->R[8] = R[2][2];
//     }
//     __host__ __device__ inline rotation_matrix(float_double*R){
//         this->R[0] = R[0];
//         this->R[1] = R[1];
//         this->R[2] = R[2];
//         this->R[3] = R[3];
//         this->R[4] = R[4];
//         this->R[5] = R[5];
//         this->R[6] = R[6];
//         this->R[7] = R[7];
//         this->R[8] = R[8];
//     }
//     __host__ __device__ inline rotation_matrix(){
//         R[0] = -2;
//         R[1] = -2;
//         R[2] = -2;
//         R[3] = -2;
//         R[4] = -2;
//         R[5] = -2;
//         R[6] = -2;
//         R[7] = -2;
//         R[8] = -2;
//     }
//     //overload [] operator
//     __host__ __device__ inline float_double operator[](int i){
//         return R[i];
//     }
//     //overload * operator
//     __host__ __device__ inline rotation_matrix operator*(rotation_matrix R_){
//         float_double result[9];
//         result[0] = R[0]*R_[0] + R[1]*R_[3] + R[2]*R_[6];
//         result[1] = R[0]*R_[1] + R[1]*R_[4] + R[2]*R_[7];
//         result[2] = R[0]*R_[2] + R[1]*R_[5] + R[2]*R_[8];
//         result[3] = R[3]*R_[0] + R[4]*R_[3] + R[5]*R_[6];
//         result[4] = R[3]*R_[1] + R[4]*R_[4] + R[5]*R_[7];
//         result[5] = R[3]*R_[2] + R[4]*R_[5] + R[5]*R_[8];
//         result[6] = R[6]*R_[0] + R[7]*R_[3] + R[8]*R_[6];
//         result[7] = R[6]*R_[1] + R[7]*R_[4] + R[8]*R_[7];
//         result[8] = R[6]*R_[2] + R[7]*R_[5] + R[8]*R_[8];
//         return rotation_matrix(result);
//     }
//     //overload * operator
//     __host__ __device__ inline point3d operator*(point3d p){
//         point3d result;
//         result.x = R[0]*p.x + R[1]*p.y + R[2]*p.z;
//         result.y = R[3]*p.x + R[4]*p.y + R[5]*p.z;
//         result.z = R[6]*p.x + R[7]*p.y + R[8]*p.z;
//         return result;
//     }
//     //transpose
//     __host__ __device__ inline rotation_matrix transpose(){
//         float_double result[9];
//         result[0] = R[0];
//         result[1] = R[3];
//         result[2] = R[6];
//         result[3] = R[1];
//         result[4] = R[4];
//         result[5] = R[7];
//         result[6] = R[2];
//         result[7] = R[5];
//         result[8] = R[8];
//         return rotation_matrix(result);
//     }
//     __host__ __device__ inline bool is_initialized(){
//         return R[0] != -2;
//     }
// };

class spherical_harmonic_color{
    public:
    point3d sh[16];
    
    __host__ __device__ inline spherical_harmonic_color(){
        for (int i = 0; i < 16; i++)
        {
            sh[i] = point3d(0, 0, 0);
        }
    }

    __host__ __device__ inline spherical_harmonic_color(float_double *sh){
        for (int i = 0; i < 16; i++)
        {
            this->sh[i] = point3d(sh[i*3], sh[i*3+1], sh[i*3+2]);
        }
    }
    __host__ __device__ inline spherical_harmonic_color(point3d *sh){
        for (int i = 0; i < 16; i++)
        {
            this->sh[i] = sh[i];
        }
    }

    __host__ __device__ inline void to_rgb(point3d &result, point3d dir, unsigned char deg){
        result = SH_C0 * sh[0];
        //printf("sh[0] %f %f %f\n", sh[0].x, sh[0].y, sh[0].z);
        //printf("result %f %f %f\n", result.x, result.y, result.z);
        //printf("deg %d\n", deg);
        if (deg > 0)
        {
            float x = dir.x, y = dir.y, z = dir.z;
            result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
    
            if (deg > 1)
            {
                float xx = x * x, yy = y * y, zz = z * z;
                float xy = x * y, yz = y * z, xz = x * z;
                result = result +
                    SH_C2_0 * xy * sh[4] +
                    SH_C2_1 * yz * sh[5] +
                    SH_C2_2 * (2.0f * zz - xx - yy) * sh[6] +
                    SH_C2_3 * xz * sh[7] +
                    SH_C2_4 * (xx - yy) * sh[8];
    
                if (deg > 2)
                {
                    result = result +
                        SH_C3_0 * y * (3.0f * xx - yy) * sh[9] +
                        SH_C3_1 * xy * z * sh[10] +
                        SH_C3_2 * y * (4.0f * zz - xx - yy) * sh[11] +
                        SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
                        SH_C3_4 * x * (4.0f * zz - xx - yy) * sh[13] +
                        SH_C3_5 * z * (xx - yy) * sh[14] +
                        SH_C3_6 * x * (xx - 3.0f * yy) * sh[15];
                }
            }
        }
        result = result+0.5;
        //result = result +(float_double)0.5;
        result.x = min(1., max(0., result.x));
        result.y = min(1., max(0., result.y));
        result.z = min(1., max(0., result.z));
    }
};

class colored_gaussian_kernel_1D{
    public:
    point3d color;
    float_double mu;
    float_double sigma;
    float_double log_weight;

    __host__ __device__ inline colored_gaussian_kernel_1D(point3d color, float_double sigma, float_double log_weight, float_double mu){
        this->color = color;
        this->sigma = sigma;
        this->log_weight = log_weight;
        this->mu = mu;
    }
    __host__ __device__ inline colored_gaussian_kernel_1D(){
        this->color = point3d(0, 0, 0);
        this->sigma = 0;
        this->log_weight = 0;
        this->mu = 0;
    }
    // override operator()
    __host__ __device__ inline float_double operator()(float_double x){
        float_double result = (x - mu) / sigma;
        //std::cout << "result() ,sigma " << result << " " << sigma << std::endl;
        return exp(-0.5 * result * result + log_weight);
    }

    __host__ __device__ inline float_double start(bool debug = false){
        //return mu - 5 * sigma;
        //first antecedent of exp(negligeable_val_when_exp)
        /*if(2 * (negligeable_val_when_exp - log_weight)*sigma*sigma>0){
            return mu-sqrt(2 * (negligeable_val_when_exp - log_weight)*sigma*sigma);
        }
        else{
            return mu;
        }*/
        /*offset = 2 * (negligeable_val_when_exp - log_weight)*sigma*sigma;
        if(offset>0){
            return mu-sqrt(offset);
        }
        else{
            return mu;
        }*/
        //instead of looking at the value, look at the integral
        float_double c = 1/(2*sigma*sigma);
        float_double sqr_c = 1/(sqrt(2)*sigma);
        float_double v = 1-(exp(negligeable_val_when_exp-log_weight)*sqr_c/(sqrt(M_PI)));
        float_double ofset;
        if(debug){
            printf("sqr_c %f,log_weight %f (negligeable_val_when_exp-log_weight)*sqr_c %f, exp(negligeable_val_when_exp-log_weight)*sqr_c %f, v %f, erfinvf(v) %f\n", sqr_c,log_weight, (negligeable_val_when_exp-log_weight)*sqr_c, exp(negligeable_val_when_exp-log_weight)*sqr_c, v, erfinvf(v));
        }
        if(v>0){
            ofset = erfinvf(v)/sqr_c;
        }
        else{
            ofset = 0;
        }
        return mu-ofset;
    }
    __host__ __device__ inline float_double end(){
        //return mu + 5 * sigma;
        //second antecedent of exp(negligeable_val_when_exp)
        /*if(2 * (negligeable_val_when_exp - log_weight)*sigma*sigma>0){
            return mu +sqrt(2 * (negligeable_val_when_exp - log_weight)*sigma*sigma);
        }
        else{
            return mu;
        }*/
        /*if(offset>0){
            return mu+sqrt(offset);
        }
        else{
            return mu;
        }*/
        //instead of looking at the value, look at the integral
        float_double c = 1/(2*sigma*sigma);
        float_double sqr_c = 1/(sqrt(2)*sigma);
        float_double v = 1-(exp(negligeable_val_when_exp-log_weight)*sqr_c/(sqrt(M_PI)));
        float_double ofset;
        if(v>0){
            ofset = erfinvf(v)/sqr_c;
        }
        else{
            ofset = 0;
        }
        return mu+ofset;
    }

    __host__ __device__ inline float_double get_integral(float_double a, float_double b){
        if (a > b)
        {
            return 0;
        }
        /*float_double result = 0;
        for (int i = 0; i < k_steps; i++)
        {
            //std::cout << "i "<< i <<" result " << result << std::endl;
            float_double x = a + (b - a) * i / k_steps;
            result += (*this)(x);
        }*/
        #ifdef rescaling_integral_1
            if(a == this->start()){
                a = 0;
            }
            if(b == this->end()){
                b = INFINITY;
            }
        #endif
        //use the error function to compute the integral
        float_double c = 1/(2*sigma*sigma);
        float_double sqr_c = 1/(sqrt(2)*sigma);
        //float_double result = exp(log_weight) * (sqrt(M_PI)/2) * sigma*(erf((b - mu) / (sqrt(2) * sigma)) - erf((a - mu) / (sqrt(2) * sigma)));
        float_double result = (exp(log_weight) * sqrt(M_PI)/(2*sqr_c)) * (erf((b - mu) * sqr_c) - erf((a - mu) * sqr_c));
        //printf("a-mu %f, b-mu %f, erf((b - mu) * sqr_c) %f, erf((a - mu) * sqr_c) %f, (erf((b - mu) * sqr_c) - erf((a - mu) * sqr_c)) %f , (exp(log_weight) * sqrt(M_PI)/(2*sqr_c))%f , result %f\n", a-mu, b-mu, erf((b - mu) * sqr_c), erf((a - mu) * sqr_c), (erf((b - mu) * sqr_c) - erf((a - mu) * sqr_c)), (exp(log_weight) * sqrt(M_PI)/(2*sqr_c)), result);
        if(log_weight > 0){
            //printf("sigma %f, mu %f, log_weight %f, a %f, b %f, result %f\n", sigma, mu, log_weight, a, b, result);
        }
        //std::cout << "result " << result << std::endl;
        /*if(result * (b - a) / k_steps > 1){
            std::cout << "result " << result << " (b - a) / k_steps " << (b - a) / k_steps << std::endl;
        }*/
        //return min(result * (b - a) / k_steps, 1.0f);
        //return min(result, 0.99f);
        #ifdef rescaling_integral_2
            float_double my_start = this->start();
            float_double my_end = this->end();
            if(a<my_start || a>my_end || b<my_start || b>my_end){
                //printf("out\n");
                return result;
            }
            float_double ofset = 2 * (negligeable_val_when_exp - log_weight)*sigma*sigma;
            float_double total_integral = (exp(log_weight) * sqrt(M_PI)/(sqr_c));
            if(ofset<=0)
            {
                return total_integral;
            }
            float_double v = total_integral/((exp(log_weight) * sqrt(M_PI)/(2*sqr_c)) * (erf((my_end - mu) * sqr_c) - erf((my_start - mu) * sqr_c)));
            //float_double v = (exp(log_weight) * sqrt(M_PI)/(sqr_c))/((exp(log_weight) * sqrt(M_PI)/(2*sqr_c)) * (erf((my_end - mu) * sqr_c) - erf((my_start - mu) * sqr_c)));
            //float_double v = 2/((erf((my_end - mu) * sqr_c) - erf((my_start - mu) * sqr_c)));
            //float_double v = 2/((erf((my_end - mu) * sqr_c) - erf((my_start - mu) * sqr_c)));
            //printf("v %f, ofset %f\n", v, ofset);
            //printf("result %f\n", result);
            result*= v;
        #endif
        return result;
    }

    __host__ __device__ inline float_double get_integral(){
        return get_integral(start(), end());
    }

};

class visual_gaussian_kernel{
    public:
    gaussian_kernel2_3D kernel;
    spherical_harmonic_color color;
    #ifdef can_buffer_sigma
        float_double6 sigma_buffer;
    #endif
    #ifdef can_buffer_color
        point3d rgb_buffer;
    #endif
    __host__ __device__ inline visual_gaussian_kernel(gaussian_kernel2_3D kernel, float_double *sh){
        this->kernel = kernel;
        this->color = spherical_harmonic_color(sh);
        #ifdef can_buffer_sigma
            this->sigma_buffer[0] = -1;
        #endif
        #ifdef can_buffer_color
            this->rgb_buffer = point3d(-1, -1, -1);
        #endif
    }
    __host__ __device__ inline visual_gaussian_kernel(gaussian_kernel2_3D kernel, point3d *sh){
        this->kernel = kernel;
        this->color = spherical_harmonic_color(sh);
        #ifdef can_buffer_sigma
            this->sigma_buffer[0] = -1;
        #endif
        #ifdef can_buffer_color
            this->rgb_buffer = point3d(-1, -1, -1);
        #endif
    }
    __host__ __device__ inline visual_gaussian_kernel(gaussian_kernel2_3D kernel, spherical_harmonic_color color){
        this->kernel = kernel;
        this->color = color;
        #ifdef can_buffer_sigma
            this->sigma_buffer[0] = -1;
        #endif
        #ifdef can_buffer_color
            this->rgb_buffer = point3d(-1, -1, -1);
        #endif
    }
    __host__ __device__ inline visual_gaussian_kernel(){
        this->kernel = gaussian_kernel2_3D();
        this->color = spherical_harmonic_color();
        #ifdef can_buffer_sigma
            this->sigma_buffer[0] = -1;
        #endif
        #ifdef can_buffer_color
            this->rgb_buffer = point3d(-1, -1, -1);
        #endif
    }
    __host__ __device__ inline point3d get_rgb(point3d src){
        #ifdef can_buffer_color
            if (rgb_buffer.x == -1){
        #endif
                point3d result = point3d(0, 0, 0);
                //color.to_rgb(result, dir, 2);
                //rgb_buffer = result;
                point3d d = kernel.mu-src;
                float_double n = sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
                d.x = d.x/n;
                d.y = d.y/n;
                d.z = d.z/n;
                color.to_rgb(result, d, 0);
                #ifdef can_buffer_color
                    rgb_buffer = result;
                #endif
                //rgb_buffer = result;
                return result;
        #ifdef can_buffer_color
            }
            else{
                return rgb_buffer;
            }
        #endif
    }
    #ifdef can_buffer_color
        __host__ __device__ inline void reset_rgb_buffer(){
            rgb_buffer = point3d(-1, -1, -1);
        }
    #endif
    // override operator()
    __host__ __device__ float_double operator()(point3d x){
        float_double result = kernel(x);
        if (result  < negligeable_val_when_exp)
        {
            return 0;
        }
        return exp(result);
    }
    __host__ __device__ inline float_double6 get_reoriented_sigma_inv(rotation_matrix dir){
        #ifdef can_buffer_sigma
            if (sigma_buffer[0] != -1){
                return sigma_buffer;
            }
        #endif
        rotation_matrix new_r = rotation_matrix(kernel.quaternions4)*(dir.transpose());
        float_double M[3][3];
        float_double6 precomputed_reoriented_sigma_inv;
        M[0][0] = (1/kernel.scales3.x) * new_r[0];
        M[1][0] = (1/kernel.scales3.y) * new_r[3];
        M[2][0] = (1/kernel.scales3.z) * new_r[6];
        precomputed_reoriented_sigma_inv[0] = M[0][0]*M[0][0] + M[1][0]*M[1][0] + M[2][0]*M[2][0];
        M[0][1] = (1/kernel.scales3.x) * new_r[1];
        M[1][1] = (1/kernel.scales3.y) * new_r[4];
        M[2][1] = (1/kernel.scales3.z) * new_r[7];
        precomputed_reoriented_sigma_inv[1] = M[0][0]*M[0][1] + M[1][0]*M[1][1] + M[2][0]*M[2][1];
        M[0][2] = (1/kernel.scales3.x) * new_r[2];
        M[1][2] = (1/kernel.scales3.y) * new_r[5];
        M[2][2] = (1/kernel.scales3.z) * new_r[8];
        precomputed_reoriented_sigma_inv[2] = M[0][0]*M[0][2] + M[1][0]*M[1][2] + M[2][0]*M[2][2];
        precomputed_reoriented_sigma_inv[3] = M[0][1]*M[0][1] + M[1][1]*M[1][1] + M[2][1]*M[2][1];
        precomputed_reoriented_sigma_inv[4] = M[0][1]*M[0][2] + M[1][1]*M[1][2] + M[2][1]*M[2][2];
        precomputed_reoriented_sigma_inv[5] = M[0][2]*M[0][2] + M[1][2]*M[1][2] + M[2][2]*M[2][2];
        #ifdef can_buffer_sigma
            sigma_buffer = precomputed_reoriented_sigma_inv;
        #endif
        return precomputed_reoriented_sigma_inv;
    }
    __host__ __device__ inline float_double6 get_reoriented_sigma(rotation_matrix dir){
        #ifdef can_buffer_sigma
            if (sigma_buffer[0] != -1){
                return sigma_buffer;
            }
        #endif
        rotation_matrix new_r = rotation_matrix(kernel.quaternions4)*(dir.transpose());
        float_double M[3][3];
        float_double6 precomputed_reoriented_sigma;
        M[0][0] = kernel.scales3.x * new_r[0];
        M[1][0] = kernel.scales3.y * new_r[3];
        M[2][0] = kernel.scales3.z * new_r[6];
        precomputed_reoriented_sigma[0] = M[0][0]*M[0][0] + M[1][0]*M[1][0] + M[2][0]*M[2][0];
        M[0][1] = kernel.scales3.x * new_r[1];
        M[1][1] = kernel.scales3.y * new_r[4];
        M[2][1] = kernel.scales3.z * new_r[7];
        precomputed_reoriented_sigma[1] = M[0][0]*M[0][1] + M[1][0]*M[1][1] + M[2][0]*M[2][1];
        M[0][2] = kernel.scales3.x * new_r[2];
        M[1][2] = kernel.scales3.y * new_r[5];
        M[2][2] = kernel.scales3.z * new_r[8];
        precomputed_reoriented_sigma[2] = M[0][0]*M[0][2] + M[1][0]*M[1][2] + M[2][0]*M[2][2];
        precomputed_reoriented_sigma[3] = M[0][1]*M[0][1] + M[1][1]*M[1][1] + M[2][1]*M[2][1];
        precomputed_reoriented_sigma[4] = M[0][1]*M[0][2] + M[1][1]*M[1][2] + M[2][1]*M[2][2];
        precomputed_reoriented_sigma[5] = M[0][2]*M[0][2] + M[1][2]*M[1][2] + M[2][2]*M[2][2];
        #ifdef can_buffer_sigma
            sigma_buffer = precomputed_reoriented_sigma;
        #endif
        return precomputed_reoriented_sigma;
    }
    #ifdef can_buffer_sigma
        __host__ __device__ inline void reset_sigma_buffer(){
            sigma_buffer[0] = -1;
        }
    #endif
    __host__ __device__ inline colored_gaussian_kernel_1D get_1D_kernel(rotation_matrix dir, point3d src,bool debug = false){
        //to do it corectely later
        point3d color = (*this).get_rgb(src);
        float_double6 precomputed_reoriented_sigma_inv = get_reoriented_sigma_inv(dir);
        //std::cout << "original sigma scales :" << kernel.scales3.x << " " << kernel.scales3.y << " " << kernel.scales3.z << " quaternions " << kernel.quaternions4.x << " " << kernel.quaternions4.y << " " << kernel.quaternions4.z << " " << kernel.quaternions4.w << std::endl;
        //std::cout << "reoriented sigma " << precomputed_reoriented_sigma_inv[0] << " " << precomputed_reoriented_sigma_inv[1] << " " << precomputed_reoriented_sigma_inv[2] << " " << precomputed_reoriented_sigma_inv[3] << " " << precomputed_reoriented_sigma_inv[4] << " " << precomputed_reoriented_sigma_inv[5] << std::endl;
        float_double sigma_ = 1/(2*sqrt(precomputed_reoriented_sigma_inv[5]));
        //std::cout << "original log_weight " << kernel.log_weight << std::endl;
        float_double log_weight_ = kernel.log_weight;
        point3d shifted_mu = kernel.mu;
        shifted_mu.x = shifted_mu.x - src.x;
        shifted_mu.y = shifted_mu.y - src.y;
        shifted_mu.z = shifted_mu.z - src.z;
        shifted_mu = dir*shifted_mu;
        //printf("shifted_mu %f %f %f\n", shifted_mu.x, shifted_mu.y, shifted_mu.z);
    //    log_weight_ -= precomputed_reoriented_sigma_inv[0]*shifted_mu.x*shifted_mu.x;
        //std::cout << "log_weight_ a " << log_weight_ << std::endl;
    //    log_weight_ -= precomputed_reoriented_sigma_inv[3]*shifted_mu.y*shifted_mu.y;
        //std::cout << "log_weight_ b " << log_weight_ << std::endl;
    //    log_weight_ -= 2*precomputed_reoriented_sigma_inv[1]*shifted_mu.x*shifted_mu.y;
        //std::cout << "log_weight_ c " << log_weight_ << std::endl;
        //log_weight_ -= 2*precomputed_reoriented_sigma_inv[2]*shifted_mu.x*shifted_mu.z;
        //std::cout << "log_weight_ d " << log_weight_ << std::endl;
        //log_weight_ -= 2*precomputed_reoriented_sigma_inv[4]*shifted_mu.y*shifted_mu.z;
        //std::cout << "log_weight_ e " << log_weight_ << std::endl;
    //    log_weight_ += (precomputed_reoriented_sigma_inv[2]*precomputed_reoriented_sigma_inv[2]/(precomputed_reoriented_sigma_inv[5]))*shifted_mu.x*shifted_mu.x;
        //std::cout << "log_weight_ f " << log_weight_ << std::endl;
    //    log_weight_ += (precomputed_reoriented_sigma_inv[4]*precomputed_reoriented_sigma_inv[4]/(precomputed_reoriented_sigma_inv[5]))*shifted_mu.y*shifted_mu.y;
        //std::cout << "log_weight_ g " << log_weight_ << std::endl;
    //    log_weight_ += 2*(precomputed_reoriented_sigma_inv[2]*precomputed_reoriented_sigma_inv[4]/(precomputed_reoriented_sigma_inv[5]))*shifted_mu.x*shifted_mu.y;
        //std::cout << "log_weight_ h " << log_weight_ << std::endl;
        //log_weight_ += 2*precomputed_reoriented_sigma_inv[2]*shifted_mu.x*shifted_mu.z;
        //std::cout << "log_weight_ i " << log_weight_ << std::endl;
        //log_weight_ += 2*precomputed_reoriented_sigma_inv[4]*shifted_mu.y*shifted_mu.z;
        //std::cout << "log_weight_ " << log_weight_ << std::endl;
        //std::cout << "sigma_ " << sigma_ << std::endl;

        //compute log_weight_ an other way
        float_double6 precomputed_reoriented_sigma = get_reoriented_sigma(dir);
        float_double inv_det = 1/(precomputed_reoriented_sigma[0]*precomputed_reoriented_sigma[3] - precomputed_reoriented_sigma[1]*precomputed_reoriented_sigma[1]);
        float_double xx = precomputed_reoriented_sigma[3]*inv_det;
        float_double yy = precomputed_reoriented_sigma[0]*inv_det;
        float_double xy = -precomputed_reoriented_sigma[1]*inv_det;
        log_weight_ = kernel.log_weight - 0.5*(xx*shifted_mu.x*shifted_mu.x + yy*shifted_mu.y*shifted_mu.y + 2*xy*shifted_mu.x*shifted_mu.y);





        float_double mu_ = shifted_mu.z +(shifted_mu.x*precomputed_reoriented_sigma_inv[2] + shifted_mu.y*precomputed_reoriented_sigma_inv[4])/precomputed_reoriented_sigma_inv[5];
        //std::cout << "mu_ " << mu_ << std::endl;
        //free(precomputed_reoriented_sigma_inv);
        return colored_gaussian_kernel_1D(color, sigma_, log_weight_,mu_);
    }
};

class index_start_end{
    public:
    array_indexes_type index;
    bool start;
    float_double value;
    __host__ __device__ inline index_start_end(int index, bool start, float_double value){
        this->index = index;
        this->start = start;
        this->value = value;
    }
    __host__ __device__ inline index_start_end(){
        this->index = 0;
        this->start = false;
        this->value = 0;
    }

    //overload comparison operators

    __host__ __device__ inline bool operator<(const index_start_end &other) const{
        return value < other.value || (value == other.value && start > other.start) || (value == other.value && start == other.start && index < other.index);
    }
    __host__ __device__ inline bool operator>(const index_start_end &other) const{
        return value > other.value || (value == other.value && start < other.start) || (value == other.value && start == other.start && index > other.index);
    }
    __host__ __device__ inline bool operator<=(const index_start_end &other) const{
        return value < other.value || (value == other.value && start >= other.start) || (value == other.value && start == other.start && index <= other.index);
    }
    __host__ __device__ inline bool operator>=(const index_start_end &other) const{
        return value > other.value || (value == other.value && start <= other.start) || (value == other.value && start == other.start && index >= other.index);
    }
    __host__ __device__ inline bool operator==(const index_start_end &other) const{
        return value == other.value && index == other.index && start == other.start;
    }
    __host__ __device__ inline bool operator!=(const index_start_end &other) const{
        return value != other.value || index != other.index || start != other.start;
    }
};

class index_value{
    public:
    array_indexes_type index;
    float_double value;
    __host__ __device__ inline index_value(array_indexes_type index, float_double value){
        this->index = index;
        this->value = value;
    }
    __host__ __device__ inline index_value(){
        this->index = 0;
        this->value = 0;
    }
    __host__ __device__ inline bool operator<(const index_value &other) const{
        return value < other.value;
    }
    __host__ __device__ inline bool operator>(const index_value &other) const{
        return value > other.value;
    }
    __host__ __device__ inline bool operator<=(const index_value &other) const{
        return value <= other.value;
    }
    __host__ __device__ inline bool operator>=(const index_value &other) const{
        return value >= other.value;
    }
    __host__ __device__ inline bool operator==(const index_value &other) const{
        return value == other.value && index == other.index;
    }
    __host__ __device__ inline bool operator!=(const index_value &other) const{
        return value != other.value || index != other.index;
    }

};