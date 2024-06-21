__host__ __device__ inline float_double sign(float_double x){
    if(x>0){
        return 1;
    }
    if(x<0){
        return -1;
    }
    return 0;
}
__host__ __device__ inline void ray_info_(point3d src_, point3d& src, rotation_matrix dir_, rotation_matrix& dir, float_double pixel_x, float_double pixel_y, int width, int height){
    if(orthogonal){
        float_double x = ((pixel_x/(width-1))-0.5)*x_scaling;
        float_double y = ((pixel_y/(height-1))-0.5)*y_scaling;
        src = src_+((dir_.transpose())*(point3d(x, y, 0)));
        dir = dir_;
    }
    else{
        float_double theta_y;
        float_double theta_x;
        if(angle_1){
            //the new source is the original one
            src = src_;
            //the new direction is the original one rotated according to two angles
            
            
            
            /*float_double theta_x = ((pixel_x/(width-1))-0.5)*(fov/180.)*M_PI;
            float_double theta_y = ((pixel_y/(height-1))-0.5)*(fov/180.)*M_PI;*/

            float_double theta_x_ = ((pixel_x/(width-1))-0.5)*(fov/180.)*M_PI;
            float_double theta_y_ = ((pixel_y/(height-1))-0.5)*(fov/180.)*M_PI;

            theta_y = -theta_y_;
            theta_x = asinf(-cosf(theta_y_)*sinf(theta_x_));
        }
        else{
            float_double l  = screen_dist*tanf((2*fov/180.)*M_PI/2);
            float_double x = ((pixel_x/(width-1))-0.5)*l;
            float_double y = ((pixel_y/(height-1))-0.5)*l;
            float_double z = screen_dist;
            src = src_+((dir_.transpose())*(point3d(x, y, z)));

            float_double theta_x_ = atanf(x/screen_dist);
            float_double theta_y_ = atanf(y/screen_dist);
            theta_y = -theta_y_;
            theta_x = asinf(-cosf(theta_y_)*sinf(theta_x_));            
        }        
        float cosx,cosy,sinx,siny;
        cosx = cosf(theta_x);
        cosy = cosf(theta_y);
        sinx = sinf(theta_x);
        siny = sinf(theta_y);
        float_double rotationxy[9];
        rotationxy[0] = cosx;
        rotationxy[1] = 0;
        rotationxy[2] = sinx;
        rotationxy[3] = -siny*sinx;
        rotationxy[4] = cosy;
        rotationxy[5] = siny*cosx;
        rotationxy[6] = -cosy*sinx;
        rotationxy[7] = -siny;
        rotationxy[8] = cosy*cosx;
        dir = rotation_matrix(rotationxy)*dir_;
        //free(rotationxy);
    }   
}
class ray_info_buffered{
    public:
    point3d src;
    rotation_matrix dir;
    __host__ __device__ ray_info_buffered(point3d src_, rotation_matrix dir_){
        src = src_;
        dir = dir_;
    }
    __host__ __device__ ray_info_buffered(){
        src = point3d(0,0,0);
        dir = rotation_matrix();
    }
    __host__ __device__ ray_info_buffered(point3d src_, rotation_matrix dir_, float_double pixel_x, float_double pixel_y, int width, int height){
        ray_info_(src_, src, dir_, dir, pixel_x, pixel_y, width, height);
    }
    __host__ __device__ void get_info(point3d src_, rotation_matrix dir_, float_double pixel_x, float_double pixel_y, int width, int height){
        if (!dir.is_initialized()){
            ray_info_(src_, src, dir_, dir, pixel_x, pixel_y, width, height);
        }
    }
};
__host__ __device__ inline void ray_info_z_dir_src(point3d src_, point3d& src, rotation_matrix dir_,point3d& z_dir, float_double pixel_x, float_double pixel_y, int width, int height){
    if(orthogonal){
        float_double x = ((pixel_x/(width-1))-0.5)*x_scaling;
        float_double y = ((pixel_y/(height-1))-0.5)*y_scaling;
        src = src_+((dir_.transpose())*(point3d(x, y, 0)));
        z_dir.x = dir_[6];
        z_dir.y = dir_[7];
        z_dir.z = dir_[8];
    }
    else{
        float_double theta_y;
        float_double theta_x;
        if(angle_1){
            //the new source is the original one
            src = src_;
            //the new direction is the original one rotated according to two angles
            
            
            
            /*float_double theta_x = ((pixel_x/(width-1))-0.5)*(fov/180.)*M_PI;
            float_double theta_y = ((pixel_y/(height-1))-0.5)*(fov/180.)*M_PI;*/

            float_double theta_x_ = ((pixel_x/(width-1))-0.5)*(fov/180.)*M_PI;
            float_double theta_y_ = ((pixel_y/(height-1))-0.5)*(fov/180.)*M_PI;

            theta_y = -theta_y_;
            theta_x = asinf(-cosf(theta_y_)*sinf(theta_x_));
        }
        else{
            float_double l  = screen_dist*tanf((2*fov/180.)*M_PI/2);
            float_double x = ((pixel_x/(width-1))-0.5)*l;
            float_double y = ((pixel_y/(height-1))-0.5)*l;
            float_double z = screen_dist;
            src = src_+((dir_.transpose())*(point3d(x, y, z)));

            float_double theta_x_ = atanf(x/screen_dist);
            float_double theta_y_ = atanf(y/screen_dist);
            theta_y = -theta_y_;
            theta_x = asinf(-cosf(theta_y_)*sinf(theta_x_));            
        }
        /*point3d dir_z = point3d(-cosf(theta_y)*sinf(theta_x),-sinf(theta_y),cosf(theta_y)*cosf(theta_x));
        point3d s = (point3d(dir_[0]+dir_[1]+dir_[2],dir_[3]+dir_[4]+dir_[5],dir_[6]+dir_[7]+dir_[8]));
        z_dir->x = dir_z.x * s.x;
        z_dir->y = dir_z.y * s.y;
        z_dir->z = dir_z.z * s.z;*/
        /*z_dir->x = (dir_[0]+dir_[1]+dir_[2]) * (-cosf(theta_y)*sinf(theta_x));
        z_dir->y = (dir_[3]+dir_[4]+dir_[5]) * (-sinf(theta_y));
        z_dir->z = (dir_[6]+dir_[7]+dir_[8]) * (cosf(theta_y)*cosf(theta_x));*/
        /*float_double * rotationxy = new float_double[9];
        rotationxy[0] = cosf(theta_x);
        rotationxy[1] = 0;
        rotationxy[2] = sinf(theta_x);
        rotationxy[3] = -sinf(theta_y)*sinf(theta_x);
        rotationxy[4] = cosf(theta_y);
        rotationxy[5] = sinf(theta_y)*cosf(theta_x);
        rotationxy[6] = -cosf(theta_y)*sinf(theta_x);
        rotationxy[7] = -sinf(theta_y);
        rotationxy[8] = cosf(theta_y)*cosf(theta_x);
        rotation_matrix dir = rotation_matrix(rotationxy)*dir_;
        free(rotationxy);
        z_dir->x = dir[6];
        z_dir->y = dir[7];
        z_dir->z = dir[8];*/
        float cosx,cosy,sinx,siny;
        cosx = cosf(theta_x);
        cosy = cosf(theta_y);
        sinx = sinf(theta_x);
        siny = sinf(theta_y);
        //point3d dir_z = point3d(-cosf(theta_y)*sinf(theta_x),-sinf(theta_y),cosf(theta_y)*cosf(theta_x));
        point3d dir_z = point3d(-cosy*sinx,-siny,cosy*cosx);
        z_dir = (dir_.transpose()*dir_z);
    }   
}
__host__ __device__ inline void get_point_nearest_screen_cord_(point3d src_, rotation_matrix dir_,float_double& pixel_x, float_double& pixel_y, point3d point,int width, int height){
    if(orthogonal){
        point3d x_vec;
        point3d y_vec;

        x_vec = dir_.transpose()*point3d(1,0,0);
        y_vec = dir_.transpose()*point3d(0,1,0);
        point3d point_vec = point-src_;
        pixel_x = (((point_vec*x_vec)/x_scaling)+0.5)*(width-1);
        pixel_y = (((point_vec*y_vec)/y_scaling)+0.5)*(height-1);
    }
    else{
        point3d d = point-src_;
        d = dir_ * d;
        d = d/d.norm();
        float siny = -d.y;
        float_double theta_y = asinf(siny);
        float cosy = cosf(theta_y);
        float sinx = -d.x/cosy;
        float_double theta_x = asinf(sinx);

        if(angle_1){
            float_double theta_y_ = -theta_y;
            float_double sinx_ = sinf(theta_x)/cosy;
            float_double theta_x_ = asinf(sinx_);
            //corect the signs
            theta_y_ = abs(theta_y_) * sign(d.y);
            theta_x_ = abs(theta_x_) * sign(d.x);

            pixel_x = ((theta_x_/((fov/180.)*M_PI))+0.5)*(width-1);
            pixel_y = ((theta_y_/((fov/180.)*M_PI))+0.5)*(height-1);
            
            
        }
        else{
            /*float_double l  = screen_dist*tanf((2*fov/180.)*M_PI/2);
            float_double x = ((pixel_x/(width-1))-0.5)*l;
            float_double y = ((pixel_y/(height-1))-0.5)*l;
            float_double z = screen_dist;
            src = src_+((dir_.transpose())*(point3d(x, y, z)));

            float_double theta_x_ = atanf(x/screen_dist);
            float_double theta_y_ = atanf(y/screen_dist);
            theta_y = -theta_y_;
            theta_x = asinf(-cosf(theta_y_)*sinf(theta_x_));*/
            float_double l  = screen_dist*tanf((2*fov/180.)*M_PI/2);
            float_double theta_y_ = -theta_y;
            float_double sinx_ = sinf(theta_x)/cosy;
            float_double theta_x_ = asinf(sinx_);
            //corect the signs
            theta_y_ = abs(theta_y_) * sign(d.y);
            theta_x_ = abs(theta_x_) * sign(d.x);

            float_double x = tanf(theta_x_)*screen_dist;
            float_double y = tanf(theta_y_)*screen_dist;
            //corect the signs
            x = abs(x) * sign(d.x);
            y = abs(y) * sign(d.y);
            pixel_x = ((x/l)+0.5)*(width-1);
            pixel_y = ((y/l)+0.5)*(height-1);

        }
    }
}