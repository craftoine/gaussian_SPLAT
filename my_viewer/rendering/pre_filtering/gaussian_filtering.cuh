
__host__ __device__ inline void get_bounding_rec_(point3d src_, rotation_matrix dir_, float_double& min_x, float_double& max_x, float_double& min_y, float_double& max_y, visual_gaussian_kernel g, int width, int height){
    float_double pixel_center_x, pixel_center_y;
    get_point_nearest_screen_cord_(src_, dir_, pixel_center_x, pixel_center_y, g.kernel.mu, width, height);
    point3d src;
    rotation_matrix dir;
    ray_info_(src_, src, dir_, dir, pixel_center_x, pixel_center_y, width, height);
    float_double6 precomputed_reoriented_sigma  = g.get_reoriented_sigma(dir);
    float_double6 precomputed_reoriented_sigma_inv  = g.get_reoriented_sigma_inv(dir);
    /*float_double sigma_x = precomputed_reoriented_sigma[0]-precomputed_reoriented_sigma[2]*precomputed_reoriented_sigma[2]/precomputed_reoriented_sigma[5];
    float_double sigma_y = precomputed_reoriented_sigma[3]-precomputed_reoriented_sigma[4]*precomputed_reoriented_sigma[4]/precomputed_reoriented_sigma[5];
    float_double sigma_xy = precomputed_reoriented_sigma[1]-precomputed_reoriented_sigma[2]*precomputed_reoriented_sigma[4]/precomputed_reoriented_sigma[5];
    */
    float_double inv_det = 1/(precomputed_reoriented_sigma[0]*precomputed_reoriented_sigma[3]-precomputed_reoriented_sigma[1]*precomputed_reoriented_sigma[1]);
    float_double sigma_x = precomputed_reoriented_sigma[3]*inv_det;
    float_double sigma_y = precomputed_reoriented_sigma[0]*inv_det;
    float_double sigma_xy = -precomputed_reoriented_sigma[1]*inv_det;

    float_double new_log_weight = g.kernel.log_weight+log(sqrt(M_PI*2/precomputed_reoriented_sigma_inv[5]));
    //float_double new_log_weight = g.kernel.log_weight;
    float_double det = sigma_x*sigma_y-sigma_xy*sigma_xy;
    float_double mid = (sigma_x+sigma_y)/2;
    float_double lambda1 = mid+sqrt(mid*mid-det);
    float_double lambda2 = mid-sqrt(mid*mid-det);
    float_double lambda_min = min(lambda1, lambda2);
    
    float_double eigen_vector1_x = 1.;
    float_double eigen_vector1_y = (lambda1-sigma_x)/sigma_xy;
    float_double eigen_vector2_y = 1.;
    float_double eigen_vector2_x = (lambda2-sigma_y)/sigma_xy;
    float_double norm1 = sqrt(eigen_vector1_x*eigen_vector1_x+eigen_vector1_y*eigen_vector1_y);
    float_double norm2 = sqrt(eigen_vector2_x*eigen_vector2_x+eigen_vector2_y*eigen_vector2_y);
    eigen_vector1_x /= norm1;
    eigen_vector1_y /= norm1;
    eigen_vector2_x /= norm2;
    eigen_vector2_y /= norm2;

    point3d centroid_min_x = g.kernel.mu-dir.transpose()*((point3d(eigen_vector1_x, eigen_vector1_y, 0)/sqrt(lambda1/2))*sqrt((-negligeable_val_when_exp+new_log_weight)));
    point3d centroid_max_x = g.kernel.mu+dir.transpose()*((point3d(eigen_vector1_x, eigen_vector1_y, 0)/sqrt(lambda1/2))*sqrt((-negligeable_val_when_exp+new_log_weight)));
    point3d centroid_min_y = g.kernel.mu-dir.transpose()*((point3d(eigen_vector2_x, eigen_vector2_y, 0)/sqrt(lambda2/2))*sqrt((-negligeable_val_when_exp+new_log_weight)));
    point3d centroid_max_y = g.kernel.mu+dir.transpose()*((point3d(eigen_vector2_x, eigen_vector2_y, 0)/sqrt(lambda2/2))*sqrt((-negligeable_val_when_exp+new_log_weight)));
    float_double centroid_max_x_pixel_x, centroid_max_x_pixel_y, centroid_min_x_pixel_x, centroid_min_x_pixel_y, centroid_max_y_pixel_x, centroid_max_y_pixel_y, centroid_min_y_pixel_x, centroid_min_y_pixel_y;
    //get the screen coordinates of the centroids min / max x/y
    get_point_nearest_screen_cord_(src_, dir_, centroid_min_x_pixel_x, centroid_min_x_pixel_y, centroid_min_x, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_max_x_pixel_x, centroid_max_x_pixel_y, centroid_max_x, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_min_y_pixel_x, centroid_min_y_pixel_y, centroid_min_y, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_max_y_pixel_x, centroid_max_y_pixel_y, centroid_max_y, width, height);
    //get the bounding box of the gaussian
    min_x = min(centroid_min_x_pixel_x, min(centroid_max_x_pixel_x, min(centroid_min_y_pixel_x, centroid_max_y_pixel_x)));
    max_x = max(centroid_min_x_pixel_x, max(centroid_max_x_pixel_x, max(centroid_min_y_pixel_x, centroid_max_y_pixel_x)));
    min_y = min(centroid_min_x_pixel_y, min(centroid_max_x_pixel_y, min(centroid_min_y_pixel_y, centroid_max_y_pixel_y)));
    max_y = max(centroid_min_x_pixel_y, max(centroid_max_x_pixel_y, max(centroid_min_y_pixel_y, centroid_max_y_pixel_y)));
    //free(precomputed_reoriented_sigma);
     
    /*//get the 3d eigan values/vectors intead of the 2d ones
    float_double6 precomputed_reoriented_sigma  = g.get_reoriented_sigma_inv(dir);
    float_double sigma_x = precomputed_reoriented_sigma[0];
    float_double sigma_y = precomputed_reoriented_sigma[3];
    float_double sigma_z = precomputed_reoriented_sigma[5];
    float_double sigma_xy = precomputed_reoriented_sigma[1];
    float_double sigma_xz = precomputed_reoriented_sigma[2];
    float_double sigma_yz = precomputed_reoriented_sigma[4];
    float_double new_log_weight = g.kernel.log_weight+log(sqrt(M_PI/precomputed_reoriented_sigma[5]));
    //we now the 3 eigan values
    float_double lambda1 = g.kernel.scales3.x *g.kernel.scales3.x;
    float_double lambda2 = g.kernel.scales3.y *g.kernel.scales3.y;
    float_double lambda3 = g.kernel.scales3.z *g.kernel.scales3.z;

    //we know the 3 eigen vectors
    rotation_matrix r = (dir.transpose())* (rotation_matrix(g.kernel.quaternions4).transpose());
    point3d eigen_vector1 = r*point3d(1,0,0);
    point3d eigen_vector2 = r*point3d(0,1,0);
    point3d eigen_vector3 = r*point3d(0,0,1);
    //eigan vectors are already normalized
    
    //get the 3d bounding box
    point3d centroid_min_1 = g.kernel.mu-eigen_vector1*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda1));
    point3d centroid_max_1 = g.kernel.mu+eigen_vector1*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda1));
    point3d centroid_min_2 = g.kernel.mu-eigen_vector2*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda2));
    point3d centroid_max_2 = g.kernel.mu+eigen_vector2*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda2));
    point3d centroid_min_3 = g.kernel.mu-eigen_vector3*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda3));
    point3d centroid_max_3 = g.kernel.mu+eigen_vector3*(sqrt((-negligeable_val_when_exp+new_log_weight)/lambda3));
    float_double centroid_max_1_pixel_x, centroid_max_1_pixel_y, centroid_min_1_pixel_x, centroid_min_1_pixel_y, centroid_max_2_pixel_x, centroid_max_2_pixel_y, centroid_min_2_pixel_x, centroid_min_2_pixel_y, centroid_max_3_pixel_x, centroid_max_3_pixel_y, centroid_min_3_pixel_x, centroid_min_3_pixel_y;
    //get the screen coordinates of the centroids min / max 1/2/3
    get_point_nearest_screen_cord_(src_, dir_, centroid_min_1_pixel_x, centroid_min_1_pixel_y, centroid_min_1, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_max_1_pixel_x, centroid_max_1_pixel_y, centroid_max_1, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_min_2_pixel_x, centroid_min_2_pixel_y, centroid_min_2, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_max_2_pixel_x, centroid_max_2_pixel_y, centroid_max_2, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_min_3_pixel_x, centroid_min_3_pixel_y, centroid_min_3, width, height);
    get_point_nearest_screen_cord_(src_, dir_, centroid_max_3_pixel_x, centroid_max_3_pixel_y, centroid_max_3, width, height);
    //get the bounding box of the gaussian
    min_x = min(centroid_min_1_pixel_x, min(centroid_max_1_pixel_x, min(centroid_min_2_pixel_x, min(centroid_max_2_pixel_x, min(centroid_min_3_pixel_x, centroid_max_3_pixel_x)))));
    max_x = max(centroid_min_1_pixel_x, max(centroid_max_1_pixel_x, max(centroid_min_2_pixel_x, max(centroid_max_2_pixel_x, max(centroid_min_3_pixel_x, centroid_max_3_pixel_x)))));
    min_y = min(centroid_min_1_pixel_y, min(centroid_max_1_pixel_y, min(centroid_min_2_pixel_y, min(centroid_max_2_pixel_y, min(centroid_min_3_pixel_y, centroid_max_3_pixel_y)))));
    max_y = max(centroid_min_1_pixel_y, max(centroid_max_1_pixel_y, max(centroid_min_2_pixel_y, max(centroid_max_2_pixel_y, max(centroid_min_3_pixel_y, centroid_max_3_pixel_y)))));
    //free(precomputed_reoriented_sigma);*/
}
__host__ __device__ inline void get_worst_val_(float_double& val, point3d src_, rotation_matrix dir_, float_double min_x, float_double max_x, float_double min_y, float_double max_y,visual_gaussian_kernel g,int width, int height,ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type global_block_id,array_indexes_type ray_info_buffered_pixels_width,bool debug = false){
    val = 0;
    return;
    float_double middle_x = (min_x+max_x)/2;
    float_double middle_y = (min_y+max_y)/2;
    point3d src;
    rotation_matrix dir;

    //ray_info_(src_, src, dir_, dir, middle_x, middle_y, width, height);
    src = ray_info_buffered_pixels_midles[global_block_id].src;
    dir = ray_info_buffered_pixels_midles[global_block_id].dir;
  
    float_double6 precomputed_reoriented_sigma_inv  = g.get_reoriented_sigma_inv(dir);
    /*//get sigma according to then nearest point on the screen from it's centroid
    point3d src__;
    rotation_matrix dir__;
    float_double pixel_x, pixel_y;
    get_point_nearest_screen_cord_(src_, dir_, pixel_x, pixel_y, g.kernel.mu, width, height);
    ray_info_(src_, src__, dir_, dir__, pixel_x, pixel_y, width, height);
    float_double6 precomputed_reoriented_sigma_inv  = g.get_reoriented_sigma(dir__);
    */
    float_double l;
    point3d d = g.kernel.mu-src;
    d = dir * d;
    l = d.z;
    
    point3d src_min_x;
    rotation_matrix dir_min_x;
    //ray_info_(src_, &src_min_x, dir_, &dir_min_x, min_x, middle_y, width, height);
    src_min_x = ray_info_buffered_pixels[(array_indexes_type)(min_x + ((array_indexes_type)middle_y)*ray_info_buffered_pixels_width)].src;
    dir_min_x = ray_info_buffered_pixels[(array_indexes_type)(min_x + ((array_indexes_type)middle_y)*ray_info_buffered_pixels_width)].dir;
    point3d z_min_x = dir * point3d(dir_min_x[6], dir_min_x[7], dir_min_x[8]);
    /*point3d z_dir_min_x;
    ray_info_z_dir_src(src_, src_min_x, dir_,z_dir_min_x, min_x, middle_y, width, height);
    point3d z_min_x = dir * z_dir_min_x;*/
    //printf("point3d(dir_min_x[6], dir_min_x[7], dir_min_x[8]) %f %f %f, z_dir_min_x_ %f %f %f\n", dir_min_x[6], dir_min_x[7], dir_min_x[8], z_dir_min_x_.x, z_dir_min_x_.y, z_dir_min_x_.z);

    point3d diff_midle_min_x = src_min_x-src;
    diff_midle_min_x = dir*diff_midle_min_x;

    
    /*float_double x_min_x = diff_midle_min_x.x + ((z_min_x.x/z_min_x.z)*(diff_midle_min_x.z + l));
    float_double y_min_x = diff_midle_min_x.y + ((z_min_x.y/z_min_x.z)*(diff_midle_min_x.z + l));*/
    //take the true l instead
    point3d my_shifted_mu = g.kernel.mu - src_min_x;
    my_shifted_mu = dir * my_shifted_mu;
    float_double my_mu = my_shifted_mu.z + (my_shifted_mu.x*precomputed_reoriented_sigma_inv[2]+my_shifted_mu.y*precomputed_reoriented_sigma_inv[4])/precomputed_reoriented_sigma_inv[5];
    float_double x_min_x = diff_midle_min_x.x + ((z_min_x.x/z_min_x.z)*(diff_midle_min_x.z + my_mu));
    float_double y_min_x = diff_midle_min_x.y + ((z_min_x.y/z_min_x.z)*(diff_midle_min_x.z + my_mu));
    x_min_x = x_min_x - d.x;
    y_min_x = y_min_x - d.y;


    point3d src_max_x;
    rotation_matrix dir_max_x;
    //ray_info_(src_, &src_max_x, dir_, &dir_max_x, max_x, middle_y, width, height);
    src_max_x = ray_info_buffered_pixels[(array_indexes_type)(max_x + ((array_indexes_type)middle_y)*ray_info_buffered_pixels_width)].src;
    dir_max_x = ray_info_buffered_pixels[(array_indexes_type)(max_x + ((array_indexes_type)middle_y)*ray_info_buffered_pixels_width)].dir;
    point3d z_max_x = dir * point3d(dir_max_x[6], dir_max_x[7], dir_max_x[8]);
    /*point3d z_dir_max_x;
    ray_info_z_dir_src(src_, src_max_x, dir_,z_dir_max_x, max_x, middle_y, width, height);
    point3d z_max_x = dir * z_dir_max_x;*/
    point3d diff_midle_max_x = src_max_x-src;
    diff_midle_max_x = dir*diff_midle_max_x;

    /*float_double x_max_x = diff_midle_max_x.x + ((z_max_x.x/z_max_x.z)*(diff_midle_max_x.z + l));
    float_double y_max_x = diff_midle_max_x.y + ((z_max_x.y/z_max_x.z)*(diff_midle_max_x.z + l));*/
    //take the true l instead
    my_shifted_mu = g.kernel.mu - src_max_x;
    my_shifted_mu = dir * my_shifted_mu;
    my_mu = my_shifted_mu.z + (my_shifted_mu.x*precomputed_reoriented_sigma_inv[2]+my_shifted_mu.y*precomputed_reoriented_sigma_inv[4])/precomputed_reoriented_sigma_inv[5];
    float_double x_max_x = diff_midle_max_x.x + ((z_max_x.x/z_max_x.z)*(diff_midle_max_x.z + my_mu));
    float_double y_max_x = diff_midle_max_x.y + ((z_max_x.y/z_max_x.z)*(diff_midle_max_x.z + my_mu));

    x_max_x = x_max_x - d.x;
    y_max_x = y_max_x - d.y;

    point3d src_min_y;
    rotation_matrix dir_min_y;
    //ray_info_(src_, &src_min_y, dir_, &dir_min_y, middle_x, min_y, width, height);
    src_min_y = ray_info_buffered_pixels[(array_indexes_type)(((array_indexes_type)middle_x) + min_y*ray_info_buffered_pixels_width)].src;
    dir_min_y = ray_info_buffered_pixels[(array_indexes_type)(((array_indexes_type)middle_x) + min_y*ray_info_buffered_pixels_width)].dir;
    point3d z_min_y = dir * point3d(dir_min_y[6], dir_min_y[7], dir_min_y[8]);
    /*point3d z_dir_min_y;
    ray_info_z_dir_src(src_, src_min_y, dir_,z_dir_min_y, middle_x, min_y, width, height);
    point3d z_min_y = dir * z_dir_min_y;*/
    point3d diff_midle_min_y = src_min_y-src;
    diff_midle_min_y = dir*diff_midle_min_y;
    /*float_double x_min_y = diff_midle_min_y.x + ((z_min_y.x/z_min_y.z)*(diff_midle_min_y.z + l));
    float_double y_min_y = diff_midle_min_y.y + ((z_min_y.y/z_min_y.z)*(diff_midle_min_y.z + l));*/
    //take the true l instead
    my_shifted_mu = g.kernel.mu - src_min_y;
    my_shifted_mu = dir * my_shifted_mu;
    my_mu = my_shifted_mu.z + (my_shifted_mu.x*precomputed_reoriented_sigma_inv[2]+my_shifted_mu.y*precomputed_reoriented_sigma_inv[4])/precomputed_reoriented_sigma_inv[5];
    float_double x_min_y = diff_midle_min_y.x + ((z_min_y.x/z_min_y.z)*(diff_midle_min_y.z + my_mu));
    float_double y_min_y = diff_midle_min_y.y + ((z_min_y.y/z_min_y.z)*(diff_midle_min_y.z + my_mu));
    x_min_y = x_min_y - d.x;
    y_min_y = y_min_y - d.y;

    point3d src_max_y;
    rotation_matrix dir_max_y;
    //ray_info_(src_, &src_max_y, dir_, &dir_max_y, middle_x, max_y, width, height);
    src_max_y = ray_info_buffered_pixels[(array_indexes_type)(((array_indexes_type)middle_x) + max_y*ray_info_buffered_pixels_width)].src;
    dir_max_y = ray_info_buffered_pixels[(array_indexes_type)(((array_indexes_type)middle_x) + max_y*ray_info_buffered_pixels_width)].dir;
    point3d z_max_y = dir * point3d(dir_max_y[6], dir_max_y[7], dir_max_y[8]);
    /*point3d z_dir_max_y;
    ray_info_z_dir_src(src_, src_max_y, dir_,z_dir_max_y, middle_x, max_y, width, height);
    point3d z_max_y = dir * z_dir_max_y;*/
    point3d diff_midle_max_y = src_max_y-src;
    diff_midle_max_y = dir*diff_midle_max_y;

    /*float_double x_max_y = diff_midle_max_y.x + ((z_max_y.x/z_max_y.z)*(diff_midle_max_y.z + l));
    float_double y_max_y = diff_midle_max_y.y + ((z_max_y.y/z_max_y.z)*(diff_midle_max_y.z + l));*/
    //take the true l instead
    my_shifted_mu = g.kernel.mu - src_max_y;
    my_shifted_mu = dir * my_shifted_mu;
    my_mu = my_shifted_mu.z + (my_shifted_mu.x*precomputed_reoriented_sigma_inv[2]+my_shifted_mu.y*precomputed_reoriented_sigma_inv[4])/precomputed_reoriented_sigma_inv[5];
    float_double x_max_y = diff_midle_max_y.x + ((z_max_y.x/z_max_y.z)*(diff_midle_max_y.z + my_mu));
    float_double y_max_y = diff_midle_max_y.y + ((z_max_y.y/z_max_y.z)*(diff_midle_max_y.z + my_mu));

    x_max_y = x_max_y - d.x;
    y_max_y = y_max_y - d.y;



    float_double x1 = min(x_min_x, min(x_max_x, min(x_min_y, x_max_y)));
    float_double x2 = max(x_min_x, max(x_max_x, max(x_min_y, x_max_y)));
    float_double y1 = min(y_min_x, min(y_max_x, min(y_min_y, y_max_y)));
    float_double y2 = max(y_min_x, max(y_max_x, max(y_min_y, y_max_y)));
    if(debug){
        printf("x1 %f, x2 %f, y1 %f, y2 %f\n", x1, x2, y1, y2);
    }
    float_double x_square_inf = min(x1*x1, x2*x2);
    if(signbit(x1) != signbit(x2)){
        x_square_inf = 0;
    }

    float_double y_square_inf = min(y1*y1, y2*y2);
    if(signbit(y1) != signbit(y2)){
        y_square_inf = 0;
    }

    float_double xy_inf = min(x1*y1, min( x2*y2, min( x1*y2, x2*y1)));
    if(signbit(x1) != signbit(x2) || signbit(y1) != signbit(y2)){
        xy_inf = min(0., xy_inf);
    }


    float_double x_square_sup = max(x1*x1, x2*x2);

    float_double y_square_sup = max(y1*y1, y2*y2);

    float_double xy_sup = max(x1*y1,max( x2*y2,max( x1*y2, x2*y1)));
    if(signbit(x1) != signbit(x2) || signbit(y1) != signbit(y2)){
        xy_sup = max(0., xy_sup);
    }


    
    /*float_double sigma_x = precomputed_reoriented_sigma[0];
    float_double sigma_y = precomputed_reoriented_sigma[3];
    float_double sigma_xy = precomputed_reoriented_sigma[1];*/
    float_double6 precomputed_reoriented_sigma = g.get_reoriented_sigma(dir);
    float_double inv_det = 1/(precomputed_reoriented_sigma[0]*precomputed_reoriented_sigma[3]-precomputed_reoriented_sigma[1]*precomputed_reoriented_sigma[1]);
    float_double sigma_x = precomputed_reoriented_sigma[3]*inv_det;
    float_double sigma_y = precomputed_reoriented_sigma[0]*inv_det;
    float_double sigma_xy = -precomputed_reoriented_sigma[1]*inv_det;

    float_double new_log_weight = g.kernel.log_weight+log(sqrt(M_PI/precomputed_reoriented_sigma_inv[5]));
    val = -sigma_x*x_square_inf-sigma_y*y_square_inf;
    if(debug){
        printf("x_square_inf %f, y_square_inf %f, xy_inf %f, x_square_sup %f, y_square_sup %f, xy_sup %f, sigma_x %f, sigma_y %f, sigma_xy %f, new_log_weight %f, val %f\n", x_square_inf, y_square_inf, xy_inf, x_square_sup, y_square_sup, xy_sup, sigma_x, sigma_y, sigma_xy, new_log_weight, val);
    }
    if (sigma_xy > 0){
        val += -2*sigma_xy*xy_inf;
    }
    else{
        val += -2*sigma_xy*xy_sup;
    }
    if(debug){
        printf("val %f\n", val);
    }
    val += new_log_weight;
    //free(precomputed_reoriented_sigma);

}