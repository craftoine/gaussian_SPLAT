
#include "class.cuh"
#include "rendering/rendering_functions.cuh"
//include so we can use cv
#include "opencv2/opencv.hpp"

using namespace cv;

int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }

    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (cudaGetLastError() != cudaSuccess)
        {
            std::cerr << "Error getting device properties for device " << device << std::endl;
            return 1;
        }

        if (deviceProp.major >= 1)
        {
            break;
        }
    }

    if (device == deviceCount)
    {
        std::cerr << "No CUDA devices with at least compute capability 1.0 found" << std::endl;
        return 1;
    }

    cudaSetDevice(device);
    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Error setting CUDA device" << std::endl;
        return 1;
    }
    //increase heap size limit
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2147483647);
    array_indexes_type N = 1000000;
    std::vector<gaussian_kernel2_3D> kernels3;
    for(array_indexes_type i=0;i<N;i++){
        point3d mu;
        /*mu.x= 0+ 0*10*((float_double)rand()/(float_double)RAND_MAX - 0.5);
        mu.y= 0+ 0*10*((float_double)rand()/(float_double)RAND_MAX - 0.5);
        mu.z= 10+0*((float_double)rand()/(float_double)RAND_MAX - 0.5);*/
        
        /*float_double theta = 2*M_PI*((float_double)rand()/(float_double)RAND_MAX);
        float_double phi = M_PI*((float_double)rand()/(float_double)RAND_MAX);*/
        float_double r = sqrt(5*5+ 300*300*((float_double)rand()/(float_double)RAND_MAX));
        /*mu.x = r*sin(phi)*cos(theta);
        mu.y = r*sin(phi)*sin(theta);
        mu.z = r*cos(phi);*/
        //generate with a normal law centered in 0
        mu.x = erfcinvf(2*((float_double)rand()/(float_double)RAND_MAX));
        mu.y = erfcinvf(2*((float_double)rand()/(float_double)RAND_MAX));
        mu.z = erfcinvf(2*((float_double)rand()/(float_double)RAND_MAX));
        float_double norm = sqrt(mu.x*mu.x+mu.y*mu.y+mu.z*mu.z);
        mu.x = mu.x/norm;
        mu.y = mu.y/norm;
        mu.z = mu.z/norm;
        mu.x*=r;
        mu.y*=r;
        mu.z*=r;
        //std::cout << "mu: " << mu.x << " " << mu.y << " " << mu.z << std::endl;
        float_double* scale_ = new float_double[3];
        random_scale(scale_);
        for(dimension_type j=0;j<3;j++){
            scale_[j] = scale_[j]*sqrt(r)/(sqrt(5)*3);
        }
        float_double* q = new float_double[4];
        random_unitary_quaternion(q);
        random_unitary_quaternion(q);
        const float min_weight = 0.2;
        float_double weight =min_weight+((1-min_weight)*((float_double)rand()/(float_double)RAND_MAX));
        kernels3.push_back(gaussian_kernel2_3D(mu, log(weight*500/r) , scale_, q));
        free(q);
        free(scale_);
    }
    //std::cout << "last kernel: " << kernels3[N-1] << std::endl;
    std::cout << "kernels3 created" << std::endl;

    std::vector<visual_gaussian_kernel> v_kernels;
    for(array_indexes_type i=0;i<N;i++){
        //generate the random spherical harmonics
        point3d* sh = new point3d[16];
        for(array_indexes_type j=0;j<16;j++){
            sh[j].x = ((float_double)rand()/(float_double)RAND_MAX);
            sh[j].y = ((float_double)rand()/(float_double)RAND_MAX);
            sh[j].z = ((float_double)rand()/(float_double)RAND_MAX);
        }
        spherical_harmonic_color sh_color(sh);
        v_kernels.push_back(visual_gaussian_kernel(kernels3[i], sh_color));
        free(sh);
    }
    std::cout << "v_kernels created" << std::endl;
    
    //exchange the last with the first kernel
    /*visual_gaussian_kernel temp = v_kernels[0];
    v_kernels[0] = v_kernels[N-1];
    v_kernels[N-1] = temp;
    N= 1;*/
    /*//exchange the last with the second kernel
    temp = v_kernels[1];
    v_kernels[1] = v_kernels[N-1];
    v_kernels[N-1] = temp;
    N = 2;*/
    /*//print all my kernels
    for(array_indexes_type i=0;i<N;i++){
        std::cout << "kernel " << i << ": " << v_kernels[i].kernel << std::endl;
    }*/
    //load true gaussians instead

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
    
    /*std::vector<visual_gaussian_kernel> v_kernels_true;
    //std::ifstream infile3("../knn_gaussian/knn_gaussian_cpu/input.ply", std::ios::binary);
    std::ifstream infile3("point_cloud.ply", std::ios::binary);
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

        point3d sh[16];
        for(array_indexes_type j=0;j<16;j++){
            float sh_temp[3];
            infile3.read(reinterpret_cast<char*>(&sh_temp[0]), sizeof(float));
            infile3.read(reinterpret_cast<char*>(&sh_temp[1]), sizeof(float));
            infile3.read(reinterpret_cast<char*>(&sh_temp[2]), sizeof(float));
            sh[j].x = sh_temp[0];
            sh[j].y = sh_temp[1];
            sh[j].z = sh_temp[2];
        }
        spherical_harmonic_color sh_color(sh);
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
        /*opacity =( 1/(1+exp(-opacity)));
        opacity = log(opacity);*/
        //opacity = 0;

        //create scale and quaternion vectors*/
        /*float_double scale[3];
        scale[0] = (float_double) scale_0;
        scale[1] = (float_double) scale_1;
        scale[2] = (float_double) scale_2;
        
        float_double q[4];
        q[0] = (float_double) rot_0;
        q[1] = (float_double) rot_1;
        q[2] = (float_double) rot_2;
        q[3] = (float_double) rot_3;

        point3d mu;
        mu.x = (float_double)x;
        mu.y = (float_double)y;
        mu.z = (float_double)z;
*/

        //float_double det = scale[0]*scale[1]*scale[2]*scale[0]*scale[1]*scale[2];
        //opacity += log(sqrt(pow((2*M_PI),3)*det));
        //float_double min_sig = max(abs(scale[0]),max(abs(scale[1]),abs(scale[2])));
        //min_sig = min_sig*min_sig;
        /*rotation_matrix rot(q);
        float_double M[3][3];
        M[0][0] = (1/scale[0])*rot[0];
        M[1][0] = (1/scale[1])*rot[3];
        M[2][0] = (1/scale[2])*rot[6];
        M[0][1] = (1/scale[0])*rot[1];
        M[1][1] = (1/scale[1])*rot[4];
        M[2][1] = (1/scale[2])*rot[7];
        M[0][2] = (1/scale[0])*rot[2];
        M[1][2] = (1/scale[1])*rot[5];
        M[2][2] = (1/scale[2])*rot[8];
        float_double min_sig = 1/(M[0][2]*M[0][2] + M[1][2]*M[1][2] + M[2][2]*M[2][2]);*/
        //opacity += -log(sqrt(2*M_PI)*min_sig);
        //opacity = log(-log(1-opacity)/sqrt(2*M_PI*min_sig));
        
   /*     gaussian_kernel2_3D kernel(mu, opacity, scale, q);

        visual_gaussian_kernel v_kernel(kernel, sh_color);
        v_kernels_true.push_back(v_kernel);
        i3++;
        //std::cout << i3 << " opacity : "<< opacity << std::endl;
    }
    std::cout << "v_kernels_true created" << std::endl;
    N = v_kernels_true.size();
    std::cout << "N: " << N << std::endl;
    v_kernels = v_kernels_true;
    
    //print first kernel
    std::cout << "first kernel: " << v_kernels[5000].kernel << std::endl;
*/

























    point3d src;
    /*src.x = 0;
    src.y = 2;
    src.z = -1;*/
    /*src.x = -0.8;
    src.y = 1.3;
    src.z = -0;*/
    /*src.x = 0;
    src.y = 0.1;
    src.z = -1.7;*/
    src.x = -1;
    src.y = 0;
    src.z = -2;
    float_double * identity_rotation = new float_double[9];
    for (int j = 0; j < 9; j++)
    {
        identity_rotation[j] = 0;
    }
    identity_rotation[0] = 1;
    identity_rotation[4] = 1;
    identity_rotation[8] = 1;
    rotation_matrix dir_ = rotation_matrix(identity_rotation);
    free(identity_rotation);
    const unsigned int width = 1000;
    const unsigned int height = 1000;
    //render the screen on cpu
    point3d* screen = new point3d[width*height];
    //void render_screen (point3d *screen, point3d src,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height)
    render_screen(screen, src, dir_, v_kernels.data(), min(N,3), width, height);
    std::cout << "screen rendered" << std::endl;

    //show the screen openning a window
    //cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0,0,0));
    ///replace witht hieght and width
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0,0,0));

    /*for(array_indexes_type i=0;i<100;i++){
        for(array_indexes_type j=0;j<100;j++){
            img.at<cv::Vec3b>(height-i-1,j)[0] = screen[i*100+j].x;
            img.at<cv::Vec3b>(height-i-1,j)[1] = screen[i*100+j].y;
            img.at<cv::Vec3b>(height-i-1,j)[2] = screen[i*100+j].z;
        }
    }*/
    std::cout << "image created" << std::endl;
    for(array_indexes_type i=0;i<height;i++){
        for(array_indexes_type j=0;j<width;j++){
            //std::cout << "i: " << i << " j: " << j << std::endl;
            //std::cout << "r: " << screen[j*height+i].x << " g: " << screen[j*height+i].y << " b: " << screen[j*height+i].z << std::endl;
            img.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
            img.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
            img.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
        }
    }
    cv::imshow("image", img);
    cv::waitKey(0);
    //gpu version
    //render the screen on gpu
    point3d* screen_gpu;
    cudaMalloc(&screen_gpu, width*height*sizeof(point3d));
    
    //copy the kernels to the gpu
    visual_gaussian_kernel* v_kernels_gpu;
    cudaMalloc(&v_kernels_gpu, N*sizeof(visual_gaussian_kernel));
    printf("v_kernels allocated for a size of %ld \n", N*sizeof(visual_gaussian_kernel));
    cudaMemcpy(v_kernels_gpu, v_kernels.data(), N*sizeof(visual_gaussian_kernel), cudaMemcpyHostToDevice);
    std::cout << "v_kernels copied to gpu" << std::endl;

    //render the screen on gpu
    array_indexes_type block_size_x = rendering_block_size;
    array_indexes_type block_size_y = rendering_block_size;
    array_indexes_type grid_size_x = width/block_size_x + (width%block_size_x == 0 ? 0 : 1);
    array_indexes_type grid_size_y = height/block_size_y + (height%block_size_y == 0 ? 0 : 1);
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);
    //dim3 grid(2, 2);
    //array_indexes_type shared_memory_size = N * sizeof(index_value);
    array_indexes_type shared_memory_size = 2*N * sizeof(index_start_end) + sizeof(array_indexes_type*);
    //array_indexes_type shared_memory_size = 2*N * sizeof(index_start_end);
    //render_screen_block<<<grid, block, shared_memory_size>>>(screen_gpu, src,dir_, v_kernels_gpu, N, width, height);
    //render_screen_block_mix<<<grid, block, shared_memory_size>>>(screen_gpu, src,dir_, v_kernels_gpu, N, width, height);
/*  render_screen_block_my<<<grid, block, shared_memory_size>>>(screen_gpu, src,dir_, v_kernels_gpu, min(100,N), width, height);
    //render_screen_block_my_2<<<grid, block, shared_memory_size>>>(screen_gpu, src,dir_, v_kernels_gpu, N, width, height);
    cudaDeviceSynchronize();
    //cath the error
*/    cudaError_t error = cudaGetLastError();
/*    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "screen rendered on gpu" << std::endl;

    //copy the screen to the host
    cudaMemcpy(screen, screen_gpu, width*height*sizeof(point3d), cudaMemcpyDeviceToHost);
    std::cout << "screen copied to host" << std::endl;

    //show the screen openning a window
    cv::Mat img_gpu(height, width, CV_8UC3, cv::Scalar(0,0,0));
    for(array_indexes_type i=0;i<height;i++){
        for(array_indexes_type j=0;j<width;j++){
            img_gpu.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
            img_gpu.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
            img_gpu.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
        }
    }
    cv::imshow("image_gpu", img_gpu);
    cv::waitKey(0);
*/
    cudaFree(screen_gpu);


    //test the solution with selection of kernels

    //FIRSTLY PRECOMPUTE RAY DIURECTION AND SOURCES
    //__global__ void clean_and_compute_ray_info_buffered(int width, int height, point3d src, rotation_matrix dir, ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type ray_info_buffered_pixels_width){
    ray_info_buffered* ray_info_buffered_pixels;
    cudaMalloc(&ray_info_buffered_pixels, grid_size_x*grid_size_y*block_size_x*block_size_y*sizeof(ray_info_buffered));
    printf("ray_info_buffered_pixels allocated for a size of %ld \n", grid_size_x*grid_size_y*block_size_x*block_size_y*sizeof(ray_info_buffered));
    ray_info_buffered* ray_info_buffered_pixels_midles;
    cudaMalloc(&ray_info_buffered_pixels_midles, grid_size_x*grid_size_y*sizeof(ray_info_buffered));
    printf("ray_info_buffered_pixels_midles allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(ray_info_buffered));    
    clean_and_compute_ray_info_buffered<<<grid, block>>>(width, height, src, dir_, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "ray_info_buffered_pixels computed" << std::endl;

    int selected_blocks = 81;
    int selected_threads = 32;
    array_indexes_type ** selected_kernels;
    cudaMalloc(&selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
    printf("selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
    printf("allocated at %p \n", selected_kernels);

    array_indexes_type * stacks_sizes;
    cudaMalloc(&stacks_sizes, grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
    printf("stacks_sizes allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
    printf("allocated at %p \n", stacks_sizes);

    //__global__ void select_gaussian_kernel_2(visual_gaussian_kernel* kernels, int n_kernels, int width, int height, point3d src, rotation_matrix dir, array_indexes_type ** selected_kernels, array_indexes_type * stacks_sizes, ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type ray_info_buffered_pixels_width){
    select_gaussian_kernel_2<<<selected_blocks,selected_threads>>>(v_kernels_gpu, N, width, height, src, dir_, selected_kernels, stacks_sizes, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "selected_kernels computed" << std::endl;

    array_indexes_type* nb_selected_kernels;
    cudaMalloc(&nb_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type));
    printf("nb_selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type));
    array_indexes_type** result_selected_kernels;
    cudaMalloc(&result_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*));
    printf("result_selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*));
    //launch compact_selected_kernels_2 on one SAMEW GRID AS RENDERING, on same block as rendering
    //__global__ void compact_selected_kernels_2(visual_gaussian_kernel* kernels, int n_kernels, int width, int height, point3d src, rotation_matrix dir, array_indexes_type ** selected_kernels, array_indexes_type * stacks_sizes, array_indexes_type ** result_selected_kernels, array_indexes_type * result_nb_selected_kernels,int nb_selected_threads){
    compact_selected_kernels_2<<<grid, block>>>(v_kernels_gpu, N, width, height, src, dir_, selected_kernels, stacks_sizes, result_selected_kernels, nb_selected_kernels, selected_threads*selected_blocks);
    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "compact_selected_kernels computed" << std::endl;
    //free the selected_kernels array 
    //LAUNCH __global__ void free_select_gaussian_kernel_2(int n_kernels, int width, int height, array_indexes_type ** selected_kernels){
    // with same parameters as select_gaussian_kernel_2
    free_select_gaussian_kernel_2<<<selected_blocks,selected_threads>>>(N, width, height, selected_kernels);
    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    cudaFree(selected_kernels);
    std::cout << "selected_kernels freed" << std::endl;
    cudaFree(stacks_sizes);
    point3d* screen_gpu2;
    cudaMalloc(&screen_gpu2, width*height*sizeof(point3d));

    //compute max N
    array_indexes_type max_N = 0;
    array_indexes_type sum_N = 0;
    array_indexes_type* nb_selected_kernels_host = new array_indexes_type[grid_size_x*grid_size_y];
    cudaMemcpy(nb_selected_kernels_host, nb_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
    for(array_indexes_type i=0;i<grid_size_x*grid_size_y;i++){
        sum_N += nb_selected_kernels_host[i];
        if(nb_selected_kernels_host[i]>max_N){
            max_N = nb_selected_kernels_host[i];
        }
    }

    void** shared_sorted_indexes__;
    cudaMalloc(&shared_sorted_indexes__, grid_size_x*grid_size_y*sizeof(void*));
    

    free(nb_selected_kernels_host);


    std::cout << "max_N: " << max_N << std::endl;
    std::cout << "sum_N: " << sum_N << std::endl;
    std::cout << "memory usage estimated :" << sum_N*sizeof(index_value) << std::endl;
    array_indexes_type shared_memory_size_ = max_N * sizeof(index_value);
    //array_indexes_type shared_memory_size_ = 2*max_N * sizeof(index_start_end) + sizeof(array_indexes_type*);
    //array_indexes_type shared_memory_size_ = 2*max_N * sizeof(index_start_end);
    //dim3 grid_(2,1);
    //launch rendering
    //void render_screen_block_selected(point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int width, int height,array_indexes_type ** result_selected_kernels, array_indexes_type * result_nb_selected_kernels, ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type ray_info_buffered_pixels_width){
    //render_screen_block_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    //render_screen_block_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
    //render_screen_block_mix_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    //render_screen_block_mix_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
    //render_screen_block_my_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    //render_screen_block_my_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
    //render_screen_block_my_2_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
    render_screen_block_my_2_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
    cudaFree(shared_sorted_indexes__);

    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::cout << "screen rendered on gpu with selected kernels" << std::endl;
    cudaFree(result_selected_kernels);
    cudaFree(nb_selected_kernels);
    //copy the screen to the host
    cudaMemcpy(screen, screen_gpu2, width*height*sizeof(point3d), cudaMemcpyDeviceToHost);
    cudaFree(screen_gpu2);
    std::cout << "screen copied to host" << std::endl;

    //show the screen openning a window
    cv::Mat img_gpu_selected(height, width, CV_8UC3, cv::Scalar(0,0,0));
    for(array_indexes_type i=0;i<height;i++){
        for(array_indexes_type j=0;j<width;j++){
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
        }
    }
    cv::imshow("image_gpu_selected", img_gpu_selected);
    cv::waitKey(0);
    

    float_double alpha =0.;
    float_double beta = 0.;
    float_double gamma = 0.;
    int i = 0;
    while(true){
        //compute rendering time
        auto start = std::chrono::high_resolution_clock::now();


        i++;
        //std::cout << "alpha: " << alpha << " beta: " << beta << " gamma: " << gamma << std::endl;
        
        //rotate the screen
        float_double * rotation = new float_double[9];
        rotation[0] = cos(alpha)*cos(beta);
        rotation[1] = cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma);
        rotation[2] = cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma);
        rotation[3] = sin(alpha)*cos(beta);
        rotation[4] = sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma);
        rotation[5] = sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma);
        rotation[6] = -sin(beta);
        rotation[7] = cos(beta)*sin(gamma);
        rotation[8] = cos(beta)*cos(gamma);

        dir_ = rotation_matrix(rotation);
        //std::cout << "rotation: " << rotation[0] << " " << rotation[1] << " " << rotation[2] << std::endl;
        //std::cout << "rotation: " << rotation[3] << " " << rotation[4] << " " << rotation[5] << std::endl;
        //std::cout << "rotation: " << rotation[6] << " " << rotation[7] << " " << rotation[8] << std::endl;
        //std::cout << "quaternion: " << dir_.quaternions4.x << " " << dir_.quaternions4.y << " " << dir_.quaternions4.z << " " << dir_.quaternions4.w << std::endl;
        //std::cout << "dir_matrix: " << dir_[0] << " " << dir_[1] << " " << dir_[2] << std::endl;
        //std::cout << "dir_matrix: " << dir_[3] << " " << dir_[4] << " " << dir_[5] << std::endl;
        //std::cout << "dir_matrix: " << dir_[6] << " " << dir_[7] << " " << dir_[8] << std::endl;
        //std::cout << "dir_transpose_matrix: " << dir_.transpose()[0] << " " << dir_.transpose()[1] << " " << dir_.transpose()[2] << std::endl;
        //std::cout << "dir_transpose_matrix: " << dir_.transpose()[3] << " " << dir_.transpose()[4] << " " << dir_.transpose()[5] << std::endl;
        //std::cout << "dir_transpose_matrix: " << dir_.transpose()[6] << " " << dir_.transpose()[7] << " " << dir_.transpose()[8] << std::endl;
        free(rotation);
        //render the screen on gpu

        //do the blocks stuff
        //firstly precompute ray direction and sources
        clean_and_compute_ray_info_buffered<<<grid, block>>>(width, height, src, dir_, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        std::cout << "ray_info_buffered_pixels computed" << std::endl;

        cudaMalloc(&selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
        printf("selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
        printf("allocated at %p \n", selected_kernels);

        cudaMalloc(&stacks_sizes, grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
        printf("stacks_sizes allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
        printf("allocated at %p \n", stacks_sizes);

        select_gaussian_kernel_2<<<selected_blocks,selected_threads>>>(v_kernels_gpu, N, width, height, src, dir_, selected_kernels, stacks_sizes, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        std::cout << "selected_kernels computed" << std::endl;

        cudaMalloc(&nb_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type));
        cudaMalloc(&result_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*));
        compact_selected_kernels_2<<<grid, block>>>(v_kernels_gpu, N, width, height, src, dir_, selected_kernels, stacks_sizes, result_selected_kernels, nb_selected_kernels, selected_threads*selected_blocks);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        std::cout << "compact_selected_kernels computed" << std::endl;
        free_select_gaussian_kernel_2<<<selected_blocks,selected_threads>>>(N, width, height, selected_kernels);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        cudaFree(selected_kernels);
        std::cout << "selected_kernels freed" << std::endl;
        cudaFree(stacks_sizes);
        point3d* screen_gpu2;
        cudaMalloc(&screen_gpu2, width*height*sizeof(point3d));

        //compute max N
        max_N = 0;
        nb_selected_kernels_host = new array_indexes_type[grid_size_x*grid_size_y];
        cudaMemcpy(nb_selected_kernels_host, nb_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
        for(array_indexes_type i=0;i<grid_size_x*grid_size_y;i++){
            if(nb_selected_kernels_host[i]>max_N){
                max_N = nb_selected_kernels_host[i];
            }
        }

        void** shared_sorted_indexes__;
        cudaMalloc(&shared_sorted_indexes__, grid_size_x*grid_size_y*sizeof(void*));
    

        free(nb_selected_kernels_host);

        std::cout << "max_N: " << max_N << std::endl;
        shared_memory_size_ = max_N * sizeof(index_value);
        //shared_memory_size_ = 2*max_N * sizeof(index_start_end) + sizeof(array_indexes_type*);
        //shared_memory_size_ = 2*max_N * sizeof(index_start_end);
        //std::cout << "shared_memory_size_: " << shared_memory_size_ << "for " << 2*max_N<< " index_start_end" << std::endl;

        //grid_(2,1);
        //render_screen_block_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        //render_screen_block_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
        //render_screen_block_mix_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        //render_screen_block_mix_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
        //render_screen_block_my_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        //render_screen_block_my_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
        //render_screen_block_my_2_selected<<<grid, block, shared_memory_size_>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        render_screen_block_my_2_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        cudaFree(shared_sorted_indexes__);
        std::cout << "screen rendered on gpu with selected kernels" << std::endl;
        cudaFree(result_selected_kernels);
        cudaFree(nb_selected_kernels);
        //copy the screen to the host
        cudaMemcpy(screen, screen_gpu2, width*height*sizeof(point3d), cudaMemcpyDeviceToHost);
        cudaFree(screen_gpu2);
        std::cout << "screen copied to host" << std::endl;

        //show the screen openning a window
        cv::Mat img_gpu_selected(height, width, CV_8UC3, cv::Scalar(0,0,0));
        for(array_indexes_type i=0;i<height;i++){
            for(array_indexes_type j=0;j<width;j++){
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
            }
        }
        cv::imshow("image_gpu_selected", img_gpu_selected);
        cv::waitKey(1);





        alpha += 0.01/10;
        beta += 0.0171/10;
        gamma += 0.0059/10;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "elapsed time: " << elapsed.count() << "s" << std::endl;
    }
    cudaFree(ray_info_buffered_pixels);
    cudaFree(ray_info_buffered_pixels_midles);
    cudaFree(v_kernels_gpu);
    free(screen);
    
    return 0;
}
