#include "class.cuh"
#include "rendering/rendering_functions.cuh"
#include "opencv2/opencv.hpp"
#define overlapping
using namespace cv;

int main(int argc, char *argv[]){
    int deviceCount;
    cudaError_t error;
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
    array_indexes_type N;
    
    std::vector<visual_gaussian_kernel> v_kernels;
    
    
    if(argc>1){
        std::vector<visual_gaussian_kernel> v_kernels;
        std::ifstream infile3(argv[1], std::ios::binary);
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

            point3d mu;
            mu.x = (float_double)x;
            mu.y = (float_double)y;
            mu.z = (float_double)z;

            gaussian_kernel2_3D kernel(mu, opacity, scale, q);

            visual_gaussian_kernel v_kernel(kernel, sh_color);
            v_kernels.push_back(v_kernel);
            i3++;
            //std::cout << i3 << " opacity : "<< opacity << std::endl;
        }
        std::cout << "v_kernels created" << std::endl;
        N = v_kernels.size();
        std::cout << "N: " << N << std::endl;
    }

    else{
        std::vector<gaussian_kernel2_3D> kernels3;
        N = 1000000;
        
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
                scale_[j] = scale_[j]*sqrt(r)/(sqrt(5)*5);
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
        
        for(array_indexes_type i=0;i<N;i++){
            //generate the random spherical harmonics
            point3d* sh = new point3d[16];
            for(array_indexes_type j=0;j<16;j++){
                sh[j].x = ((float_double)rand()/(float_double)RAND_MAX)*0.1;
                sh[j].y = ((float_double)rand()/(float_double)RAND_MAX)*0.1;
                sh[j].z = ((float_double)rand()/(float_double)RAND_MAX)*0.1;
            }
            spherical_harmonic_color sh_color(sh);
            v_kernels.push_back(visual_gaussian_kernel(kernels3[i], sh_color));
            free(sh);
        }
    }

    
    point3d src;
    src.x = 0;
    src.y = 0;
    src.z = 0;
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
    array_indexes_type block_size_x = rendering_block_size;
    array_indexes_type block_size_y = rendering_block_size;
    array_indexes_type grid_size_x = width/block_size_x + (width%block_size_x == 0 ? 0 : 1);
    array_indexes_type grid_size_y = height/block_size_y + (height%block_size_y == 0 ? 0 : 1);
    dim3 block(block_size_x, block_size_y);
    dim3 grid(grid_size_x, grid_size_y);
    visual_gaussian_kernel* v_kernels_gpu;
    cudaMalloc(&v_kernels_gpu, N*sizeof(visual_gaussian_kernel));
    //printf("v_kernels allocated for a size of %ld \n", N*sizeof(visual_gaussian_kernel));
    cudaMemcpy(v_kernels_gpu, v_kernels.data(), N*sizeof(visual_gaussian_kernel), cudaMemcpyHostToDevice);
    point3d* screen = new point3d[width*height];
    //render the screen on gpu
    
    //FIRSTLY PRECOMPUTE RAY DIRECTION AND SOURCES
    ray_info_buffered* ray_info_buffered_pixels;
    cudaMalloc(&ray_info_buffered_pixels, grid_size_x*grid_size_y*block_size_x*block_size_y*sizeof(ray_info_buffered));
    //printf("ray_info_buffered_pixels allocated for a size of %ld \n", grid_size_x*grid_size_y*block_size_x*block_size_y*sizeof(ray_info_buffered));
    ray_info_buffered* ray_info_buffered_pixels_midles;
    cudaMalloc(&ray_info_buffered_pixels_midles, grid_size_x*grid_size_y*sizeof(ray_info_buffered));
    //printf("ray_info_buffered_pixels_midles allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(ray_info_buffered));    
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

    //filter the gaussians
    int selected_blocks = 81;
    int selected_threads = 32;
    array_indexes_type ** selected_kernels;
    cudaMalloc(&selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
    //printf("selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
    //printf("allocated at %p \n", selected_kernels);

    array_indexes_type * stacks_sizes;
    cudaMalloc(&stacks_sizes, grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
    //printf("stacks_sizes allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
    //printf("allocated at %p \n", stacks_sizes);

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
    //printf("nb_selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type));
    array_indexes_type** result_selected_kernels;
    cudaMalloc(&result_selected_kernels, grid_size_x*grid_size_y*sizeof(array_indexes_type*));
    //printf("result_selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*));
    //launch compact_selected_kernels_2 on one SAMEW GRID AS RENDERING, on same block as rendering
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
    //LAUNCH __global__ free_select_gaussian_kernel_2
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


    //launch rendering

    #ifdef overlapping
        //overlapping
        render_screen_block_my_2_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
    #else
        //no overlapping
        render_screen_block_mix_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
    #endif
        
    
    cudaFree(shared_sorted_indexes__);

    cudaDeviceSynchronize();
    //cath the error
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    //std::cout << "screen rendered on gpu with selected kernels" << std::endl;
    cudaFree(result_selected_kernels);
    cudaFree(nb_selected_kernels);
    //copy the screen to the host
    cudaMemcpy(screen, screen_gpu2, width*height*sizeof(point3d), cudaMemcpyDeviceToHost);
    cudaFree(screen_gpu2);
    //std::cout << "screen copied to host" << std::endl;

    //show the screen openning a window
    cv::Mat img_gpu_selected(height, width, CV_8UC3, cv::Scalar(0,0,0));
    for(array_indexes_type i=0;i<height;i++){
        for(array_indexes_type j=0;j<width;j++){
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
            img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
        }
    }
    cv::imshow("image", img_gpu_selected);
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
        //printf("selected_kernels allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type*) * selected_blocks * selected_threads);
        //printf("allocated at %p \n", selected_kernels);

        cudaMalloc(&stacks_sizes, grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
        //printf("stacks_sizes allocated for a size of %ld \n", grid_size_x*grid_size_y*sizeof(array_indexes_type) * selected_blocks * selected_threads);
        //printf("allocated at %p \n", stacks_sizes);

        select_gaussian_kernel_2<<<selected_blocks,selected_threads>>>(v_kernels_gpu, N, width, height, src, dir_, selected_kernels, stacks_sizes, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x);
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        //std::cout << "selected_kernels computed" << std::endl;

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
        //std::cout << "compact_selected_kernels computed" << std::endl;
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
        //std::cout << "selected_kernels freed" << std::endl;
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

        #ifdef overlapping
            //overlapping
            render_screen_block_my_2_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_start_end**)shared_sorted_indexes__);
        #else
            //no overlapping
            render_screen_block_mix_selected_my_shared<<<grid, block>>>(screen_gpu2, src,dir_, v_kernels_gpu, width, height, result_selected_kernels, nb_selected_kernels, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, grid_size_x*block_size_x, (index_value**)shared_sorted_indexes__);
        #endif
        
        cudaDeviceSynchronize();
        //cath the error
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }
        cudaFree(shared_sorted_indexes__);
        //std::cout << "screen rendered on gpu with selected kernels" << std::endl;
        cudaFree(result_selected_kernels);
        cudaFree(nb_selected_kernels);
        //copy the screen to the host
        cudaMemcpy(screen, screen_gpu2, width*height*sizeof(point3d), cudaMemcpyDeviceToHost);
        cudaFree(screen_gpu2);
        //std::cout << "screen copied to host" << std::endl;

        //show the screen openning a window
        cv::Mat img_gpu_selected(height, width, CV_8UC3, cv::Scalar(0,0,0));
        for(array_indexes_type i=0;i<height;i++){
            for(array_indexes_type j=0;j<width;j++){
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[2] = (unsigned char)(screen[j*height+i].x*255);//RED
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[1] = (unsigned char)(screen[j*height+i].y*255);//GREEN
                img_gpu_selected.at<cv::Vec3b>(height-i-1,j)[0] = (unsigned char)(screen[j*height+i].z*255);//BLUE
            }
        }
        cv::imshow("image", img_gpu_selected);
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
