#define _USE_CUDA_
#include "../construct_tree/construct.hpp"
#include "../../common/common_header.hpp"
#include "def_functions.cuh"
#define Sort
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

    array_indexes_type N =10000;
    //test the kdtree3
    std::vector<gaussian_kernel2_3D> kernels3;
    for(array_indexes_type i=0;i<N;i++){
        point3d mu;
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
    
    //copy the tree array to the GPU
    char* tree3_data = (char*)malloc(root3.size);
    for(array_indexes_type i=0;i<root3.size;i++){
        tree3_data[i] = root3.data[i];
    }
    char* tree3_data_GPU;
    cudaMalloc(&tree3_data_GPU, root3.size);
    cudaMemcpy(tree3_data_GPU, tree3_data, root3.size, cudaMemcpyHostToDevice);

    //void search1nn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n)
    array_indexes_type N_points = 100000000;
    /*point3d* xs = (point3d*)malloc(N_points*sizeof(point3d));
    for(array_indexes_type i=0;i<N_points;i++){
        xs[i].x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        xs[i].y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        xs[i].z = 1000*((float_double)rand()/(float_double)RAND_MAX);
    }*/
    std::vector<point3d> xs_;
    for(array_indexes_type i=0;i<N_points;i++){
        point3d mu;
        mu.x= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.y= 1000*((float_double)rand()/(float_double)RAND_MAX);
        mu.z= 1000*((float_double)rand()/(float_double)RAND_MAX);
        xs_.push_back(mu);
    }
    point3d mini = point3d(0,0,0);
    point3d maxi = point3d(1000,1000,1000);
    #ifdef Sort
        std::cout << "start sort" << std::endl;
        //std::sort(xs_.begin(), xs_.end(), [&](point3d a, point3d b) -> bool {return a.hilbert_curve_cord(mini, maxi) < b.hilbert_curve_cord(mini, maxi);});
        std::vector<std::pair<point3d, double>> xs_hilbert;
        for(array_indexes_type i=0;i<N_points;i++){
            xs_hilbert.push_back(std::make_pair(xs_[i], xs_[i].hilbert_curve_cord(mini, maxi)));
        }
        std::cout << "start sort" << std::endl;
        std::sort(xs_hilbert.begin(), xs_hilbert.end(), [&](std::pair<point3d, double> a, std::pair<point3d, double> b) -> bool {return a.second < b.second;});
        std::cout << "end sort" << std::endl;
        for(array_indexes_type i=0;i<N_points;i++){
            xs_[i] = xs_hilbert[i].first;
        }
    #endif

    /*std::sort(xs_.begin(), xs_.end(), [&](point3d a, point3d b) -> bool {
        float_double a_h = (a.x/1000)*100 + ((a.y/1000)*100)*100 + ((a.z/1000)*100)*100*100;
        float_double b_h = (b.x/1000)*100 + ((b.y/1000)*100)*100 + ((b.z/1000)*100)*100*100;
        return a_h < b_h;
    });*/
    //std::sort(xs_.begin(), xs_.end(), [&](point3d a, point3d b) -> bool {return a.x < b.x;});
    point3d* xs = (point3d*)malloc(N_points*sizeof(point3d));
    for(array_indexes_type i=0;i<N_points;i++){
        xs[i] = xs_[i];
    }
    point3d* xs_GPU;
    cudaMalloc(&xs_GPU, N_points*sizeof(point3d));
    cudaMemcpy(xs_GPU, xs, N_points*sizeof(point3d), cudaMemcpyHostToDevice);

    gaussian_kernel2_3D* ress = (gaussian_kernel2_3D*)malloc(N_points*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_GPU;
    cudaMalloc(&ress_GPU, N_points*sizeof(gaussian_kernel2_3D));

    float_double* min_dist_ = (float_double*)malloc(sizeof(float_double)*N_points);
    for(array_indexes_type i=0;i<N_points;i++){
        min_dist_[i] = max_float_double;
    }
    float_double* min_dist_GPU;
    cudaMalloc(&min_dist_GPU, sizeof(float_double)*N_points);
    cudaMemcpy(min_dist_GPU, min_dist_, sizeof(float_double)*N_points, cudaMemcpyHostToDevice);

    gaussian_kernel2_3D* kernels = (gaussian_kernel2_3D*)malloc(N*sizeof(gaussian_kernel2_3D));
    for(array_indexes_type i=0;i<N;i++){
        kernels[i] = kernels3[i];
    }
    gaussian_kernel2_3D* kernels_GPU;
    cudaMalloc(&kernels_GPU, N*sizeof(gaussian_kernel2_3D));
    cudaMemcpy(kernels_GPU, kernels, N*sizeof(gaussian_kernel2_3D), cudaMemcpyHostToDevice);

    clock_t start, end;
    start = clock();
    //search1nn_arr_non_rec<<<1, 1024>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);
    //use the good number of threads according to the device
    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                     // maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 
  
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                        search1nn_arr_non_rec, 0, 0);
    
    int MaxGridSize;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&MaxGridSize, search1nn_arr_non_rec, blockSize, 0);
    gridSize = std::min((size_t)((N_points + blockSize - 1) / blockSize), (size_t)MaxGridSize);
    //gridSize = ((N_points + blockSize - 1) / blockSize);
    //blockSize = 500;
    gridSize = 82*3;
    blockSize = 128;
    std::cout << "blockSize: " << blockSize << std::endl;
    std::cout << "minGridSize: " << minGridSize << std::endl;
    std::cout << "gridSize: " << gridSize << std::endl;
    
    //show ocupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, search1nn_arr_non_rec, blockSize, 0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                      (float)(props.maxThreadsPerMultiProcessor / 
                              props.warpSize);
    std::cout << "Occupancy: " << occupancy << std::endl;

    


    search1nn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);

    //search1nn_arr_non_rec<<<1, 1>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);
    cudaDeviceSynchronize();



    //handle errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // print the CUDA error message and exit
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        std::cout << (void*)tree3_data_GPU << std::endl;
        std::cout << (void*)xs_GPU << std::endl;
        std::cout << (void*)ress_GPU << std::endl;
        std::cout << (void*)min_dist_GPU << std::endl;
        std::cout << (void*)kernels_GPU << std::endl;
        //exit(-1);
    }


    end = clock();
    
    cudaMemcpy(ress, ress_GPU, N_points*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_, min_dist_GPU, N_points*sizeof(float_double), cudaMemcpyDeviceToHost);
    
    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force
    /*array_indexes_type tr = 0;
    for(array_indexes_type i=0;i<N_points;i++){
        std::cout << i << std::endl;
        gaussian_kernel2_3D res;
        float_double min_dist = max_float_double;
        for(array_indexes_type j=0;j<N;j++){
            float_double dist = kernels[j].distance(xs[i]);
            if (dist < min_dist){
                min_dist = dist;
                res = kernels[j];
                tr = j;
            }
        }
        if (abs(min_dist - min_dist_[i])> 0.0001 && !(abs(min_dist+negligeable_val_when_exp) <0.001 && abs(min_dist_[i]- max_float_double)<0.001*max_float_double) && !(abs(min_dist_[i] +negligeable_val_when_exp) <0.001 && abs(min_dist- max_float_double)<0.001*max_float_double)){
            std::cout << min_dist << "!=" << min_dist_[i] << std::endl;
            std::cout <<abs(min_dist - min_dist_[i]) << std::endl;
            std::cout << abs(min_dist+negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist_[i]- max_float_double) << std::endl;
            std::cout << abs(min_dist_[i] +negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist- max_float_double) << std::endl;
            std::cout << ress[i] << "!=" << res << std::endl;
            std::cout << xs[i] << std::endl;
            //exit(1);
            break;
        }
    }*/
    


    std::cout << "Time for the search1nn_arr_non_rec random: " << (float)(end-start)/CLOCKS_PER_SEC << std::endl;

    //free the memory
    //free(tree3_data);
    //cudaFree(tree3_data_GPU);
    free(ress);
    cudaFree(ress_GPU);
    free(min_dist_);
    cudaFree(min_dist_GPU);
    std::cout << "END of uniform test" << std::endl;


    
    // //test the brute force on gpu
    // //void search1nn_brute_force(point3d* x, gaussian_kernel2_3D* ress, float_double* min_dist, gaussian_kernel2_3D* kernels, array_indexes_type n, array_indexes_type n_kernels)
    // gaussian_kernel2_3D* ress_bf = (gaussian_kernel2_3D*)malloc(N_points*sizeof(gaussian_kernel2_3D));
    // gaussian_kernel2_3D* ress_bf_GPU;
    // cudaMalloc(&ress_bf_GPU, N_points*sizeof(gaussian_kernel2_3D));

    // float_double* min_dist_bf = (float_double*)malloc(sizeof(float_double)*N_points);
    // for(array_indexes_type i=0;i<N_points;i++){
    //     min_dist_bf[i] = max_float_double;
    // }
    // float_double* min_dist_bf_GPU;
    // cudaMalloc(&min_dist_bf_GPU, sizeof(float_double)*N_points);
    // cudaMemcpy(min_dist_bf_GPU, min_dist_bf, sizeof(float_double)*N_points, cudaMemcpyHostToDevice);

    // clock_t start__, end__;
    // start__ = clock();
    // search1nn_brute_force<<<gridSize, blockSize>>>(xs_GPU, ress_bf_GPU, min_dist_bf_GPU,kernels_GPU , N_points, N);

    // cudaDeviceSynchronize();
    // end__ = clock();

    // cudaMemcpy(ress_bf, ress_bf_GPU, N_points*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    // cudaMemcpy(min_dist_bf, min_dist_bf_GPU, N_points*sizeof(float_double), cudaMemcpyDeviceToHost);
    // cudaFree(ress_bf_GPU);
    // cudaFree(min_dist_bf_GPU);

    // std::cout << "Gpu done" << std::endl;
    // std::cout << "Time for the search1nn_brute_force random: " << (float)(end__-start__)/CLOCKS_PER_SEC << std::endl;


    //test the kNN search
    //void search_knn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type k, array_indexes_type n, array_indexes_type* number_found_){
    
    array_indexes_type k = K_knn;
    array_indexes_type* number_found = (array_indexes_type*)malloc(sizeof(array_indexes_type)*N_points);
    for(array_indexes_type i=0;i<N_points;i++){
        number_found[i] = 0;
    }
    array_indexes_type* number_found_GPU;
    cudaMalloc(&number_found_GPU, sizeof(array_indexes_type)*N_points);
    cudaMemcpy(number_found_GPU, number_found, sizeof(array_indexes_type)*N_points, cudaMemcpyHostToDevice);

    array_indexes_type* ress_knn = (array_indexes_type*)malloc(N_points*k*sizeof(array_indexes_type));
    array_indexes_type* ress_knn_GPU;
    cudaMalloc(&ress_knn_GPU, N_points*k*sizeof(array_indexes_type));
    if(ress_knn_GPU == NULL){
        std::cout << "ress_knn_GPU is NULL" << std::endl;
    }
    if(ress_knn == NULL){
        std::cout << "ress_knn is NULL" << std::endl;
    }

    float_double* min_dist_knn = (float_double*)malloc(sizeof(float_double)*N_points);
    for(array_indexes_type i=0;i<N_points;i++){
        min_dist_knn[i] = max_float_double;
    }
    float_double* min_dist_knn_GPU;
    cudaMalloc(&min_dist_knn_GPU, sizeof(float_double)*N_points);
    cudaMemcpy(min_dist_knn_GPU, min_dist_knn, sizeof(float_double)*N_points, cudaMemcpyHostToDevice);


    clock_t start___, end___;
    start___ = clock();
    search_knn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_knn_GPU, min_dist_knn_GPU, kernels_GPU, N_points, number_found_GPU);

    cudaDeviceSynchronize();
    end___ = clock();

    cudaMemcpy(ress_knn, ress_knn_GPU, N_points*k*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_knn, min_dist_knn_GPU, N_points*sizeof(float_double), cudaMemcpyDeviceToHost);
    cudaMemcpy(number_found, number_found_GPU, N_points*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force
    /*for(array_indexes_type i=0;i<N_points;i++){
        std::cout << i << std::endl;
        //std::cout << number_found[i] << std::endl;
        //std::cout << min_dist_knn[i] << std::endl;
        //std::cout << kernels[ress_knn[i*k]] << std::endl;
        //std::cout << kernels[ress_knn[i*k]].distance(xs[i]) << std::endl;
        std::vector<float_double> min_dists_brute;
        for (array_indexes_type j=0;j<k;j++){
            min_dists_brute.push_back(kernels[j].distance(xs[i]));
        }
        //sort the vector
        std::sort(min_dists_brute.begin(), min_dists_brute.end());
        for(array_indexes_type j=k;j<N;j++){
            float_double dist = kernels[j].distance(xs[i]);
            //push the new distance
            min_dists_brute.push_back(dist);
            //sort the vector
            std::sort(min_dists_brute.begin(), min_dists_brute.end());
            //pop the last element
            min_dists_brute.pop_back();
        }
        //std::cout << min_dists_brute[0] << std::endl;
        bool error = false;
        for(array_indexes_type j=0;j<number_found[i];j++){
            float_double dist = kernels[ress_knn[i*k+j]].distance(xs[i]);
            if (abs(min_dists_brute[j] - dist)> 0.0001 && !(abs(min_dists_brute[j]+negligeable_val_when_exp) <0.001 && abs(dist- max_float_double)<0.001*max_float_double) && !(abs(dist +negligeable_val_when_exp) <0.001 && abs(min_dists_brute[j]- max_float_double)<0.001*max_float_double)){
                error = true;
                std::cout << min_dists_brute[j] << "!=" << dist << std::endl;
                for(array_indexes_type l=0;l<k;l++){
                    std::cout << kernels[ress_knn[i*k+l]] << std::endl;
                    std::cout << kernels[ress_knn[i*k+l]].distance(xs[i]) << std::endl;
                }
                std::cout << xs[i] << std::endl;
                for(array_indexes_type l=0;l<k;l++){
                    std::cout << min_dists_brute[l] << std::endl;
                }


            }
            break;
        }
        for(array_indexes_type j=number_found[i];j<k && ! error;j++){
            //if the brute force distance is not bigger than negligeable_val_when_exp
            if (min_dists_brute[j] < -negligeable_val_when_exp){
                error = true;
                std::cout << min_dists_brute[j] << "!=" << kernels[ress_knn[i*k+j]].distance(xs[i]) << std::endl;
            }
        }
        if (error){
            std::cout << "Error" << std::endl;
            break;
        }
    }*/

    std::cout << "Time for the search_knn_arr_non_rec random: " << (float)(end___-start___)/CLOCKS_PER_SEC << std::endl;

    //free the memory
    free(ress_knn);
    cudaFree(ress_knn_GPU);
    free(min_dist_knn);
    cudaFree(min_dist_knn_GPU);
    free(number_found);
    cudaFree(number_found_GPU);
    std::cout << "END of kNN uniform test" << std::endl;

    //test the 1nn by block

    gaussian_kernel2_3D* ress_block = (gaussian_kernel2_3D*)malloc(N_points*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_block_GPU;
    cudaMalloc(&ress_block_GPU, N_points*sizeof(gaussian_kernel2_3D));

    float_double* min_dist_block = (float_double*)malloc(sizeof(float_double)*N_points);
    for(array_indexes_type i=0;i<N_points;i++){
        min_dist_block[i] = max_float_double;
    }
    float_double* min_dist_block_GPU;
    cudaMalloc(&min_dist_block_GPU, sizeof(float_double)*N_points);
    cudaMemcpy(min_dist_block_GPU, min_dist_block, sizeof(float_double)*N_points, cudaMemcpyHostToDevice);


    //copy again the tree array to the GPU to be sure that the previous copy is clean

    cudaMemcpy(tree3_data_GPU, tree3_data, root3.size, cudaMemcpyHostToDevice);

    clock_t start_____, end_____;
    start_____ = clock();
    //void search1nn_many_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n){
   
    search1nn_many_arr_non_rec_2<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_block_GPU, min_dist_block_GPU, kernels_GPU, N_points);

    cudaDeviceSynchronize();
    end_____ = clock();
    //catch errors
    cudaError_t error_ = cudaGetLastError();
    if (error_ != cudaSuccess)
    {
        // print the CUDA error message and exit
        std::cerr << "CUDA error: " << cudaGetErrorString(error_) << std::endl;
        std::cout << (void*)tree3_data_GPU << std::endl;
        std::cout << (void*)xs_GPU << std::endl;
        std::cout << (void*)ress_block_GPU << std::endl;
        std::cout << (void*)min_dist_block_GPU << std::endl;
        std::cout << (void*)kernels_GPU << std::endl;
        //exit(-1);
    }

    cudaMemcpy(ress_block, ress_block_GPU, N_points*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_block, min_dist_block_GPU, N_points*sizeof(float_double), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force
    /*for(array_indexes_type i=0;i<N_points;i++){
        std::cout << i << std::endl;
        gaussian_kernel2_3D res;
        float_double min_dist = max_float_double;
        for(array_indexes_type j=0;j<N;j++){
            float_double dist = kernels[j].distance(xs[i]);
            if (dist < min_dist){
                min_dist = dist;
                res = kernels[j];
            }
        }
        if (abs(min_dist - min_dist_block[i])> 0.0001 && !(abs(min_dist+negligeable_val_when_exp) <0.001 && abs(min_dist_block[i]- max_float_double)<0.001*max_float_double) && !(abs(min_dist_block[i] +negligeable_val_when_exp) <0.001 && abs(min_dist- max_float_double)<0.001*max_float_double)){
            std::cout << min_dist << "!=" << min_dist_block[i] << std::endl;
            std::cout << min_dist_block[i] << " ???? " << ress_block[i].distance(xs[i]) << std::endl;
            std::cout <<abs(min_dist - min_dist_block[i]) << std::endl;
            std::cout << abs(min_dist+negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist_block[i]- max_float_double) << std::endl;
            std::cout << abs(min_dist_block[i] +negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist- max_float_double) << std::endl;
            std::cout << res << "!=" << ress_block[i] << std::endl;
            std::cout << xs[i] << std::endl;
            //exit(1);
            break;
        }
    }*/

    std::cout << "Time for the ssearch1nn_many_arr_non_rec random : " << (float)(end_____-start_____)/CLOCKS_PER_SEC << std::endl;


    //free the memory 
    free(ress_block);
    cudaFree(ress_block_GPU);
    free(min_dist_block);
    cudaFree(min_dist_block_GPU);
    std::cout << "END of 1nn by block test" << std::endl;
    //free the memory


    free(xs);
    //std::cout << "free xs" << std::endl;
    cudaFree(xs_GPU);
    //std::cout << "free xs_GPU" << std::endl;
    free(tree3_data);
    //std::cout << "free tree3_data" << std::endl;
    cudaFree(tree3_data_GPU);
    //std::cout << "free tree3_data_GPU" << std::endl;

    return 0;    
}