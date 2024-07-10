#define _USE_CUDA_
#include "../construct_tree/construct.hpp"
#include "../common_header.hpp"
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

    array_indexes_type N = 1774942;
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
    array_indexes_type N_points = 100000;
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

    //test with true data


    //load true gaussian values from a ply file
    //typical header:
    /*
    ...
    */
    std::vector<gaussian_kernel2_3D> kernels3_;
    std::ifstream infile3("../knn_gaussian_cpu/input.ply", std::ios::binary);

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

        point3d mu;
        mu.x = (float_double)x;
        mu.y = (float_double)y;
        mu.z = (float_double)z;

        kernels3_.push_back(gaussian_kernel2_3D(mu, opacity, scale, q));
        i3++;
    }
    std::cout << "kernels3 loaded" << std::endl;
    std::cout << kernels3_.size() << std::endl;
    //create the tree
    kd_tree3 tree3_(&kernels3_);
    kdtree_node3 root3_(&kernels3_);
    tree3_.root = &root3_;
    float_double min_coord3[3] = {max_float_double,max_float_double,max_float_double};
    float_double max_coord3[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    min_coord3[0] = root3_.range0;
    min_coord3[1] = root3_.range1;
    min_coord3[2] = root3_.range2;
    max_coord3[0] = root3_.range3;
    max_coord3[1] = root3_.range4;
    max_coord3[2] = root3_.range5;
    //std::cout << min_coord3[0] << " " << min_coord3[1] << " " << min_coord3[2] << std::endl;
    //std::cout << max_coord3[0] << " " << max_coord3[1] << " " << max_coord3[2] << std::endl;
    

    //copy the tree array to the GPU
    char* tree3_data_ = (char*)malloc(root3_.size);
    for(array_indexes_type i=0;i<root3_.size;i++){
        tree3_data_[i] = root3_.data[i];
    }
    char* tree3_data_GPU_;
    cudaMalloc(&tree3_data_GPU_, root3_.size);
    cudaMemcpy(tree3_data_GPU_, tree3_data_, root3_.size, cudaMemcpyHostToDevice);

    array_indexes_type N_points_ = N_points;

    //void search1nn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n)
    /*point3d* xs_ = (point3d*)malloc(N_points_*sizeof(point3d));
    for(array_indexes_type i=0;i<N_points_;i++){
        xs_[i].x = min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        xs_[i].y = min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        xs_[i].z = min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
    }
    point3d* xs_GPU_;
    cudaMalloc(&xs_GPU_, N_points_*sizeof(point3d));
    cudaMemcpy(xs_GPU_, xs_, N_points_*sizeof(point3d), cudaMemcpyHostToDevice);*/

    std::vector<point3d> xs__;
    for(array_indexes_type i=0;i<N_points_;i++){
        point3d mu;
        mu.x= min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        mu.y= min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        mu.z= min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        xs__.push_back(mu);
    }
    #ifdef Sort
        //std::sort(xs__.begin(), xs__.end(), [&](point3d a, point3d b) -> bool {return a.x < b.x;});
        point3d min_ = point3d(min_coord3[0],min_coord3[1],min_coord3[2]);
        point3d max_ = point3d(max_coord3[0],max_coord3[1],max_coord3[2]);
        //std::sort(xs__.begin(), xs__.end(), [&](point3d a, point3d b) -> bool {return a.hilbert_curve_cord(min_, max_) < b.hilbert_curve_cord(min_, max_);});
        std::vector<std::pair<point3d, double>> xs_hilbert_;
        for(array_indexes_type i=0;i<N_points_;i++){
            xs_hilbert_.push_back(std::make_pair(xs__[i], xs__[i].hilbert_curve_cord(min_, max_)));
        }
        std::sort(xs_hilbert_.begin(), xs_hilbert_.end(), [&](std::pair<point3d, double> a, std::pair<point3d, double> b) -> bool {return a.second < b.second;});
        for(array_indexes_type i=0;i<N_points_;i++){
            xs__[i] = xs_hilbert_[i].first;
        }
    #endif
    point3d* xs2_ = (point3d*)malloc(N_points_*sizeof(point3d));
    for(array_indexes_type i=0;i<N_points_;i++){
        xs2_[i] = xs__[i];
    }
    point3d* xs_GPU_;
    cudaMalloc(&xs_GPU_, N_points_*sizeof(point3d));
    cudaMemcpy(xs_GPU_, xs2_, N_points_*sizeof(point3d), cudaMemcpyHostToDevice);

    gaussian_kernel2_3D* ress_ = (gaussian_kernel2_3D*)malloc(N_points_*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_GPU_;
    cudaMalloc(&ress_GPU_, N_points_*sizeof(gaussian_kernel2_3D));

    float_double* min_dist__ = (float_double*)malloc(sizeof(float_double)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        min_dist__[i] = max_float_double;
    }
    float_double* min_dist_GPU_;
    cudaMalloc(&min_dist_GPU_, sizeof(float_double)*N_points_);
    cudaMemcpy(min_dist_GPU_, min_dist__, sizeof(float_double)*N_points_, cudaMemcpyHostToDevice);

    gaussian_kernel2_3D* kernels_ = (gaussian_kernel2_3D*)malloc(count3*sizeof(gaussian_kernel2_3D));
    for(array_indexes_type i=0;i<count3;i++){
        kernels_[i] = kernels3_[i];
    }
    gaussian_kernel2_3D* kernels_GPU_;
    cudaMalloc(&kernels_GPU_, count3*sizeof(gaussian_kernel2_3D));
    cudaMemcpy(kernels_GPU_, kernels_, count3*sizeof(gaussian_kernel2_3D), cudaMemcpyHostToDevice);

    clock_t start_, end_;
    start_ = clock();
    search1nn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU_, xs_GPU_, ress_GPU_, min_dist_GPU_, kernels_GPU_, N_points_);

    //search1nn_arr_non_rec<<<numBlocks, blockSize>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);
    cudaDeviceSynchronize();
    end_ = clock();

    cudaMemcpy(ress_, ress_GPU_, N_points_*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist__, min_dist_GPU_, N_points_*sizeof(float_double), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force*/
    /*for(array_indexes_type i=0;i<N_points;i++){
        std::cout << i << std::endl;
        gaussian_kernel2_3D res;
        float_double min_dist = max_float_double;
        for(array_indexes_type j=0;j<count3;j++){
            float_double dist = kernels3_[j].distance(xs2_[i]);
            if (dist < min_dist){
                min_dist = dist;
                res = kernels3_[j];
            }
        }
        if (abs(min_dist - min_dist__[i])> 0.0001 && !(abs(min_dist+negligeable_val_when_exp) <0.001 && abs(min_dist__[i]- max_float_double)<0.001*max_float_double) && !(abs(min_dist__[i] +negligeable_val_when_exp) <0.001 && abs(min_dist- max_float_double)<0.001*max_float_double)){
            std::cout << min_dist << "!=" << min_dist__[i] << std::endl;
            std::cout <<abs(min_dist - min_dist__[i]) << std::endl;
            std::cout << abs(min_dist+negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist__[i]- max_float_double) << std::endl;
            std::cout << abs(min_dist__[i] +negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist- max_float_double) << std::endl;
            exit(1);
        }
    }*/

    std::cout << "Time for the search1nn_arr_non_rec: " << (float)(end_-start_)/CLOCKS_PER_SEC << std::endl;

    //free the memory
    free(ress_);
    cudaFree(ress_GPU_);
    free(min_dist__);
    cudaFree(min_dist_GPU_);
    std::cout << "END of true test" << std::endl;

    /*
    //test the brute force on gpu
    //void search1nn_brute_force(point3d* x, gaussian_kernel2_3D* ress, float_double* min_dist, gaussian_kernel2_3D* kernels, array_indexes_type n, array_indexes_type n_kernels)
    gaussian_kernel2_3D* ress_bf = (gaussian_kernel2_3D*)malloc(N_points_*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_bf_GPU;
    cudaMalloc(&ress_bf_GPU, N_points_*sizeof(gaussian_kernel2_3D));

    float_double* min_dist_bf = (float_double*)malloc(sizeof(float_double)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        min_dist_bf[i] = max_float_double;
    }
    float_double* min_dist_bf_GPU;
    cudaMalloc(&min_dist_bf_GPU, sizeof(float_double)*N_points_);
    cudaMemcpy(min_dist_bf_GPU, min_dist_bf, sizeof(float_double)*N_points_, cudaMemcpyHostToDevice);

    clock_t start__, end__;
    start__ = clock();
    search1nn_brute_force<<<gridSize, blockSize>>>(xs_GPU, ress_bf_GPU, min_dist_bf_GPU,kernels_GPU , N_points_, count3);

    cudaDeviceSynchronize();
    end__ = clock();

    cudaMemcpy(ress_bf, ress_bf_GPU, N_points_*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_bf, min_dist_bf_GPU, N_points_*sizeof(float_double), cudaMemcpyDeviceToHost);
    cudaFree(ress_bf_GPU);
    cudaFree(min_dist_bf_GPU);

    std::cout << "Gpu done" << std::endl;
    std::cout << "Time for the search1nn_brute_force random: " << (float)(end__-start__)/CLOCKS_PER_SEC << std::endl;

    //test the brute force on gpu
    //void search1nn_brute_force(point3d* x, gaussian_kernel2_3D* ress, float_double* min_dist, gaussian_kernel2_3D* kernels, array_indexes_type n, array_indexes_type n_kernels)
    gaussian_kernel2_3D* ress_bf_ = (gaussian_kernel2_3D*)malloc(N_points_*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_bf_GPU_;
    cudaMalloc(&ress_bf_GPU_, N_points_*sizeof(gaussian_kernel2_3D));

    float_double* min_dist_bf_ = (float_double*)malloc(sizeof(float_double)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        min_dist_bf_[i] = max_float_double;
    }
    float_double* min_dist_bf_GPU_;
    cudaMalloc(&min_dist_bf_GPU_, sizeof(float_double)*N_points_);
    cudaMemcpy(min_dist_bf_GPU_, min_dist_bf_, sizeof(float_double)*N_points_, cudaMemcpyHostToDevice);

    clock_t start___, end___;
    start___ = clock();
    search1nn_brute_force<<<gridSize, blockSize>>>(xs_GPU_, ress_bf_GPU_, min_dist_bf_GPU_,kernels_GPU_ , N_points_, count3);

    cudaDeviceSynchronize();
    end___ = clock();

    cudaMemcpy(ress_bf_, ress_bf_GPU_, N_points_*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_bf_, min_dist_bf_GPU_, N_points_*sizeof(float_double), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    std::cout << "Time for the search1nn_brute_force: " << (float)(end___-start___)/CLOCKS_PER_SEC << std::endl;

    cudaFree(ress_bf_GPU_);
    cudaFree(min_dist_bf_GPU_);

    std::cout << "END of brute force test" << std::endl;
    */



    //test the kNN search
    //void search_knn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type k, array_indexes_type n, array_indexes_type* number_found_){
    
    array_indexes_type k = K_knn;
    array_indexes_type* number_found = (array_indexes_type*)malloc(sizeof(array_indexes_type)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        number_found[i] = 0;
    }
    array_indexes_type* number_found_GPU;
    cudaMalloc(&number_found_GPU, sizeof(array_indexes_type)*N_points_);
    cudaMemcpy(number_found_GPU, number_found, sizeof(array_indexes_type)*N_points_, cudaMemcpyHostToDevice);

    array_indexes_type* ress_knn = (array_indexes_type*)malloc(N_points_*k*sizeof(array_indexes_type));
    array_indexes_type* ress_knn_GPU;
    cudaMalloc(&ress_knn_GPU, N_points_*k*sizeof(array_indexes_type));
    if(ress_knn_GPU == NULL){
        std::cout << "ress_knn_GPU is NULL" << std::endl;
    }
    if(ress_knn == NULL){
        std::cout << "ress_knn is NULL" << std::endl;
    }

    float_double* min_dist_knn = (float_double*)malloc(sizeof(float_double)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        min_dist_knn[i] = max_float_double;
    }
    float_double* min_dist_knn_GPU;
    cudaMalloc(&min_dist_knn_GPU, sizeof(float_double)*N_points_);
    cudaMemcpy(min_dist_knn_GPU, min_dist_knn, sizeof(float_double)*N_points_, cudaMemcpyHostToDevice);


    clock_t start___, end___;
    start___ = clock();
    search_knn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_knn_GPU, min_dist_knn_GPU, kernels_GPU, N_points_, number_found_GPU);

    cudaDeviceSynchronize();
    end___ = clock();

    cudaMemcpy(ress_knn, ress_knn_GPU, N_points_*k*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_knn, min_dist_knn_GPU, N_points_*sizeof(float_double), cudaMemcpyDeviceToHost);
    cudaMemcpy(number_found, number_found_GPU, N_points_*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);

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

    //test with true data
    //void search_knn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type k, array_indexes_type n, array_indexes_type* number_found_){
    array_indexes_type k_ = K_knn;
    array_indexes_type* number_found_ = (array_indexes_type*)malloc(sizeof(array_indexes_type)*N_points_);
    for(array_indexes_type i=0;i<N_points_;i++){
        number_found_[i] = 0;
    }
    array_indexes_type* number_found_GPU_;
    cudaMalloc(&number_found_GPU_, sizeof(array_indexes_type)*N_points_);
    cudaMemcpy(number_found_GPU_, number_found_, sizeof(array_indexes_type)*N_points_, cudaMemcpyHostToDevice);


    array_indexes_type N_points__ = N_points_;
    array_indexes_type* ress_knn_ = (array_indexes_type*)malloc(N_points__*k_*sizeof(array_indexes_type));
    array_indexes_type* ress_knn_GPU_;
    cudaMalloc(&ress_knn_GPU_, N_points__*k_*sizeof(array_indexes_type));

    float_double* min_dist_knn_ = (float_double*)malloc(sizeof(float_double)*N_points__);
    for(array_indexes_type i=0;i<N_points__;i++){
        min_dist_knn_[i] = max_float_double;
    }
    float_double* min_dist_knn_GPU_;
    cudaMalloc(&min_dist_knn_GPU_, sizeof(float_double)*N_points__);
    cudaMemcpy(min_dist_knn_GPU_, min_dist_knn_, sizeof(float_double)*N_points__, cudaMemcpyHostToDevice);

    clock_t start____, end____;
    start____ = clock();
    search_knn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU_, xs_GPU_, ress_knn_GPU_, min_dist_knn_GPU_, kernels_GPU_, N_points__, number_found_GPU_);

    cudaDeviceSynchronize();
    end____ = clock();

    cudaMemcpy(ress_knn_, ress_knn_GPU_, N_points__*k_*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_knn_, min_dist_knn_GPU_, N_points__*sizeof(float_double), cudaMemcpyDeviceToHost);
    cudaMemcpy(number_found_, number_found_GPU_, N_points__*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force
    /*for(array_indexes_type i=0;i<N_points_;i++){
        //std::cout << i << std::endl;
        //std::cout << number_found_[i] << std::endl;
        //std::cout << min_dist_knn_[i] << std::endl;
        //std::cout << kernels[ress_knn_[i*k_]] << std::endl;
        //std::cout << kernels[ress_knn_[i*k_]].distance(xs2_[i]) << std::endl;
        std::vector<float_double> min_dists_brute;
        for (array_indexes_type j=0;j<k_;j++){
            min_dists_brute.push_back(kernels_[j].distance(xs2_[i]));
        }
        //sort the vector
        std::sort(min_dists_brute.begin(), min_dists_brute.end());
        for(array_indexes_type j=k_;j<count3;j++){
            float_double dist = kernels_[j].distance(xs2_[i]);
            //push the new distance
            min_dists_brute.push_back(dist);
            //sort the vector
            std::sort(min_dists_brute.begin(), min_dists_brute.end());
            //pop the last element
            min_dists_brute.pop_back();
        }
        //std::cout << min_dists_brute[0] << std::endl;
        bool error = false;
        for(array_indexes_type j=0;j<number_found_[i];j++){
            float_double dist = kernels_[ress_knn_[i*k_+j]].distance(xs2_[i]);
            if (abs(min_dists_brute[j] - dist)> 0.0001 && !(abs(min_dists_brute[j]+negligeable_val_when_exp) <0.001 && abs(dist- max_float_double)<0.001*max_float_double) && !(abs(dist +negligeable_val_when_exp) <0.001 && abs(min_dists_brute[j]- max_float_double)<0.001*max_float_double)){
                error = true;
                std::cout << min_dists_brute[j] << "!=" << dist << std::endl;
                for(array_indexes_type l=0;l<k_;l++){
                    std::cout << kernels_[ress_knn_[i*k_+l]] << std::endl;
                    std::cout << kernels_[ress_knn_[i*k_+l]].distance(xs2_[i]) << std::endl;
                }
                std::cout << xs2_[i] << std::endl;
                for(array_indexes_type l=0;l<k_;l++){
                    std::cout << min_dists_brute[l] << std::endl;
                }
                
            }
            break;
        }
        for(array_indexes_type j=number_found_[i];j<k_ && ! error;j++){
            //if the brute force distance is not bigger than negligeable_val_when_exp
            if (min_dists_brute[j] < -negligeable_val_when_exp){
                error = true;
                std::cout << min_dists_brute[j] << "!=" << kernels_[ress_knn_[i*k_+j]].distance(xs2_[i]) << std::endl;
            }
        }
        if (error){
            std::cout << "Error" << std::endl;
            break;
        }
    }*/

    std::cout << "Time for the search_knn_arr_non_rec: " << (float)(end____-start____)/CLOCKS_PER_SEC << std::endl;
    
    //free the memory
    free(ress_knn_);
    cudaFree(ress_knn_GPU_);
    free(min_dist_knn_);
    cudaFree(min_dist_knn_GPU_);
    free(number_found_);
    cudaFree(number_found_GPU_);
    std::cout << "END of kNN true test" << std::endl;


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

    //test the 1nn by block with true data
    //void search1nn_many_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n){
    
    gaussian_kernel2_3D* ress_block_ = (gaussian_kernel2_3D*)malloc(N_points__*sizeof(gaussian_kernel2_3D));
    gaussian_kernel2_3D* ress_block_GPU_;
    cudaMalloc(&ress_block_GPU_, N_points__*sizeof(gaussian_kernel2_3D));

    float_double* min_dist_block_ = (float_double*)malloc(sizeof(float_double)*N_points__);
    for(array_indexes_type i=0;i<N_points__;i++){
        min_dist_block_[i] = max_float_double;
    }
    float_double* min_dist_block_GPU_;
    cudaMalloc(&min_dist_block_GPU_, sizeof(float_double)*N_points__);
    cudaMemcpy(min_dist_block_GPU_, min_dist_block_, sizeof(float_double)*N_points__, cudaMemcpyHostToDevice);

    clock_t start______, end______;
    start______ = clock();
    search1nn_many_arr_non_rec_2<<<gridSize, blockSize>>>(tree3_data_GPU_, xs_GPU_, ress_block_GPU_, min_dist_block_GPU_, kernels_GPU_, N_points__);

    cudaDeviceSynchronize();
    end______ = clock();

    cudaMemcpy(ress_block_, ress_block_GPU_, N_points__*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_block_, min_dist_block_GPU_, N_points__*sizeof(float_double), cudaMemcpyDeviceToHost);

    std::cout << "Gpu done" << std::endl;
    //check the results by testing comparing with brute force
    /*for(array_indexes_type i=0;i<N_points__;i++){
        std::cout << i << std::endl;
        gaussian_kernel2_3D res;
        float_double min_dist = max_float_double;
        for(array_indexes_type j=0;j<count3;j++){
            float_double dist = kernels_[j].distance(xs2_[i]);
            if (dist < min_dist){
                min_dist = dist;
                res = kernels_[j];
            }
        }
        if (abs(min_dist - min_dist_block_[i])> 0.0001 && !(abs(min_dist+negligeable_val_when_exp) <0.001 && abs(min_dist_block_[i]- max_float_double)<0.001*max_float_double) && !(abs(min_dist_block_[i] +negligeable_val_when_exp) <0.001 && abs(min_dist- max_float_double)<0.001*max_float_double)){
            std::cout << min_dist << "!=" << min_dist_block_[i] << std::endl;
            std::cout <<abs(min_dist - min_dist_block_[i]) << std::endl;
            std::cout << abs(min_dist+negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist_block_[i]- max_float_double) << std::endl;
            std::cout << abs(min_dist_block_[i] +negligeable_val_when_exp) << std::endl;
            std::cout << abs(min_dist- max_float_double) << std::endl;
            std::cout << res << "!=" << ress_block_[i] << std::endl;
            std::cout << xs2_[i] << std::endl;
            //exit(1);
            break;
        }
    }*/

    std::cout << "Time for the search1nn_many_arr_non_rec: " << (float)(end______-start______)/CLOCKS_PER_SEC << std::endl;








    //free the memory


    free(xs);
    //std::cout << "free xs" << std::endl;
    cudaFree(xs_GPU);
    //std::cout << "free xs_GPU" << std::endl;
    free(tree3_data);
    //std::cout << "free tree3_data" << std::endl;
    cudaFree(tree3_data_GPU);
    //std::cout << "free tree3_data_GPU" << std::endl;


    free(xs2_);
    //std::cout << "free xs2_" << std::endl;
    cudaFree(xs_GPU_);
    //std::cout << "free xs_GPU_" << std::endl;
    free(tree3_data_);
    //std::cout << "free tree3_data_" << std::endl;
    cudaFree(tree3_data_GPU_);
    //std::cout << "free tree3_data_GPU_" << std::endl;

    return 0;    
}