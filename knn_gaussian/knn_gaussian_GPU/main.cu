#define _USE_CUDA_
#include "../construct_tree/construct.hpp"
#include "../../common/common_header.hpp"
#include "def_functions.cuh"
#define Sort
int main(int argc, char *argv[]){
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
    array_indexes_type N;
    std::vector<gaussian_kernel2_3D> kernels3;
    //if an input file is given the input gaussian from the input file else generate random gaussians
    if(argc>1){
        std::cout<<"Reading from file: "<<argv[1]<<std::endl;
        std::ifstream infile(argv[1],std::ios_base::binary);
        std::string buff;
        std::getline(infile,buff);
        std::getline(infile,buff);

        std::string dummy;
        std::getline(infile,buff);
        std::stringstream ss(buff);
        ss>>dummy>>dummy>>N;

        std::cout<<"N: "<<N<<std::endl;
        
        while(std::getline(infile,buff)){
            if(buff.compare("end_header") == 0)
                break;
        }

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
                float dummy_;
                infile.read(reinterpret_cast<char*>(&dummy_), sizeof(float));
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

            kernels3.push_back(gaussian_kernel2_3D(mu, opacity, scale, q));
        }
    }else{
        std::cout<<"Generating random gaussians"<<std::endl;
        N = 1000000;
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
    }
    std::cout<<"Number of gaussians: "<<kernels3.size()<<std::endl;
    //construct the tree
    kd_tree3 tree3(&kernels3);
    kdtree_node3 root3(&kernels3);
    tree3.root = &root3;
    std::cout<<"Tree constructed"<<std::endl;
    array_indexes_type N_points = 1000000;

    float_double min_coord3[3] = {max_float_double,max_float_double,max_float_double};
    float_double max_coord3[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    min_coord3[0] = root3.range0;
    min_coord3[1] = root3.range1;
    min_coord3[2] = root3.range2;
    max_coord3[0] = root3.range3;
    max_coord3[1] = root3.range4;
    max_coord3[2] = root3.range5;

    std::vector<point3d> xs;
    for(array_indexes_type i=0;i<N_points;i++){
        point3d mu;
        mu.x= min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        mu.y= min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        mu.z= min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        xs.push_back(mu);
    }
    #ifdef Sort
        std::cout<<"Sorting the points using hilbert curve"<<std::endl;
        point3d min_ = point3d(min_coord3[0],min_coord3[1],min_coord3[2]);
        point3d max_ = point3d(max_coord3[0],max_coord3[1],max_coord3[2]);
        std::vector<std::pair<point3d, double>> xs_hilbert;
        for(array_indexes_type i=0;i<N_points;i++){
            xs_hilbert.push_back(std::make_pair(xs[i], xs[i].hilbert_curve_cord(min_, max_)));
        }
        std::sort(xs_hilbert.begin(), xs_hilbert.end(), [&](std::pair<point3d, double> a, std::pair<point3d, double> b) -> bool {return a.second < b.second;});
        for(array_indexes_type i=0;i<N_points;i++){
            xs[i] = xs_hilbert[i].first;
        }
        std::cout<<"Points sorted"<<std::endl;
    #endif
    point3d* xs_ = (point3d*)malloc(N_points*sizeof(point3d));
    for(array_indexes_type i=0;i<N_points;i++){
        xs_[i] = xs[i];
    }
    char* tree3_data_GPU;
    cudaMalloc(&tree3_data_GPU, root3.size);
    cudaMemcpy(tree3_data_GPU, root3.data, root3.size, cudaMemcpyHostToDevice);
    point3d* xs_GPU;
    cudaMalloc(&xs_GPU, N_points*sizeof(point3d));
    cudaMemcpy(xs_GPU, xs_, N_points*sizeof(point3d), cudaMemcpyHostToDevice);
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
    int blockSize,minGridSize,gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
        search1nn_arr_non_rec, 0, 0);
    int MaxGridSize;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&MaxGridSize, search1nn_arr_non_rec, blockSize, 0);
    gridSize = std::min((size_t)((N_points + blockSize - 1) / blockSize), (size_t)MaxGridSize);
    gridSize = 82*3;
    blockSize = 128;
    std::cout << "blockSize: " << blockSize << std::endl;
    std::cout << "minGridSize: " << minGridSize << std::endl;
    std::cout << "gridSize: " << gridSize << std::endl;
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, search1nn_arr_non_rec, blockSize, 0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                      (float)(props.maxThreadsPerMultiProcessor / 
                              props.warpSize);
    std::cout << "Occupancy: " << occupancy << std::endl;
    clock_t start, end;
//1nn search
    start = clock();
    //search1nn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);
    search1nn_many_arr_non_rec_2<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_GPU, min_dist_GPU, kernels_GPU, N_points);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    end = clock();
    std::cout<<"Elapsed time for 1nn : "<< (float)(end - start) / CLOCKS_PER_SEC<<std::endl;
    cudaMemcpy(ress, ress_GPU, N_points*sizeof(gaussian_kernel2_3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_, min_dist_GPU, sizeof(float_double)*N_points, cudaMemcpyDeviceToHost);

    free(ress);
    cudaFree(ress_GPU);
    free(min_dist_);
    cudaFree(min_dist_GPU);

//knn search
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
    float_double* min_dist_knn = (float_double*)malloc(sizeof(float_double)*N_points);
    for(array_indexes_type i=0;i<N_points;i++){
        min_dist_knn[i] = max_float_double;
    }
    float_double* min_dist_knn_GPU;
    cudaMalloc(&min_dist_knn_GPU, sizeof(float_double)*N_points);
    cudaMemcpy(min_dist_knn_GPU, min_dist_knn, sizeof(float_double)*N_points, cudaMemcpyHostToDevice);

    start = clock();
    search_knn_arr_non_rec<<<gridSize, blockSize>>>(tree3_data_GPU, xs_GPU, ress_knn_GPU, min_dist_knn_GPU, kernels_GPU, N_points, number_found_GPU);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    end = clock();
    std::cout<<"Elapsed time for knn with k = "<<K_knn<<" : "<< (float)(end - start) / CLOCKS_PER_SEC<<std::endl;

    cudaMemcpy(ress_knn, ress_knn_GPU, N_points*k*sizeof(array_indexes_type), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_dist_knn, min_dist_knn_GPU, sizeof(float_double)*N_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(number_found, number_found_GPU, sizeof(array_indexes_type)*N_points, cudaMemcpyDeviceToHost);

    free(ress_knn);
    cudaFree(ress_knn_GPU);
    free(min_dist_knn);
    cudaFree(min_dist_knn_GPU);
    free(number_found);
    cudaFree(number_found_GPU);

    free(kernels);
    cudaFree(kernels_GPU);
    free(xs_);
    cudaFree(xs_GPU);

    cudaFree(tree3_data_GPU);
    return 0;
}   