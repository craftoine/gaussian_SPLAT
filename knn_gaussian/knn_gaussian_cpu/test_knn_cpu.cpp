//knn with gaussian like kernel
#include "../construct_tree/construct.hpp"
#include "../common_header.hpp"
#include "def_functions.hpp"
int main(){
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
    //std::cout << tree3 <<std::endl;
    //make random points
    clock_t start_uniform_1_3, end_uniform_1_3, start_uniform_k_3, end_uniform_k_3, start_true_1_3, end_true_1_3, start_true_k_3, end_true_k_3;
    start_uniform_1_3 = clock();
    for (int i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = max_float_double;
        gaussian_kernel2_3D res;
        tree3.search_1nn(&res, x, &min_dist_kd);
        //std::cout << min_dist_kd << std::endl;
        /*
        //compute the distance by brute force to check
        float_double min_dist = kernels3[0].distance(x);
        array_indexes_type min_dist_ind = max_array_indexes_type;
        for(array_indexes_type j=1;j<N;j++){
            float_double dist = kernels3[j].distance(x);
            if (dist < min_dist){
                min_dist = dist;
                min_dist_ind = j;
            }
        }
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist_kd << " " << min_dist << std::endl;
        if ((abs(min_dist_kd - min_dist) > 0) && (!(min_dist_kd == max_float_double || min_dist_kd == -negligeable_val_when_exp) || !(min_dist == max_float_double || min_dist == -negligeable_val_when_exp))){
            std::cout << "error" << std::endl;
            std::cout << min_dist_kd << " " << min_dist << std::endl;
            std::cout << res << kernels3[min_dist_ind] << std::endl;
            std::cout << x << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels3[j] << std::endl;
                std::cout << kernels3[j].distance(x) << std::endl;
            }
            break;
        }
        else{
            if(std::isnan(min_dist_kd) || std::isnan(min_dist)){
                std::cout << "error" << std::endl;
                std::cout << min_dist_kd << " " << min_dist << std::endl;
                std::cout << res << kernels3[min_dist_ind] << std::endl;
                std::cout << x << std::endl;
                for(array_indexes_type j=0;j<N;j++){
                    std::cout << kernels3[j] << std::endl;
                    std::cout << kernels3[j].distance(x) << std::endl;
                }
                break;
            }
        }
        
        */
    }
    end_uniform_1_3 = clock();

    //search knn
    const array_indexes_type k3 = K_knn;
    start_uniform_k_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = 1000*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.y = 1000*((float_double)rand()/(float_double)RAND_MAX);
        x.z = 1000*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = max_float_double;
        gaussian_kernel2_3D* found[k3];
        array_indexes_type number_found = 0;
        tree3.search_knn(x, &min_dist, found, k3, &number_found);
        /*
        //compute the distance by brute force to check
        //std::cout << "brute force" << std::endl;
        std::vector<float_double> min_dist_brute;
        for (array_indexes_type j=0;j<k3;j++){
            min_dist_brute.push_back(kernels3[j].distance(x));
        }
        //sort the 2 vectors
        std::sort(min_dist_brute.begin(), min_dist_brute.end());
        for(array_indexes_type j=k3;j<N;j++){
            //put the new distance in the list
            float_double dist = kernels3[j].distance(x);
            //push the new distance
            min_dist_brute.push_back(dist);
            std::sort(min_dist_brute.begin(), min_dist_brute.end());
            min_dist_brute.pop_back();
        }
        
        //std::cout << "brute force done" << std::endl;
        //std::cout<< min_dist << " " << min_dist_brute[k3-1] << std::endl;
        //for(array_indexes_type j=0;j<k3;j++){
        //    std::cout << min_dist_brute[j] << std::endl;
        //}
        bool error = false;
        for(array_indexes_type j=0;j<number_found;j++){
            if (abs(found[j]->distance(x) - min_dist_brute[j]) > 0){
                error = true;
                std::cout << "error at " << j << std::endl;
                std::cout << found[j]->distance(x) << " " << min_dist_brute[j] << std::endl;
            }
        }
        for(array_indexes_type j=number_found;j<k3;j++){
            if(min_dist_brute[j] != std::min(max_float_double,-negligeable_val_when_exp)){
                error = true;
                std::cout << "error at " << j << std::endl;
            }
        }
        if (error){
            std::cout << "error" << std::endl;
            std::cout << "min_dist" << " " << "min_dist_brute" << std::endl;
            std::cout << min_dist << " " << min_dist_brute[k3-1] << std::endl;
            std::cout << "x"<<std::endl;
            std::cout << x << std::endl;
            std::cout << "number_found" << std::endl;
            std::cout << number_found << std::endl;
            std::cout << std::endl;
            std::cout <<  " " << "found[j]" << std::endl;
            std::cout << "min_dist_brute[j]" << " " << "found[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<k3;j++){
                std::cout  << " "<<found[j] << std::endl;
                std::cout << min_dist_brute[j] <<" "<< (*found)[j].distance(x) << std::endl;
            }
            std::cout << std::endl;
            std::cout << "kernels[j]" << " " << "kernels[j].distance(x)" << std::endl;
            for(array_indexes_type j=0;j<N;j++){
                std::cout << kernels3[j] << std::endl;
                std::cout << kernels3[j].distance(x) << std::endl;
            }
            break;
        }
        */
        
    }
    end_uniform_k_3 = clock();

    //load true gaussian values from a ply file
    //typical header:
    /*
    ...
    */
    std::vector<gaussian_kernel2_3D> kernels3_;
    std::ifstream infile3("input.ply", std::ios_base::binary);

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

        /*float_double mu[3];
        mu[0] = (float_double)x;
        mu[1] = (float_double)y;
        mu[2] = (float_double)z;*/
        point3d mu;
        mu.x = (float_double)x;
        mu.y = (float_double)y;
        mu.z = (float_double)z;

        kernels3_.push_back(gaussian_kernel2_3D(mu, opacity, scale, q));
        i3++;
    }
    std::cout << "kernels3 loaded" << std::endl;

    //create the tree
    kd_tree3 tree3_(&kernels3_);
    kdtree_node3 root3_(&kernels3_);
    tree3_.root = &root3_;
    float_double min_coord3[3] = {max_float_double,max_float_double,max_float_double};
    float_double max_coord3[3] = {std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min(),std::numeric_limits<float_double>::min()};
    /*for(dimension_type i=0;i<3;i++){
        min_coord3[i] = root3_.ranges[i];
        max_coord3[i] = root3_.ranges[i+3];
    }*/
    min_coord3[0] = root3_.range0;
    min_coord3[1] = root3_.range1;
    min_coord3[2] = root3_.range2;
    max_coord3[0] = root3_.range3;
    max_coord3[1] = root3_.range4;
    max_coord3[2] = root3_.range5;
    //std::cout << min_coord3[0] << " " << min_coord3[1] << " " << min_coord3[2] << std::endl;
    //std::cout << max_coord3[0] << " " << max_coord3[1] << " " << max_coord3[2] << std::endl;
    
    //search the 1nn
    float_double min_dist_ever_seen3 = max_float_double;
    point3d x_ever_seen3;
    start_true_1_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord3[j] + (max_coord3[j]-min_coord3[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist_kd = max_float_double;
        gaussian_kernel2_3D res;
        tree3_.search_1nn(&res, x, &min_dist_kd);
        /*//compare the distance with the one given by the other tree on the same point
        double min_dist = res.distance(x);
        gaussian_kernel<3,3> res_;
        tree_.search_1nn(&res_, x, &min_dist);
        std::cout << min_dist_kd << " " << min_dist << std::endl;
        */
    }
    end_true_1_3 = clock();
    std::cout << "end 1nn" << std::endl;
    /*std::cout << min_dist_ever_seen3 << std::endl;
    std::cout << x_ever_seen3 << std::endl;*/
    //search the knn
    const array_indexes_type k3_ = K_knn;
    start_true_k_3 = clock();
    for(array_indexes_type i=0;i<1000000;i++){
        //std::cout << i << std::endl;
        point3d x;
        /*for(dimension_type j=0;j<3;j++){
            x.x[j] = min_coord3[j] + (max_coord3[j]-min_coord3[j])*((float_double)rand()/(float_double)RAND_MAX);
        }*/
        x.x = min_coord3[0] + (max_coord3[0]-min_coord3[0])*((float_double)rand()/(float_double)RAND_MAX);
        x.y = min_coord3[1] + (max_coord3[1]-min_coord3[1])*((float_double)rand()/(float_double)RAND_MAX);
        x.z = min_coord3[2] + (max_coord3[2]-min_coord3[2])*((float_double)rand()/(float_double)RAND_MAX);
        float_double min_dist = max_float_double;
        gaussian_kernel2_3D* found[k3_];
        array_indexes_type number_found = 0;
        tree3_.search_knn(x, &min_dist, found, k3_, &number_found);
    }
    end_true_k_3 = clock();

    
    std::cout << "end" << std::endl;
    std::cout << " ignored evaluations of gaussian kernel less than " << exp(negligeable_val_when_exp) << std::endl;
    std::cout <<"time statistics" << std::endl;
    std::cout << "uniform 1nn 3 : " << (float_double)(end_uniform_1_3-start_uniform_1_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "uniform knn 3 : " << (float_double)(end_uniform_k_3-start_uniform_k_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true 1nn 3 : " << (float_double)(end_true_1_3-start_true_1_3)/CLOCKS_PER_SEC << std::endl;
    std::cout << "true knn 3 : " << (float_double)(end_true_k_3-start_true_k_3)/CLOCKS_PER_SEC << std::endl;


    return 0;
}