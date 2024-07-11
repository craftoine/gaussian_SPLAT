
#include "../../common/common_define.hpp"
#include "../../common/common_class.hpp"



#define Simple_heuristic true
#define sigma_heuristic true
#define sigma_split_midle false

#define max_sigma_factor 150.
#define max_depth 27
#define max_leaf_size 1
#define min_node_size 1

#define K_knn 20

#define Size_point_block 2


#if max_leaf_size > 1
    struct leaf{
        array_indexes_type is_leaf;
        array_indexes_type start;
        array_indexes_type end;
    };
#else
    struct 
    #ifdef _USE_CUDA_
        __align__(16)
    #endif
    leaf{
        array_indexes_type is_leaf;
        array_indexes_type start;
    };
#endif
struct range_data{
    float_double range0;
    float_double range1;
    float_double range2;
    float_double range3;
    float_double range4;
    float_double range5;
    float_double range6;
    float_double range7;
};

struct 
#ifdef _USE_CUDA_
    __align__(16)
#endif
    node{
    array_indexes_type right;
    range_data left_range;
    range_data right_range;
};
    








class kdtree_node3{
    public:
    array_indexes_type size;
    char* data;
    float_double range0;
    float_double range1;
    float_double range2;
    float_double range3;
    float_double range4;
    float_double range5;
    float_double range6;
    float_double range7;
    kdtree_node3(size_t size, char* data, float_double ranges0, float_double ranges1, float_double ranges2, float_double ranges3, float_double ranges4, float_double ranges5, float_double ranges6, float_double ranges7){
        this->size = size;
        this->data = data;
        this->range0 = ranges0;
        this->range1 = ranges1;
        this->range2 = ranges2;
        this->range3 = ranges3;
        this->range4 = ranges4;
        this->range5 = ranges5;
        this->range6 = ranges6;
        this->range7 = ranges7;
    }

    kdtree_node3(){
        size = 0;
        data = nullptr;
        range0 = 0;
        range1 = 0;
        range2 = 0;
        range3 = 0;
        range4 = 0;
        range5 = 0;
        range6 = 0;
        range7 = 0;
    
    }
    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type start, array_indexes_type end, dimension_type split_dim, array_indexes_type depth=0){
        //std::cout << start << " " << end << std::endl;
        if(start<end){
            build(ks, start, end, split_dim, depth);
        }
        else{
            build_leaf(start, end, ks);
        }          
    }

    kdtree_node3(std::vector<gaussian_kernel2_3D>* ks, array_indexes_type depth=0){
        if(ks->size()>1){
            build(ks, 0, ks->size()-1, 0, depth);
        }
        else{
            build_leaf(0, 0, ks);
        }
    }

    void build(std::vector<gaussian_kernel2_3D>* kernels, array_indexes_type start, array_indexes_type end, dimension_type split_dim, array_indexes_type depth=0){
        if(depth >= max_depth || (1+end)-start <= min_node_size){
            //std::cout << "leaf" << std::endl;
            //std::cout << (1+end)-start << " " << min_node_size << std::endl;
            //std::cout << depth << " " << max_depth << std::endl;
            build_leaf(start, end, kernels);
            return;
        }
        if(sigma_heuristic){
            float_double max_sigma = std::numeric_limits<float_double>::min();
            float_double min_sigma = max_float_double;
            
            for(array_indexes_type i=start;i<=end;i++){
                float_double sigma_i = abs((*kernels)[i].scales3.x);
                sigma_i = std::max(sigma_i, abs((*kernels)[i].scales3.y));
                sigma_i = std::max(sigma_i, abs((*kernels)[i].scales3.z));
                max_sigma = std::max(max_sigma, sigma_i);
                min_sigma = std::min(min_sigma, sigma_i);                
            }
            if (max_sigma/min_sigma > max_sigma_factor){
                float_double sigma_midle = (max_sigma+min_sigma)/2;
                //sort the kernels by sigma and plit in two according to the sigma_midle
                std::sort(kernels->begin()+start, kernels->begin()+end+1, [](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
                    float_double sigma1 = std::max(std::max(abs(g1.scales3.x), abs(g1.scales3.y)), abs(g1.scales3.z));
                    float_double sigma2 = std::max(std::max(abs(g2.scales3.x), abs(g2.scales3.y)), abs(g2.scales3.z));
                    return sigma1 < sigma2;
                });
                #if sigma_split_midle
                    array_indexes_type new_size = (end-start)/2;
                    array_indexes_type left_end = start + new_size;
                    array_indexes_type split = left_end + 1;
                #else
                    array_indexes_type split = start;
                    for(array_indexes_type i=start;i<=end;i++){
                        float_double sigma_i = std::max(std::max(abs((*kernels)[i].scales3.x), abs((*kernels)[i].scales3.y)), abs((*kernels)[i].scales3.z));
                        if (sigma_i > sigma_midle){
                            split = i;
                            break;
                        }
                    }
                #endif
                kdtree_node3 left = kdtree_node3(kernels, start, split-1, (split_dim+1)%3, depth+1);
                kdtree_node3 right = kdtree_node3(kernels, split, end, (split_dim+1)%3, depth+1);
                //array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char) + 2*(3+3+1+1)*sizeof(float_double)/sizeof(char) + left.size + right.size;
                array_indexes_type size = sizeof(node) + left.size + right.size;
                //char* data = (char*)malloc(size);
                char* data = new char[size];
                const unsigned char size_local = sizeof(node);
                ((node *) data)->left_range.range0 = left.range0;
                ((node *) data)->left_range.range1 = left.range1;
                ((node *) data)->left_range.range2 = left.range2;
                ((node *) data)->left_range.range3 = left.range3;
                ((node *) data)->left_range.range4 = left.range4;
                ((node *) data)->left_range.range5 = left.range5;
                ((node *) data)->left_range.range6 = left.range6;
                ((node *) data)->left_range.range7 = left.range7;
                ((node *) data)->right = size_local+left.size;
                ((node *) data)->right_range.range0 = right.range0;
                ((node *) data)->right_range.range1 = right.range1;
                ((node *) data)->right_range.range2 = right.range2;
                ((node *) data)->right_range.range3 = right.range3;
                ((node *) data)->right_range.range4 = right.range4;
                ((node *) data)->right_range.range5 = right.range5;
                ((node *) data)->right_range.range6 = right.range6;
                ((node *) data)->right_range.range7 = right.range7;
                for(array_indexes_type i=0;i<left.size;i++){
                    data[size_local+i] = left.data[i];
                }
                const array_indexes_type ofset4 = size_local + left.size;
                for(array_indexes_type i=0;i<right.size;i++){
                    data[ofset4+i] = right.data[i];
                }
                this->size = size;
                this->data = data;
                range0 = std::min(left.range0, right.range0);
                range1 = std::min(left.range1, right.range1);
                range2 = std::min(left.range2, right.range2);
                range3 = std::max(left.range3, right.range3);
                range4 = std::max(left.range4, right.range4);
                range5 = std::max(left.range5, right.range5);
                range6 = std::max(left.range6, right.range6);
                range7 = std::max(left.range7, right.range7);
                if(left.data != nullptr){
                    free(left.data);
                    left.data = nullptr;
                }
                if(right.data != nullptr){
                    free(right.data);
                    right.data = nullptr;
                }
                return;
            }
        }
        std::sort(kernels->begin()+start, kernels->begin()+end+1, [&split_dim](gaussian_kernel2_3D g1, gaussian_kernel2_3D g2){
            return g1.mu[split_dim] < g2.mu[split_dim];
        });
        kdtree_node3 left;
        kdtree_node3 right;
        if(Simple_heuristic){
            array_indexes_type new_size = (end-start)/2;
            array_indexes_type left_end = start + new_size;
            array_indexes_type right_start = left_end + 1;
            left = kdtree_node3(kernels, start, left_end, (split_dim+1)%3, depth+1);
            right = kdtree_node3(kernels, right_start, end, (split_dim+1)%3, depth+1);
        }
        else{
            point3d midle;
            float_double max_dim[3];
            float_double min_dim[3];
            for(dimension_type i=0;i<3;i++){
                max_dim[i] = std::numeric_limits<float_double>::min();
                min_dim[i] = max_float_double;
            }
            
            for(array_indexes_type j=start;j<=end;j++){
                for(dimension_type i=0;i<3;i++){
                    max_dim[i] = std::max(max_dim[i], (*kernels)[j].mu[i]);
                    min_dim[i] = std::min(min_dim[i], (*kernels)[j].mu[i]);
                }
            }
            midle.x = (max_dim[0]+min_dim[0])/2;
            midle.y = (max_dim[1]+min_dim[1])/2;
            midle.z = (max_dim[2]+min_dim[2])/2;
            array_indexes_type split = 0;
            array_indexes_type st = 0;
            array_indexes_type en = end-start;
            while(st < en){
                split = (st+en)/2;
                float_double distance_left = max_float_double;
                float_double distance_right = max_float_double;
                for(array_indexes_type i=0;i<split;i++){
                    distance_left = std::min(distance_left, (*kernels)[i].distance(midle));
                }
                for(array_indexes_type i=split;i<end-start;i++){
                    distance_right = std::min(distance_right, (*kernels)[i].distance(midle));
                }
                if (distance_left < distance_right){
                    en = split;
                }
                else{
                    if (distance_left > distance_right){
                        st = split+1;
                    }
                    else{
                        break;
                    }
                }
            }
            
            split = std::max(split, (array_indexes_type)1);
            split = std::min(split, end-start);
            left = kdtree_node3(kernels, start, split+start-1, (split_dim+1)%3, depth+1);
            right = kdtree_node3(kernels, split+start, end, (split_dim+1)%3, depth+1);
        }
        //array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char)+2*(3+3+1+1)*sizeof(float_double)/sizeof(char) + left.size + right.size;
        array_indexes_type size = sizeof(node) + left.size + right.size;
        char* data = new char[size];

        const unsigned char size_local = sizeof(node);
        ((node *) data)->left_range.range0 = left.range0;
        ((node *) data)->left_range.range1 = left.range1;
        ((node *) data)->left_range.range2 = left.range2;
        ((node *) data)->left_range.range3 = left.range3;
        ((node *) data)->left_range.range4 = left.range4;
        ((node *) data)->left_range.range5 = left.range5;
        ((node *) data)->left_range.range6 = left.range6;
        ((node *) data)->left_range.range7 = left.range7;
        ((node *) data)->right = size_local+left.size;
        ((node *) data)->right_range.range0 = right.range0;
        ((node *) data)->right_range.range1 = right.range1;
        ((node *) data)->right_range.range2 = right.range2;
        ((node *) data)->right_range.range3 = right.range3;
        ((node *) data)->right_range.range4 = right.range4;
        ((node *) data)->right_range.range5 = right.range5;
        ((node *) data)->right_range.range6 = right.range6;
        ((node *) data)->right_range.range7 = right.range7;
        for(array_indexes_type i=0;i<left.size;i++){
            data[size_local+i] = left.data[i];
        }
        const array_indexes_type ofset4 = size_local + left.size;
        for(array_indexes_type i=0;i<right.size;i++){
            data[ofset4+i] = right.data[i];
        }
        this->size = size;
        this->data = data;
        range0 = std::min(left.range0, right.range0);
        range1 = std::min(left.range1, right.range1);
        range2 = std::min(left.range2, right.range2);
        range3 = std::max(left.range3, right.range3);
        range4 = std::max(left.range4, right.range4);
        range5 = std::max(left.range5, right.range5);
        range6 = std::max(left.range6, right.range6);
        range7 = std::max(left.range7, right.range7);
        if(left.data != nullptr){
            free(left.data);
            left.data = nullptr;
        }
        if(right.data != nullptr){
            free(right.data);
            right.data = nullptr;
        }
    }
    void build_leaf(array_indexes_type start, array_indexes_type end, std::vector<gaussian_kernel2_3D>* ks){
        if((1+end)-start > max_leaf_size){
            std::cout << "error: leaf size is too big" << std::endl;
            std::cout << (1+end)-start << " " << max_leaf_size << std::endl;
            exit(8);
        }
        #if max_leaf_size > 1
            array_indexes_type size = 3*sizeof(array_indexes_type)/sizeof(char);
            char* data = new char[size];
            /*((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = start;
            ((array_indexes_type*)data)[2] = end;*/
            ((leaf*)data)[0].is_leaf = 0;
            ((leaf*)data)[0].start = start;
            ((leaf*)data)[0].end = end;
        #else
            array_indexes_type size = 2*sizeof(array_indexes_type)/sizeof(char);
            char* data = new char[size];
            /*((array_indexes_type*)data)[0] = 0;
            ((array_indexes_type*)data)[1] = start;*/
            ((leaf*)data)[0].is_leaf = 0;
            ((leaf*)data)[0].start = start;
        #endif          
            
            
        gaussian_kernel2_1D kernel_max = (*ks)[start].convert();
        range0 = kernel_max.mu.x;
        range1 = kernel_max.mu.y;
        range2 = kernel_max.mu.z;
        range3 = kernel_max.mu.x;
        range4 = kernel_max.mu.y;
        range5 = kernel_max.mu.z;
        range6 = kernel_max.log_weight;
        range7 = kernel_max.scale;
        for (array_indexes_type i=start+1;i<=end;i++){
            gaussian_kernel2_1D kernel = (*ks)[i].convert();
            range0 = std::min(range0, kernel.mu.x);
            range1 = std::min(range1, kernel.mu.y);
            range2 = std::min(range2, kernel.mu.z);
            range3 = std::max(range3, kernel.mu.x);
            range4 = std::max(range4, kernel.mu.y);
            range5 = std::max(range5, kernel.mu.z);
            range6 = std::max(range6, kernel.log_weight);
            range7 = std::max(range7, kernel.scale);
        }
        this->size = size;
        this->data = data;
    }
    friend std::ostream& operator<<(std::ostream& os, const kdtree_node3 &k){
        if(((array_indexes_type*)k.data)[0] == 0){
            array_indexes_type start = ((array_indexes_type*)k.data)[1];
            os << "kdtree_node(" << start << "," << start << ",";
            os << "[" << k.range0 << "," << k.range3 << "],";
            os << "[" << k.range1 << "," << k.range4 << "],";
            os << "[" << k.range2 << "," << k.range5 << "],";
            os << k.range6 << "," << k.range7 << ")";
            return os;
        }
        os << "kdtree_node(";
        os << "[" << k.range0 << "," << k.range3 << "],";
        os << "[" << k.range1 << "," << k.range4 << "],";
        os << "[" << k.range2 << "," << k.range5 << "],";
        os << k.range6 << "," << k.range7 << ",";
        //kdtree_node3* left = const_cast<kdtree_node3*>(&k)->left();
        //kdtree_node3* right = const_cast<kdtree_node3*>(&k)->right();
        //os << *left << "," << *right << ")";
        return os;
    }
};