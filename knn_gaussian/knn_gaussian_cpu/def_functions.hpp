float_double distance_arr_2(/*char* data,*/ char* rangedata, point3d x, std::vector<gaussian_kernel2_3D>* kernels){
    /*if (((array_indexes_type*)data)[0] == 0){
        array_indexes_type start = ((array_indexes_type*)data)[1];
        return (*kernels)[start].distance(x);
    }*/
    point3d projection;
    projection.x = x.x;
    projection.y = x.y;
    projection.z = x.z;
    float_double range0 = ((float_double*)rangedata)[0];
    if (x.x < range0){
        projection.x = range0;
    }
    else{float_double range3 = ((float_double*)rangedata)[3];
        if (x.x > range3){
            projection.x = range3;
        }
    }
    float_double range1 = ((float_double*)rangedata)[1];
    if (x.y < range1){
        projection.y = range1;
    }
    
    else{float_double range4 = ((float_double*)rangedata)[4];
        if (x.y > range4){
            projection.y = range4;
        }
    }
    float_double range2 = ((float_double*)rangedata)[2];
    if (x.z < range2){
        projection.z = range2;
    }
    else{float_double range5 = ((float_double*)rangedata)[5];
        if (x.z > range5){
            projection.z = range5;
        }
    }
    float_double dist = x.square_distance(projection);
    float_double range6 = ((float_double*)rangedata)[6];
    float_double range7 = ((float_double*)rangedata)[7];
    float_double d1_ = (range6 - dist/(2*range7*range7));
    float_double d1  = -(std::max(d1_, negligeable_val_when_exp));
    return d1;
}
void search1nn_arr_non_rec(char*data, point3d x, gaussian_kernel2_3D* res,float_double* min_dist_, std::vector<gaussian_kernel2_3D>* kernels){
    //std::cout << "search1nn_arr_non_rec" << std::endl;
    float_double min_dist = *min_dist_;
    array_indexes_type best = max_array_indexes_type;
    std::stack<char*> stack_pointer;
    std::stack<float_double> stack_dist;
    stack_pointer.push(data);
    stack_dist.push(std::numeric_limits<float_double>::min());
    while(!stack_pointer.empty()){
        char* current_data = stack_pointer.top();
        stack_pointer.pop();
        float_double current_dist = stack_dist.top();
        stack_dist.pop();
        if(current_dist < min_dist){
            if (((array_indexes_type*)current_data)[0] == 0){
                array_indexes_type start = ((array_indexes_type*)current_data)[1];
                /*if ((*kernels)[start].distance(x) < min_dist){
                    best = start;
                    min_dist = (*kernels)[start].distance(x);
                }*/
                #if max_leaf_size > 1
                    array_indexes_type end = ((array_indexes_type*)current_data)[2];
                    //std::cout << "start: " << start << " end: " << end << std::endl;
                    for(array_indexes_type i = start; i <= end; i++){
                        if ((*kernels)[i].distance(x) < min_dist){
                            best = i;
                            min_dist = (*kernels)[i].distance(x);
                        }
                    }
                #else
                    if ((*kernels)[start].distance(x) < min_dist){
                        best = start;
                        min_dist = (*kernels)[start].distance(x);
                    }
                #endif
            }else{
                char* left_data = current_data + ((array_indexes_type*)current_data)[0];
                char* right_data = current_data + ((array_indexes_type*)(current_data+sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
                float_double dist_left = distance_arr_2(/*left_data,*/ current_data+sizeof(array_indexes_type), x, kernels);
                float_double dist_right = distance_arr_2(/*right_data,*/ current_data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), x, kernels);
                char* first;
                char* second;
                float_double dist_first;
                float_double dist_second;
                if (dist_left < dist_right){
                    first = left_data;
                    second = right_data;
                    dist_first = dist_left;
                    dist_second = dist_right;
                }else{
                    first = right_data;
                    second = left_data;
                    dist_first = dist_right;
                    dist_second = dist_left;
                }
                if (dist_first < min_dist && dist_first <-negligeable_val_when_exp){
                    if (dist_second < min_dist && dist_second <-negligeable_val_when_exp){
                        stack_pointer.push(second);
                        stack_dist.push(dist_second);
                    }
                    stack_pointer.push(first);
                    stack_dist.push(dist_first);
                    //the first to explore on top
                }
            }
        }
    }
    if(best != max_array_indexes_type){
        *res = (*kernels)[best];
    }
    *min_dist_ = min_dist;
    //std::cout << "search1nn_arr_non_rec end" << std::endl;
}
void search_knn_arr_non_rec(char*data, point3d x, float_double* min_dist_,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found_, std::vector<gaussian_kernel2_3D>* kernels){
    //std::cout << "search_knn_arr_non_rec" << std::endl;
    array_indexes_type number_found = *number_found_;
    float_double min_dist = *min_dist_;
    std::stack<char*> stack_pointer;
    std::stack<float_double> stack_dist;
    stack_pointer.push(data);
    stack_dist.push(std::numeric_limits<float_double>::min());
    while(!stack_pointer.empty()){
        char* current_data = stack_pointer.top();
        stack_pointer.pop();
        float_double current_dist = stack_dist.top();
        stack_dist.pop();
        if(current_dist < min_dist){
            if (((array_indexes_type*)current_data)[0] == 0){
                array_indexes_type start = ((array_indexes_type*)current_data)[1];
                #if max_leaf_size == 1
                    if ((*kernels)[start].distance(x) < min_dist){
                        if (number_found == 0){
                            found[0] = new gaussian_kernel2_3D((*kernels)[start]);
                        }
                        else{
                            float_double dist = (*kernels)[start].distance(x);
                            array_indexes_type i = 0;
                            array_indexes_type j = number_found;
                            while(j > i){
                                array_indexes_type m = (i+j)/2;
                                if (found[m]->distance(x) < dist){
                                    i=m+1;
                                }
                                else{
                                    j=m;
                                }
                            }
                            if(number_found == k){
                                free(found[k-1]);
                            }
                            for(array_indexes_type l=number_found-(number_found == k);l>i;l--){
                                found[l] = found[l-1];
                            }
                            found[i] = new gaussian_kernel2_3D((*kernels)[start]);
                        }
                        number_found = std::min(k, number_found+1);
                        if(number_found == k){
                            min_dist = found[k-1]->distance(x);
                        }
                    }
                #else
                    array_indexes_type end = ((array_indexes_type*)current_data)[2];
                    for(array_indexes_type i = start; i <= end; i++){
                        //std::cout << "i: " << i << std::endl;
                        if ((*kernels)[i].distance(x) < min_dist){
                            //std::cout << "a" << std::endl;
                            if (number_found == 0){
                                //std::cout << "b" << std::endl;
                                found[0] = new gaussian_kernel2_3D((*kernels)[i]);
                            }
                            else{
                                //std::cout << "c" << std::endl;
                                float_double dist = (*kernels)[i].distance(x);
                                array_indexes_type i = 0;
                                array_indexes_type j = number_found;
                                while(j > i){
                                    array_indexes_type m = (i+j)/2;
                                    if (found[m]->distance(x) < dist){
                                        i=m+1;
                                    }
                                    else{
                                        j=m;
                                    }
                                }
                                //std::cout << "d" << std::endl;
                                if(number_found == k){
                                    //std::cout << found[k-1] << std::endl;
                                    //std::cout << *(found[k-1]) << std::endl;
                                    free(found[k-1]);
                                }
                                //std::cout << "e1" << std::endl;
                                for(array_indexes_type l=number_found;l>i;l--){
                                    found[l] = found[l-1];
                                }
                                //std::cout << "e2" << std::endl;
                                found[i] = new gaussian_kernel2_3D((*kernels)[i]);
                            }
                            //std::cout << "e" << std::endl;
                            number_found = std::min(k, number_found+1);
                            if(number_found == k){
                                min_dist = found[k-1]->distance(x);
                            }
                            //std::cout << "f" << std::endl;
                        }
                    }
                #endif
                //std::cout << "end: " << end << std::endl;
            }else{
                char* left_data = current_data + ((array_indexes_type*)current_data)[0];
                char* right_data = current_data + ((array_indexes_type*)(current_data+sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double)))[0];
                float_double dist_left = distance_arr_2(/*left_data,*/ current_data+sizeof(array_indexes_type), x, kernels);
                float_double dist_right = distance_arr_2(/*right_data,*/ current_data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), x, kernels);
                char* first;
                char* second;
                float_double dist_first;
                float_double dist_second;
                if (dist_left < dist_right){
                    first = left_data;
                    second = right_data;
                    dist_first = dist_left;
                    dist_second = dist_right;
                }else{
                    first = right_data;
                    second = left_data;
                    dist_first = dist_right;
                    dist_second = dist_left;
                }
                if (dist_first < min_dist && dist_first <-negligeable_val_when_exp){
                    if (dist_second < min_dist && dist_second <-negligeable_val_when_exp){
                        stack_pointer.push(second);
                        stack_dist.push(dist_second);
                    }
                    stack_pointer.push(first);
                    stack_dist.push(dist_first);
                    //the first to explore on top
                }
            }
        }
    }
    *number_found_ = number_found;
    *min_dist_ = min_dist;
    //std::cout << "search_knn_arr_non_rec end" << std::endl;
}
class kd_tree3{
    public:
    kdtree_node3 * root;
    std::vector<gaussian_kernel2_3D>* kernels;
    kd_tree3(std::vector<gaussian_kernel2_3D>* ks){
        root = new kdtree_node3(ks);
        kernels = ks;
    }
    void search_1nn(gaussian_kernel2_3D* res,point3d x,float_double* min_dist, size_t* debug_counter = nullptr){
        search1nn_arr_non_rec(root->data, x, res, min_dist, kernels);
    }
    void search_knn(point3d x, float_double* min_dist,gaussian_kernel2_3D** found, array_indexes_type k, array_indexes_type* number_found){
        search_knn_arr_non_rec(root->data, x, min_dist, found, k, number_found, kernels);
    }
    friend std::ostream& operator<<(std::ostream& os, const kd_tree3& k){
        os << "kd_tree(" << *k.root << ")";
        return os;
    }
};

