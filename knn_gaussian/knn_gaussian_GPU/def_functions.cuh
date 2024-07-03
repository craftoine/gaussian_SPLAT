__device__
inline float_double distance_arr(float_double range0, float_double range3, float_double range1, float_double range4, float_double range2, float_double range5, float_double range6, float_double range7, point3d x){
    float_double dist = 0;
    float_double temp = min(max(x.x, range0), range3)-x.x;
    dist+= temp*temp;
    float_double temp2 = min(max(x.y, range1), range4)-x.y;
    dist+= temp2*temp2;
    float_double temp3 = min(max(x.z, range2), range5)-x.z;
    dist+= temp3*temp3;
    float_double d1_ = (range6 - dist/(2*range7*range7));
    float_double d1  = -(max(d1_, negligeable_val_when_exp));
    return d1;
}
__device__
//inline float_double distance_arr_2(/*char* data,*/ const char* rangedata, point3d x, gaussian_kernel2_3D* kernels){
inline float_double distance_arr_2(const range_data* rangedata, point3d x, gaussian_kernel2_3D* kernels){
    float_double dist = 0;
    float_double temp = min(max(x.x, rangedata->range0), rangedata->range3)-x.x;
    dist+= temp*temp;
    float_double temp2 = min(max(x.y, rangedata->range1), rangedata->range4)-x.y;
    dist+= temp2*temp2;
    float_double temp3 = min(max(x.z, rangedata->range2), rangedata->range5)-x.z;
    dist+= temp3*temp3;
    float_double d1_ = (rangedata->range6 - dist/(2*rangedata->range7*rangedata->range7));
    float_double d1  = -(max(d1_, negligeable_val_when_exp));
    return d1;
}
__device__
inline float_double distance_arr_3_left(const node* n, point3d x, gaussian_kernel2_3D* kernels){
    float_double dist = 0;
    float_double temp = min(max(x.x, (n->left_range).range0), n->left_range.range3)-x.x;
    dist+= temp*temp;
    float_double temp2 = min(max(x.y, n->left_range.range1), n->left_range.range4)-x.y;
    dist+= temp2*temp2;
    float_double temp3 = min(max(x.z, n->left_range.range2), n->left_range.range5)-x.z;
    dist+= temp3*temp3;
    float_double d1_ = (n->left_range.range6 - dist/(2*n->left_range.range7*n->left_range.range7));
    float_double d1  = -(max(d1_, negligeable_val_when_exp));
    return d1;
}
__device__
inline float_double distance_arr_3_right(const node* n, point3d x, gaussian_kernel2_3D* kernels){
    float_double dist = 0;
    float_double temp = min(max(x.x, (n->right_range).range0), n->right_range.range3)-x.x;
    dist+= temp*temp;
    float_double temp2 = min(max(x.y, n->right_range.range1), n->right_range.range4)-x.y;
    dist+= temp2*temp2;
    float_double temp3 = min(max(x.z, n->right_range.range2), n->right_range.range5)-x.z;
    dist+= temp3*temp3;
    float_double d1_ = (n->right_range.range6 - dist/(2*n->right_range.range7*n->right_range.range7));
    float_double d1  = -(max(d1_, negligeable_val_when_exp));
    return d1;
}

//define my own stack
template <typename T, array_indexes_type max_depth_>
class stack{
    public:
    T data[(max_depth_+1)+10];
    unsigned char size;
    __device__
    stack(){
        size = 0;
    }
    __device__
    inline void push(T x){
        data[size] = x;
        size++;
    }
    __device__
    inline void pop(){
        size--;
    }
    __device__
    inline T top(){
        return data[size-1];
    }
    __device__
    inline bool empty(){
        return size == 0;
    }
    __device__ 
    inline void clear(){
        size = 0;
    }
};


__device__
inline void search1nn_arr_non_rec_aux(stack<leaf*, max_depth>& stack_pointer, point3d x, gaussian_kernel2_3D* res,float_double* min_dist_, gaussian_kernel2_3D* kernels){
    array_indexes_type best = 0;
    float_double min_dist = *min_dist_;
    while(!stack_pointer.empty()){
        const leaf* current_data = stack_pointer.top();
        stack_pointer.pop();
        if (current_data->is_leaf == 0){
            array_indexes_type start = current_data->start;
            #if max_leaf_size >1
                array_indexes_type end = current_data->end;
                for (array_indexes_type i = 0 ; i <(end-start+1); i++){
                    float_double dist = (kernels)[start+i].distance(x);
                    if(dist < min_dist){
                        best = start+(i%((1+end)-start));
                        min_dist = dist;
                    }
                }
            #else
                float_double dist = (kernels)[start].distance(x);
                if(dist < min_dist){
                    best = start;
                    min_dist = dist;
                }
            #endif
        }else{
            const node* current_data_ = (node*)current_data;
            //float_double dist_left = distance_arr_2((const range_data*)current_data+sizeof(array_indexes_type), x, kernels);
            //float_double dist_right = distance_arr_2((const range_data*)current_data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), x, kernels);
            float_double dist_left = distance_arr_3_left(current_data_, x, kernels);
            float_double dist_right = distance_arr_3_right(current_data_, x, kernels);
            //const char* second_data = current_data + ((array_indexes_type*)(current_data+(dist_left < dist_right)*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
            leaf* second_data = (leaf*)(((char *)current_data_) + ((dist_left < dist_right)*(current_data_->right) + (dist_left >= dist_right)*(sizeof(node))));
            stack_pointer.data[stack_pointer.size] = second_data;
            stack_pointer.size += (max(dist_left, dist_right) < min_dist && max(dist_left, dist_right) <-negligeable_val_when_exp);
            //const char* first_data = current_data + ((array_indexes_type*)(current_data+(dist_left >= dist_right)*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
            leaf* first_data = (leaf*)(((char *)current_data_) + ((dist_left >= dist_right)*(current_data_->right) + (dist_left < dist_right)*(sizeof(node))));
            stack_pointer.data[stack_pointer.size] = first_data;
            stack_pointer.size += (min(dist_left, dist_right) < min_dist && min(dist_left, dist_right) <-negligeable_val_when_exp);



        }
    }
    //if(best != max_array_indexes_type){
        *res = (kernels)[best];
    //}
    *min_dist_ = min_dist;
    //*min_dist_ = (float_double)k;
}

struct /*__align__(16)*/ point_index{
    point3d point;
    array_indexes_type offset_point;
};

struct /*__align__(16)*/ block_1nn{
    char* node;
    stack<point_index, Size_point_block> stack_queries;
};
    



__device__
inline void search1nn_many_arr_non_rec_aux(stack<block_1nn, 2*max_depth>& stack_pointer, gaussian_kernel2_3D* res,float_double* min_dist_, gaussian_kernel2_3D* kernels){
    array_indexes_type bests[Size_point_block];
    #pragma unroll
    for (array_indexes_type i = 0; i < Size_point_block; i++){
        bests[i] = 0;
    }
    float_double min_dists[Size_point_block];
    #pragma unroll
    for (array_indexes_type i = 0; i < Size_point_block; i++){
        min_dists[i] = min_dist_[i];
    }
    while(!stack_pointer.empty()){
        block_1nn current_block = stack_pointer.top();
        stack_pointer.pop();
        const char* current_data = current_block.node;
        block_1nn left0;
        left0.node =(char*) current_data + ((array_indexes_type*)(current_data))[0];
        left0.stack_queries.clear();
        block_1nn right0;
        right0.node = (char*)current_data + ((array_indexes_type*)(current_data+sizeof(array_indexes_type) + (3+3+1+1)*sizeof(float_double)))[0];
        right0.stack_queries.clear(); 
        block_1nn left1;
        left1.node = (char*)current_data + ((array_indexes_type*)(current_data))[0];
        left1.stack_queries.clear();
        if(((array_indexes_type*)current_data)[0] !=0){  
            while(!current_block.stack_queries.empty()){
                const point_index p = current_block.stack_queries.top();
                current_block.stack_queries.pop();
                const float_double min_dist = min_dists[p.offset_point];
                float_double dist_left = distance_arr_2((const range_data*)current_data+sizeof(array_indexes_type), p.point, kernels);
                float_double dist_right = distance_arr_2((const range_data*)current_data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), p.point, kernels);
                if(dist_left < dist_right){
                    if(dist_left < min_dist && dist_left <-negligeable_val_when_exp){
                        left0.stack_queries.push(p);
                        if(dist_right < min_dist && dist_right <-negligeable_val_when_exp){
                            right0.stack_queries.push(p);
                        }
                    }
                }else{
                    if(dist_right < min_dist && dist_right <-negligeable_val_when_exp){
                        right0.stack_queries.push(p);
                        if(dist_left < min_dist && dist_left <-negligeable_val_when_exp){
                            left1.stack_queries.push(p);
                        }
                    }   
                }
            }
            if(!left1.stack_queries.empty()){
                stack_pointer.push(left1);
            }
            if(!right0.stack_queries.empty()){
                stack_pointer.push(right0);
            }
            if(!left0.stack_queries.empty()){
                stack_pointer.push(left0);
            }
        }else{
            array_indexes_type start = ((array_indexes_type*)current_data)[1];
            #if max_leaf_size == 1
                while(!current_block.stack_queries.empty()){
                    const point_index p = current_block.stack_queries.top();
                    current_block.stack_queries.pop();
                    float_double dist = (kernels)[start].distance(p.point);
                    if(dist < min_dists[p.offset_point] && dist <-negligeable_val_when_exp){
                        bests[p.offset_point] = start;
                        min_dists[p.offset_point] = dist;
                    }
                }
            #else
                array_indexes_type end = ((array_indexes_type*)current_data)[2];
                while(!current_block.stack_queries.empty()){
                    const point_index p = current_block.stack_queries.top();
                    current_block.stack_queries.pop();
                    for (array_indexes_type i = 0 ; i <(end-start+1); i++){
                        float_double dist = (kernels)[start+i].distance(p.point);
                        if(dist < min_dists[p.offset_point] && dist <-negligeable_val_when_exp){
                            bests[p.offset_point] = start+(i%((1+end)-start));
                            min_dists[p.offset_point] = dist;
                        }
                    }
                }
            #endif
        }
    }
    #pragma unroll
    for (array_indexes_type i = 0; i < Size_point_block; i++){
        res[i] = (kernels)[bests[i]];
        min_dist_[i] = min_dists[i];
    }
}
__device__
void search1nn_many_arr_non_rec_aux_2(stack<leaf*, max_depth>& stack_pointer,point3d * xs, gaussian_kernel2_3D* res,float_double* min_dist_, gaussian_kernel2_3D* kernels){
    array_indexes_type bests[Size_point_block];
    #pragma unroll
    for (array_indexes_type i = 0; i < Size_point_block; i++){
        bests[i] = 0;
    }
    //float_double min_dists[Size_point_block];
    //#pragma unroll
    /*for (array_indexes_type i = 0; i < Size_point_block; i++){
        min_dists[i] = min_dist_[i];
    }*/
    float_double* min_dists = min_dist_;
    //point3d my_x[Size_point_block];
    //#pragma unroll
    /*for (array_indexes_type i = 0; i < Size_point_block; i++){
        my_x[i] = xs[i];
    }*/
    point3d* my_x = xs;
    while(!stack_pointer.empty()){
        const leaf* current_data = stack_pointer.top();
        stack_pointer.pop();
        if (current_data->is_leaf == 0){
            array_indexes_type start = current_data->start;
            #if max_leaf_size >1
                array_indexes_type end = current_data->end;
                //#pragma unroll
                for(array_indexes_type j = 0 ; j <Size_point_block; j++){
                    for (array_indexes_type i = 0 ; i <(end-start+1); i++){
                        float_double dist = (kernels)[start+i].distance(my_x[j]);
                        if(dist < min_dists[j]){
                            bests[j] = start+(i%((1+end)-start));
                            min_dists[j] = dist;
                        }
                    }
                }
            #else
                #pragma unroll
                for (array_indexes_type j = 0 ; j <Size_point_block; j++){
                    float_double dist = (kernels)[start].distance(my_x[j]);
                    if(dist < min_dists[j]){
                        bests[j] = start;
                        min_dists[j] = dist;
                    }
                }
            #endif
        }else{
            const node* current_data_ = (node*)current_data;
            array_indexes_type nb_left_first = 0;
            array_indexes_type nb_right_first = 0;
            bool both = false;
            #pragma unroll
            for (array_indexes_type j = 0 ; j < Size_point_block; j++){
                //float_double dist_left = distance_arr_2((const range_data*)current_data+sizeof(array_indexes_type), my_x[j], kernels);
                //float_double dist_right = distance_arr_2((const range_data*)current_data+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), my_x[j], kernels);
                float_double dist_left = distance_arr_3_left(current_data_, my_x[j], kernels);
                float_double dist_right = distance_arr_3_right(current_data_, my_x[j], kernels);
                nb_left_first += (dist_left < dist_right && dist_left < -negligeable_val_when_exp && dist_left < min_dists[j]);
                nb_right_first += (dist_right <= dist_left && dist_right < -negligeable_val_when_exp && dist_right < min_dists[j]);
                both = both || (max(dist_left, dist_right) < min_dists[j] && max(dist_left, dist_right) <-negligeable_val_when_exp);
            }
            if(nb_left_first > 0 || nb_right_first > 0){
                //const char* second_data = current_data + ((array_indexes_type*)(current_data+(nb_left_first > nb_right_first)*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
                leaf* second_data = (leaf*)(((char *)current_data_) + ((nb_left_first > nb_right_first)*(current_data_->right) + (nb_left_first <= nb_right_first)*(sizeof(node))));
                stack_pointer.data[stack_pointer.size] = second_data;
                stack_pointer.size += (both || (nb_left_first > 0 && nb_right_first > 0));
                //const char* first_data = current_data + ((array_indexes_type*)(current_data+((nb_left_first <= nb_right_first))*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
                leaf* first_data = (leaf*)(((char *)current_data_) + ((nb_left_first <= nb_right_first)*(current_data_->right) + (nb_left_first > nb_right_first)*(sizeof(node))));
                stack_pointer.data[stack_pointer.size] = first_data;
                stack_pointer.size += 1;
            }



        }
    }
    //if(best != max_array_indexes_type){
    #pragma unroll
    for (array_indexes_type i = 0; i < Size_point_block; i++){
        res[i] = (kernels)[bests[i]];
        //min_dist_[i] = min_dists[i];
    }
}


__global__
void search1nn_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n){
    array_indexes_type index = blockIdx.x * blockDim.x + threadIdx.x;
    array_indexes_type stride = blockDim.x * gridDim.x;
    stack<leaf*, max_depth> stack_pointer;
    for (array_indexes_type i = index; i < n; i+=stride){
        //*(min_dist_ +i) = (float_double)(i +999);
        stack_pointer.push((leaf*)data);
        search1nn_arr_non_rec_aux(stack_pointer, xs[i], ress+i, min_dist_+i, kernels);
        stack_pointer.clear();
    }
}

__global__
void search1nn_many_arr_non_rec(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n){
    array_indexes_type index = blockIdx.x * blockDim.x + threadIdx.x;
    array_indexes_type stride = blockDim.x * gridDim.x;
    stack<block_1nn, 2*max_depth> stack_pointer;
    for (array_indexes_type i = index; i < n; i+=stride*Size_point_block){
        block_1nn block;
        block.node = data;
        for (array_indexes_type j = 0; j < Size_point_block; j++){
            point_index p;
            p.point = xs[i+j];
            p.offset_point = j;
            block.stack_queries.push(p);
        }
        stack_pointer.push(block);
        search1nn_many_arr_non_rec_aux(stack_pointer, ress+i, min_dist_+i, kernels);
    }
}

__global__ 
void search1nn_many_arr_non_rec_2(char*data, point3d* xs, gaussian_kernel2_3D* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n){
    array_indexes_type index = blockIdx.x * blockDim.x + threadIdx.x;
    array_indexes_type stride = blockDim.x * gridDim.x;
    stack<leaf*, max_depth> stack_pointer;
    for (array_indexes_type i = index*Size_point_block; i < n; i+=stride*Size_point_block){
        stack_pointer.push((leaf*)data);
        search1nn_many_arr_non_rec_aux_2(stack_pointer, xs+i, ress+i, min_dist_+i, kernels);
    }
}

__global__
void search1nn_brute_force(point3d* x, gaussian_kernel2_3D* ress, float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n, array_indexes_type n_kernels){
    array_indexes_type index = blockIdx.x * blockDim.x + threadIdx.x;
    array_indexes_type stride = blockDim.x * gridDim.x;
    for (array_indexes_type i = index; i < n; i+=stride){
        for (array_indexes_type j = 0; j < n_kernels; j++){
            float_double dist = (kernels)[j].distance(x[i]);
            if (dist < min_dist_[i]){
                min_dist_[i] = dist;
                ress[i] = (kernels)[j];
            }
        }
    }

}


__device__
inline void search_knn_arr_non_rec_aux(stack<leaf*, max_depth>& stack_pointer, point3d x,array_indexes_type* list_index ,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type* number_found_, float_double( & dist_list)[K_knn]){
    array_indexes_type number_found = *number_found_;
    while(!stack_pointer.empty()){
        const leaf* current_data = stack_pointer.top();
        stack_pointer.pop();
        if (current_data->is_leaf == 0){
            array_indexes_type start = current_data->start;
            #if max_leaf_size >1
                array_indexes_type end = current_data->end;
                for (array_indexes_type i = 0 ; i <(end-start+1); i++){
                    float_double dist = kernels[start+i].distance(x);
                    if(dist < dist_list[K_knn-1]){
                        /*if (number_found == 0){
                            //found[0] = kernels[start];
                            dist_list[0] = dist;
                            list_index[0] = start+(i%((1+end)-start));
                        }
                        else{*/
                            array_indexes_type i = 0;
                            array_indexes_type j = number_found;
                            while(j > i){
                                array_indexes_type m = (i+j)/2;
                                //if (found[m].distance(x) < dist){
                                if (dist_list[m] < dist){
                                    i=m+1;
                                }
                                else{
                                    j=m;
                                }
                            }
                            for(array_indexes_type l=number_found-(number_found == K_knn);l>i;l--){
                                //found[l] = found[l-1];
                                dist_list[l] = dist_list[l-1];
                                list_index[l] = list_index[l-1];
                            }
                            //found[i] = kernels[start];
                            dist_list[i] = dist;
                            list_index[i] = start+(i%((1+end)-start));
                        //}
                        number_found = min(K_knn, number_found+1);
                    }
                }
            #else   
                float_double dist = kernels[start].distance(x);
                //printf("dist: %f\n", dist);
                if (dist < dist_list[K_knn-1]){
                    if (number_found == 0){
                        //found[0] = kernels[start];
                        dist_list[0] = dist;
                        list_index[0] = start;
                    }
                    else{
                        array_indexes_type i = 0;
                        array_indexes_type j = number_found;
                        while(j > i){
                            array_indexes_type m = (i+j)/2;
                            //if (found[m].distance(x) < dist){
                            if (dist_list[m] < dist){
                                i=m+1;
                            }
                            else{
                                j=m;
                            }
                        }
                        if(number_found == K_knn){
                            //free(found[k-1]);
                        }
                        for(array_indexes_type l=number_found-(number_found == K_knn);l>i;l--){
                            //found[l] = found[l-1];
                            dist_list[l] = dist_list[l-1];
                            list_index[l] = list_index[l-1];
                        }
                        //found[i] = kernels[start];
                        dist_list[i] = dist;
                        list_index[i] = start;
                    }
                    number_found = min(K_knn, number_found+1);
                }
            #endif
        }else{
            const node* current_data_ = (node*)current_data;
            /*float_double dist_left = distance_arr_2((const range_data*)current_data_+sizeof(array_indexes_type), x, kernels);
            float_double dist_right = distance_arr_2((const range_data*)current_data_+2*sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double), x, kernels);
            */
            float_double dist_left = distance_arr_3_left(current_data_, x, kernels);
            float_double dist_right = distance_arr_3_right(current_data_, x, kernels);
            //const char* second_data = current_data_ + ((array_indexes_type*)(current_data_+(dist_left < dist_right)*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
            leaf* second_data = (leaf*)(((char *)current_data_) + ((dist_left < dist_right)*(current_data_->right) + (dist_left >= dist_right)*(sizeof(node))));
            stack_pointer.data[stack_pointer.size] = second_data;
            stack_pointer.size += (max(dist_left, dist_right) < dist_list[K_knn-1] && max(dist_left, dist_right) <-negligeable_val_when_exp);
            //const char* first_data = current_data_ + ((array_indexes_type*)(current_data_+(dist_left >= dist_right)*(sizeof(array_indexes_type)+(3+3+1+1)*sizeof(float_double))))[0];
            leaf* first_data = (leaf*)(((char *)current_data_) + ((dist_left >= dist_right)*(current_data_->right) + (dist_left < dist_right)*(sizeof(node))));
            stack_pointer.data[stack_pointer.size] = first_data;
            stack_pointer.size += (min(dist_left, dist_right) < dist_list[K_knn-1] && min(dist_left, dist_right) <-negligeable_val_when_exp);
        }
    }
    *number_found_ = number_found;
    *min_dist_ = dist_list[K_knn-1];
}


__global__
void search_knn_arr_non_rec(char*data, point3d* xs, array_indexes_type* ress,float_double* min_dist_, gaussian_kernel2_3D* kernels, array_indexes_type n, array_indexes_type* number_found_){
    array_indexes_type index = blockIdx.x * blockDim.x + threadIdx.x;
    array_indexes_type stride = blockDim.x * gridDim.x;
    stack<leaf*, max_depth> stack_pointer;
    float_double dist_list[K_knn];
    for (array_indexes_type i = index; i < n; i+=stride){
        //min_dist_[i] = 999;
        //number_found_[i] = 999;
        stack_pointer.push((leaf*)data);
        #pragma unroll
        for (array_indexes_type j = 0; j < K_knn; j++){
            dist_list[j] = min_dist_[i];
        }
        search_knn_arr_non_rec_aux(stack_pointer, xs[i], ress+(i*K_knn), min_dist_+i, kernels, number_found_+i, dist_list);
    }
}























class kd_tree3{
    public:
    kdtree_node3 * root;
    std::vector<gaussian_kernel2_3D>* kernels;
    kd_tree3(std::vector<gaussian_kernel2_3D>* ks){
        root = new kdtree_node3(ks);
        kernels = ks;
    }
    friend std::ostream& operator<<(std::ostream& os, const kd_tree3& k){
        os << "kd_tree(" << *k.root << ")";
        return os;
    }
};