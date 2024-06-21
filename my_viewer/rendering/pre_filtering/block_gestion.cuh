__global__ void select_gaussian_kernel_2(visual_gaussian_kernel* kernels, int n_kernels, int width, int height, point3d src, rotation_matrix dir, array_indexes_type ** selected_kernels, array_indexes_type * stacks_sizes, ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type ray_info_buffered_pixels_width){
    int global_thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    int nb_total_threads = gridDim.x*blockDim.x;
    int nb_block_screen_x = (width/rendering_block_size)+(width%rendering_block_size!=0);
    int nb_block_screen_y = (height/rendering_block_size)+(height%rendering_block_size!=0);
    array_indexes_type size_block_kernel = n_kernels/nb_total_threads;
    array_indexes_type start_kernel = global_thread_id*size_block_kernel+ min(global_thread_id, n_kernels%nb_total_threads);
    array_indexes_type end_kernel = min(n_kernels-1,(global_thread_id+1)*size_block_kernel+ min(global_thread_id+1, n_kernels%nb_total_threads)-1);
    array_indexes_type** selected_kernels_local;
    array_indexes_type* stacks_sizes_local;
    selected_kernels_local = selected_kernels+(global_thread_id* nb_block_screen_x*nb_block_screen_y);
    stacks_sizes_local = stacks_sizes+global_thread_id*nb_block_screen_x*nb_block_screen_y;
    for(int x = 0; x<nb_block_screen_x; x++){
        for(int y = 0; y<nb_block_screen_y; y++){
            stacks_sizes_local[x*nb_block_screen_y+y] = 0;
        }
    }
    //firstly we only count the number of selected kernels
    for(array_indexes_type i = global_thread_id; i<n_kernels; i+=nb_total_threads){
        #ifdef can_buffer_sigma
            kernels[i].reset_sigma_buffer();
            //kernels[i].get_reoriented_sigma(dir);
            rotation_matrix dir_;
            point3d src_;
            float_double pixel_x, pixel_y;
            get_point_nearest_screen_cord_(src, dir, pixel_x, pixel_y, kernels[i].kernel.mu, width, height);
            ray_info_(src, src_, dir, dir_, pixel_x, pixel_y, width, height);
            kernels[i].get_reoriented_sigma(dir_);
        #endif
        #ifdef can_buffer_color
            kernels[i].reset_rgb_buffer();
        #endif
        float_double min_x, max_x, min_y, max_y;
        get_bounding_rec_(src, dir, min_x, max_x, min_y, max_y, kernels[i], width, height);
        int block_x_min, block_x_max, block_y_min, block_y_max;
        get_full_block_from_scren_cord_(min_x, min_y, max_x, max_y, block_x_min, block_x_max, block_y_min, block_y_max, width, height);
        /*if(threadIdx.x == 0 && blockIdx.x == 2){
            //printf("i %d, min_x %f, max_x %f, min_y %f, max_y %f, block_x_min %d, block_x_max %d, block_y_min %d, block_y_max %d\n", i, min_x, max_x, min_y, max_y, block_x_min, block_x_max, block_y_min, block_y_max);
        }*/
        for(int x = block_x_min; x<=block_x_max; x++){
            for(int y = block_y_min; y<=block_y_max; y++){
                float_double worst_val;
                const float_double min_x_s = x * rendering_block_size;
                const float_double min_y_s = y * rendering_block_size;
                const float_double max_x_s = min(min_x_s + rendering_block_size, (float_double)width)-1;
                const float_double max_y_s = min(min_y_s + rendering_block_size, (float_double)height)-1;
                //get_worst_val_debug(&worst_val, src, dir, min_x_s, max_x_s, min_y_s, max_y_s, kernels[i], width, height, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, x+nb_block_screen_x*y,ray_info_buffered_pixels_width,i);
                get_worst_val_(worst_val, src, dir, min_x_s, max_x_s, min_y_s, max_y_s, kernels[i], width, height, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, x+nb_block_screen_x*y,ray_info_buffered_pixels_width);
                /*if(x == 29 && (y == 16|| y == 15)){
                    printf("worst_val %f, x %d, y %d\n", worst_val, x, y);
                    printf("min_x_s %f, max_x_s %f, min_y_s %f, max_y_s %f\n", min_x_s, max_x_s, min_y_s, max_y_s);
                    get_worst_val_(worst_val, src, dir, min_x_s, max_x_s, min_y_s, max_y_s, kernels[i], width, height, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, x+nb_block_screen_x*y,ray_info_buffered_pixels_width,true);
                }*/
                if (worst_val > negligeable_val_when_exp /*|| true*/){
                    stacks_sizes_local[x*nb_block_screen_y+y] += 1;
                }
            }
        }
    }
    //then alocate the memory for the stack
    //do one big malloc per thread instad of a lot of small malloc
    array_indexes_type total_size = 0;
    for(int x = 0; x<nb_block_screen_x; x++){
        for(int y = 0; y<nb_block_screen_y; y++){
            total_size += stacks_sizes_local[x*nb_block_screen_y+y];
        }
    }
    if(total_size != 0){
        //array_indexes_type t = 0;
        array_indexes_type* big_malloc;
        big_malloc = new array_indexes_type[total_size];
        if(big_malloc == NULL){
            printf("error allocating memory1 , size %d\n", total_size* sizeof(array_indexes_type));
        }
        else{
            //printf("allocated big memory size %d\n", total_size* sizeof(array_indexes_type));
        }
        array_indexes_type curent_index = 0;
        for(int x = 0; x<nb_block_screen_x; x++){
            for(int y = 0; y<nb_block_screen_y; y++){
                selected_kernels_local[x*nb_block_screen_y+y] = big_malloc+curent_index;
                curent_index += stacks_sizes_local[x*nb_block_screen_y+y];
            }
        }
        for(int x = 0; x<nb_block_screen_x; x++){
            for(int y = 0; y<nb_block_screen_y; y++){
                stacks_sizes_local[x*nb_block_screen_y+y] = 0;
            }
        }
        for(array_indexes_type i = global_thread_id; i<n_kernels; i+=nb_total_threads){
            float_double min_x, max_x, min_y, max_y;
            get_bounding_rec_(src, dir, min_x, max_x, min_y, max_y, kernels[i], width, height);
            int block_x_min, block_x_max, block_y_min, block_y_max;
            get_full_block_from_scren_cord_(min_x, min_y, max_x, max_y, block_x_min, block_x_max, block_y_min, block_y_max, width, height);
            if(threadIdx.x == 0 && blockIdx.x == 2){
                //printf("i %d, min_x %f, max_x %f, min_y %f, max_y %f, block_x_min %d, block_x_max %d, block_y_min %d, block_y_max %d\n", i, min_x, max_x, min_y, max_y, block_x_min, block_x_max, block_y_min, block_y_max);
            }
            for(int x = block_x_min; x<=block_x_max; x++){
                for(int y = block_y_min; y<=block_y_max; y++){
                    float_double worst_val;
                    const float_double min_x_s = x * rendering_block_size;
                    const float_double min_y_s = y * rendering_block_size;
                    const float_double max_x_s = min(min_x_s + rendering_block_size, (float_double)width)-1;
                    const float_double max_y_s = min(min_y_s + rendering_block_size, (float_double)height)-1;
                    //get_worst_val_debug(&worst_val, src, dir, min_x_s, max_x_s, min_y_s, max_y_s, kernels[i], width, height, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, x+nb_block_screen_x*y,ray_info_buffered_pixels_width,i);
                    get_worst_val_(worst_val, src, dir, min_x_s, max_x_s, min_y_s, max_y_s, kernels[i], width, height, ray_info_buffered_pixels, ray_info_buffered_pixels_midles, x+nb_block_screen_x*y,ray_info_buffered_pixels_width);
                    /*if(threadIdx.x == 0 && blockIdx.x == 2){
                        //rintf("worst_val %f, min_x_s %f, max_x_s %f, min_y_s %f, max_y_s %f\n", worst_val, min_x_s, max_x_s, min_y_s, max_y_s);
                    }*/
                    if (worst_val > negligeable_val_when_exp /*|| true*/){
                        /*t++;
                        if(t>total_size){
                            printf("error t > total_size\n");
                        }*/
                        selected_kernels_local[x*nb_block_screen_y+y][stacks_sizes_local[x*nb_block_screen_y+y]] = i;
                        stacks_sizes_local[x*nb_block_screen_y+y] += 1;
                    }
                }
            }
        }
        /*if(t != total_size){
            printf("error t != total_size\n");
        }*/
    }
    else{
        for(int x = 0; x<nb_block_screen_x; x++){
            for(int y = 0; y<nb_block_screen_y; y++){
                selected_kernels_local[x*nb_block_screen_y+y] = NULL;
            }
        }
    }
}
__global__ void free_select_gaussian_kernel_2(int n_kernels, int width, int height, array_indexes_type ** selected_kernels){
    int global_thread_id = blockIdx.x*blockDim.x+threadIdx.x;
    int nb_block_screen_x = (width/rendering_block_size)+(width%rendering_block_size!=0);
    int nb_block_screen_y = (height/rendering_block_size)+(height%rendering_block_size!=0);
    array_indexes_type** selected_kernels_local;
    selected_kernels_local = selected_kernels+(global_thread_id* nb_block_screen_x*nb_block_screen_y);
    if(selected_kernels_local[0] != NULL){
        //printf("freeing %p\n", selected_kernels_local[0]);
        free(selected_kernels_local[0]);
    }
}

__global__ void compact_selected_kernels_2(visual_gaussian_kernel* kernels, int n_kernels, int width, int height, point3d src, rotation_matrix dir, array_indexes_type ** selected_kernels, array_indexes_type * stacks_sizes, array_indexes_type ** result_selected_kernels, array_indexes_type * result_nb_selected_kernels,int nb_selected_threads){
    //lauch with the same parameters as screen rendering
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    const array_indexes_type block_size = block_size_x * block_size_y;
    //const array_indexes_type * stacks_sizes_local;
    const int nb_block_screen_x = (width/rendering_block_size)+(width%rendering_block_size!=0);
    const int nb_block_screen_y = (height/rendering_block_size)+(height%rendering_block_size!=0);
    const int my_screen_block_id = block_id_x*nb_block_screen_y+block_id_y;
    

    //the selection have benn done with a total nuber of thread bigger than the number of thread of this block
    array_indexes_type thread_block_size = nb_selected_threads/block_size;

    const int start_select_thread = thread_id*thread_block_size + min(thread_id, nb_selected_threads%block_size);
    const int end_select_thread = min(nb_selected_threads-1, (thread_id+1)*thread_block_size + min(thread_id+1, nb_selected_threads%block_size)-1);
    /*if(block_id_x == 0 && block_id_y == 0){
        printf("block of cord %d %d, thread %d, start_select_thread %d, end_select_thread %d\n", block_id_x, block_id_y, thread_id, start_select_thread, end_select_thread);
    }*/
    //have to compute the cumulative sum of the number of selected kernels
    array_indexes_type cumu_sum = 0;
    //comput it badly for now
    for(array_indexes_type i = 0; (thread_id !=0 && i<start_select_thread) || (thread_id == 0 && i<nb_selected_threads); i++){
        cumu_sum += stacks_sizes[i*nb_block_screen_x*nb_block_screen_y+my_screen_block_id];
    }
    if (thread_id == 0){
        result_nb_selected_kernels[my_screen_block_id] = cumu_sum;
        if (cumu_sum == 0){
            result_selected_kernels[my_screen_block_id] = NULL;
        }
        else{
            result_selected_kernels[my_screen_block_id] = new array_indexes_type[cumu_sum];
            if(result_selected_kernels[my_screen_block_id] == NULL){
                printf("error allocating memory2 %d\n", cumu_sum* sizeof(array_indexes_type));
            }
            else{
                //printf("allocated memory2 %d\n", cumu_sum * sizeof(array_indexes_type));
            }
        }
        cumu_sum = 0;
    }
    __syncthreads();
    array_indexes_type intern_cumu_sum = 0;
    for(array_indexes_type i = start_select_thread; i<=end_select_thread; i++){
        intern_cumu_sum += stacks_sizes[i*nb_block_screen_x*nb_block_screen_y+my_screen_block_id];
    }
    array_indexes_type s = 0;
    array_indexes_type curent_select_thread_id = start_select_thread;
    for(array_indexes_type i = cumu_sum; i<cumu_sum+intern_cumu_sum; i++){
        while (s == stacks_sizes[curent_select_thread_id*nb_block_screen_x*nb_block_screen_y+my_screen_block_id]){
            /*if(stacks_sizes[curent_select_thread_id*nb_block_screen_x*nb_block_screen_y+my_screen_block_id] != 0){
                selected_kernels[curent_select_thread_id*nb_block_screen_x*nb_block_screen_y+my_screen_block_id] = NULL;
            }*/
            curent_select_thread_id++;
            s = 0;
        }
        result_selected_kernels[my_screen_block_id][i] = selected_kernels[curent_select_thread_id*nb_block_screen_x*nb_block_screen_y+my_screen_block_id][s];
        s++;
    }
}
__global__ void clean_and_compute_ray_info_buffered(int width, int height, point3d src, rotation_matrix dir, ray_info_buffered* ray_info_buffered_pixels, ray_info_buffered* ray_info_buffered_pixels_midles,array_indexes_type ray_info_buffered_pixels_width){
    //lauch with the same parameters as screen rendering
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    //const array_indexes_type block_size = block_size_x * block_size_y;
    const array_indexes_type min_x = block_id_x * block_size_x;
    const array_indexes_type min_y = block_id_y * block_size_y;
    const array_indexes_type max_x = min(min_x + block_size_x, width);
    const array_indexes_type max_y = min(min_y + block_size_y, height);
    const float_double middle_x = (float_double)(min_x + max_x) / 2.;
    const float_double middle_y = (float_double)(min_y + max_y) / 2.;
    const array_indexes_type my_x = min_x + thread_id % block_size_x;
    const array_indexes_type my_y = min_y + thread_id / block_size_x;
    //const array_indexes_type nb_block_x = gridDim.x;
    //compute my pixel
    ray_info_buffered_pixels[my_x+my_y*ray_info_buffered_pixels_width] = ray_info_buffered( src, dir, my_x, my_y, width, height);
    //if i'm thread 0 compute the middle
    if(thread_id == 0){
        ray_info_buffered_pixels_midles[block_id_x+block_id_y*gridDim.x] = ray_info_buffered( src, dir, middle_x, middle_y, width, height);
    }
}