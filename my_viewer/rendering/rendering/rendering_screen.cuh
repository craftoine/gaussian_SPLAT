__host__
void render_screen (point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height){
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            //point3d src = (point3d(((float_double)i) / (float_double)width, ((float_double)j) / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
            point3d src;
            //float_double pixel_x = ((((float_double)i) / (float_double)width)-0.5)*x_scaling;
            //float_double pixel_y = ((((float_double)j) / (float_double)height)-0.5)*y_scaling;
            rotation_matrix dir;
            //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
            ray_info_(src_, src, dir_, dir, i, j, width, height);
            //std::cout << "rendering pixel " << i << " " << j << std::endl;
            //screen[i * height + j] = render_pixel(dir, src, kernels, n_kernels);
            screen[i * height + j] = render_pixel_mix(dir, src, kernels, n_kernels);
            //screen[i * height + j] = render_pixel_my(dir, src, kernels, n_kernels);
            //std::cout << "i " << i << " j " << j << " r " << screen[i * height + j].x << " g " << screen[i * height + j].y << " b " << screen[i * height + j].z << std::endl;
        }
    }
}
__global__
void render_screen_block(point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height){
    //shared momory for the block to save the kernel order
    extern __shared__ index_value shared_sorted_indexes[];
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    const array_indexes_type block_size = block_size_x * block_size_y;
    const array_indexes_type min_x = block_id_x * block_size_x;
    const array_indexes_type min_y = block_id_y * block_size_y;
    const array_indexes_type max_x = min(min_x + block_size_x, width);
    const array_indexes_type max_y = min(min_y + block_size_y, height);
    const float_double middle_x = (float_double)(min_x + max_x) / 2.;
    const float_double middle_y = (float_double)(min_y + max_y) / 2.;

    //load the values using every thread of the block
    for (int i = thread_id; i < n_kernels; i+=block_size)
    {
        //point3d src = (point3d(middle_x / (float_double)width, middle_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)middle_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)middle_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, middle_x, middle_y, width, height);
        colored_gaussian_kernel_1D kernel = kernels[i].get_1D_kernel(dir, src);
        shared_sorted_indexes[i] = index_value(i, kernel.mu);
    }
    __syncthreads();
    array_indexes_type kernel_block_size = n_kernels/block_size;
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d min_x %d min_y %d max_x %d max_y %d middle_x %f middle_y %f kernel_block_size %d\n", block_id_x, block_id_y, min_x, min_y, max_x, max_y, middle_x, middle_y, kernel_block_size);
    }*/
    if(kernel_block_size>= 2){
        //sort the array using all the threads of the block
        // repartition of kernels is:
        // kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size kernel_block_size kernel_block_size kernel_block_size

        array_indexes_type my_start = thread_id * (kernel_block_size) + min(thread_id, (n_kernels)%(block_size));
        array_indexes_type my_end = min((thread_id+1) * (kernel_block_size) + min(thread_id+1, (n_kernels)%(block_size))-1, n_kernels-1);

        //localy sort the array
        bool local_sorted = false;
        while (!local_sorted)
        {
            local_sorted = true;
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                    shared_sorted_indexes[i+1] = temp;
                    local_sorted = false;
                }
            }
        }
        __syncthreads();
        //check if my last val smalest than the first value of the next thread
        bool no_conflict;
        if(my_end < n_kernels-1){
            if(shared_sorted_indexes[my_end] < shared_sorted_indexes[my_end+1]){
                no_conflict = true;
            }
            else{
                no_conflict = false;
            }
        }
        else{
            no_conflict = true;
        }
        __syncthreads();
        int nb_sorted = __syncthreads_count(no_conflict);

        while (nb_sorted < block_size)
        {
            //check if the my last value have to be swapped with the fist value of the next thread
            if(my_end < n_kernels-1){
                if(shared_sorted_indexes[my_end] > shared_sorted_indexes[my_end+1]){
                    index_value temp = shared_sorted_indexes[my_end];
                    shared_sorted_indexes[my_end] = shared_sorted_indexes[my_end+1];
                    shared_sorted_indexes[my_end+1] = temp;
                    local_sorted = false;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //localy sort the array
            /*local_sorted = false;
            while (!local_sorted)
            {
                local_sorted = true;
                for (int i = my_start; i < my_end; i++)
                {
                    if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                        index_value temp = shared_sorted_indexes[i];
                        shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                        shared_sorted_indexes[i+1] = temp;
                        local_sorted = false;
                    }
                }
            }*/
            //only need 2 steps, one from top to bottom and one from bottom to top
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                    shared_sorted_indexes[i+1] = temp;
                }
                else{
                    break;
                }
                    
            }
            for (int i = my_end; i > my_start; i--)
            {
                if(shared_sorted_indexes[i] < shared_sorted_indexes[i-1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i-1];
                    shared_sorted_indexes[i-1] = temp;
                }
                else{
                    break;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //check if my last val smalest than the first value of the next thread
            no_conflict = false;
            if(my_end < n_kernels-1){
                if(shared_sorted_indexes[my_end] < shared_sorted_indexes[my_end+1]){
                    no_conflict = true;
                }
                else{
                    no_conflict = false;
                }
            }
            else{
                no_conflict = true;
            }
            __syncthreads();
            nb_sorted = __syncthreads_count(no_conflict);
        }
    }
    else{
        //sort the array using only the first thread of the block
        if(thread_id == 0){
            bool sorted = false;
            while (!sorted)
            {
                sorted = true;
                for (int i = 0; i < n_kernels-1; i++)
                {
                    if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                        index_value temp = shared_sorted_indexes[i];
                        shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                        shared_sorted_indexes[i+1] = temp;
                        sorted = false;
                    }
                }
            }
        //printf("sorted value0 %f value1 %f value2 %f value3 %f value4 %f value5 %f\n", shared_sorted_indexes[0].value, shared_sorted_indexes[1].value, shared_sorted_indexes[2].value, shared_sorted_indexes[3].value, shared_sorted_indexes[4].value, shared_sorted_indexes[5].value);
        }
        __syncthreads();
    }
    //render the screen
    array_indexes_type my_x = min_x + threadIdx.x;
    array_indexes_type my_y = min_y + threadIdx.y;
    //if i am in the screen render the pixel
    if(my_x < width && my_y < height){
        //point3d src = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)my_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)my_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, my_x, my_y, width, height);
        screen[my_x * height + my_y] = render_pixel_block(dir, src, kernels, n_kernels, shared_sorted_indexes);
        //printf("i %d j %d r %f g %f b %f dir %f %f %f\n", my_x, my_y, screen[my_x * height + my_y].x, screen[my_x * height + my_y].y, screen[my_x * height + my_y].z, dir.x, dir.y, dir.z);
    }
}

__global__
void render_screen_block_mix(point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height){
    //shared momory for the block to save the kernel order
    extern __shared__ index_value shared_sorted_indexes[];
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    const array_indexes_type block_size = block_size_x * block_size_y;
    const array_indexes_type min_x = block_id_x * block_size_x;
    const array_indexes_type min_y = block_id_y * block_size_y;
    const array_indexes_type max_x = min(min_x + block_size_x, width);
    const array_indexes_type max_y = min(min_y + block_size_y, height);
    const float_double middle_x = (float_double)(min_x + max_x) / 2.;
    const float_double middle_y = (float_double)(min_y + max_y) / 2.;

    //load the values using every thread of the block
    for (int i = thread_id; i < n_kernels; i+=block_size)
    {
        //point3d src = (point3d(middle_x / (float_double)width, middle_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)middle_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)middle_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, middle_x, middle_y, width, height);
        colored_gaussian_kernel_1D kernel = kernels[i].get_1D_kernel(dir, src);
        shared_sorted_indexes[i] = index_value(i, kernel.mu);
    }
    __syncthreads();
    array_indexes_type kernel_block_size = n_kernels/block_size;
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d min_x %d min_y %d max_x %d max_y %d middle_x %f middle_y %f kernel_block_size %d\n", block_id_x, block_id_y, min_x, min_y, max_x, max_y, middle_x, middle_y, kernel_block_size);
    }*/
    if(kernel_block_size>= 2){
        //sort the array using all the threads of the block
        // repartition of kernels is:
        // kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size kernel_block_size kernel_block_size kernel_block_size

        array_indexes_type my_start = thread_id * (kernel_block_size) + min(thread_id, (n_kernels)%(block_size));
        array_indexes_type my_end = min((thread_id+1) * (kernel_block_size) + min(thread_id+1, (n_kernels)%(block_size))-1, n_kernels-1);

        //localy sort the array
        bool local_sorted = false;
        while (!local_sorted)
        {
            local_sorted = true;
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                    shared_sorted_indexes[i+1] = temp;
                    local_sorted = false;
                }
            }
        }
        __syncthreads();
        //check if my last val smalest than the first value of the next thread
        bool no_conflict;
        if(my_end < n_kernels-1){
            if(shared_sorted_indexes[my_end] < shared_sorted_indexes[my_end+1]){
                no_conflict = true;
            }
            else{
                no_conflict = false;
            }
        }
        else{
            no_conflict = true;
        }
        __syncthreads();
        int nb_sorted = __syncthreads_count(no_conflict);

        while (nb_sorted < block_size)
        {
            //check if the my last value have to be swapped with the fist value of the next thread
            if(my_end < n_kernels-1){
                if(shared_sorted_indexes[my_end] > shared_sorted_indexes[my_end+1]){
                    index_value temp = shared_sorted_indexes[my_end];
                    shared_sorted_indexes[my_end] = shared_sorted_indexes[my_end+1];
                    shared_sorted_indexes[my_end+1] = temp;
                    local_sorted = false;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //localy sort the array
            /*local_sorted = false;
            while (!local_sorted)
            {
                local_sorted = true;
                for (int i = my_start; i < my_end; i++)
                {
                    if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                        index_value temp = shared_sorted_indexes[i];
                        shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                        shared_sorted_indexes[i+1] = temp;
                        local_sorted = false;
                    }
                }
            }*/
            //only need 2 steps, one from top to bottom and one from bottom to top
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                    shared_sorted_indexes[i+1] = temp;
                }
                else{
                    break;
                }
            }
            for (int i = my_end; i > my_start; i--)
            {
                if(shared_sorted_indexes[i] < shared_sorted_indexes[i-1]){
                    index_value temp = shared_sorted_indexes[i];
                    shared_sorted_indexes[i] = shared_sorted_indexes[i-1];
                    shared_sorted_indexes[i-1] = temp;
                }
                else{
                    break;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //check if my last val smalest than the first value of the next thread
            no_conflict = false;
            if(my_end < n_kernels-1){
                if(shared_sorted_indexes[my_end] < shared_sorted_indexes[my_end+1]){
                    no_conflict = true;
                }
                else{
                    no_conflict = false;
                }
            }
            else{
                no_conflict = true;
            }
            __syncthreads();
            nb_sorted = __syncthreads_count(no_conflict);
        }
    }
    else{
        //sort the array using only the first thread of the block
        if(thread_id == 0){
            bool sorted = false;
            while (!sorted)
            {
                sorted = true;
                for (int i = 0; i < n_kernels-1; i++)
                {
                    if(shared_sorted_indexes[i] > shared_sorted_indexes[i+1]){
                        index_value temp = shared_sorted_indexes[i];
                        shared_sorted_indexes[i] = shared_sorted_indexes[i+1];
                        shared_sorted_indexes[i+1] = temp;
                        sorted = false;
                    }
                }
            }
        //printf("sorted value0 %f value1 %f value2 %f value3 %f value4 %f value5 %f\n", shared_sorted_indexes[0].value, shared_sorted_indexes[1].value, shared_sorted_indexes[2].value, shared_sorted_indexes[3].value, shared_sorted_indexes[4].value, shared_sorted_indexes[5].value);
        }
        //syncrhonize the threads
        __syncthreads();
    }
    //render the screen
    array_indexes_type my_x = min_x + threadIdx.x;
    array_indexes_type my_y = min_y + threadIdx.y;
    //if i am in the screen render the pixel
    if(my_x < width && my_y < height){
        //point3d src = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)my_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)my_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, my_x, my_y, width, height);
        screen[my_x * height + my_y] = render_pixel_mix_block(dir, src, kernels, n_kernels, shared_sorted_indexes);
        //printf("i %d j %d r %f g %f b %f dir %f %f %f\n", my_x, my_y, screen[my_x * height + my_y].x, screen[my_x * height + my_y].y, screen[my_x * height + my_y].z, dir.x, dir.y, dir.z);
    }
}

__global__
void render_screen_block_my(point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height){
    //shared momory for the block to save the kernel order
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    const array_indexes_type block_size = block_size_x * block_size_y;
    const array_indexes_type min_x = block_id_x * block_size_x;
    const array_indexes_type min_y = block_id_y * block_size_y;
    const array_indexes_type max_x = min(min_x + block_size_x, width);
    const array_indexes_type max_y = min(min_y + block_size_y, height);
    const float_double middle_x = (float_double)(min_x + max_x) / 2.;
    const float_double middle_y = (float_double)(min_y + max_y) / 2.;

    extern __shared__ char s[];
    if(thread_id == 0){
        //allocate the stack
        array_indexes_type *stack = new array_indexes_type[n_kernels+1];
        ((array_indexes_type **)s)[0] = stack;
    }
    __syncthreads();
    array_indexes_type * nb_curent_kernels = ((array_indexes_type **)s)[0];
    if(thread_id == 0){
        nb_curent_kernels[0] = 0;
    }
    array_indexes_type * share_stack = nb_curent_kernels+1;
    index_start_end* shared_sorted_indexes_ = (index_start_end *)(s+sizeof(array_indexes_type*));
    clock_t start_loading = clock();
    //load the values using every thread of the block
    for (int i = thread_id; i < n_kernels; i+=block_size)
    {
        //point3d src = (point3d(middle_x / (float_double)width, middle_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)middle_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)middle_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, middle_x, middle_y, width, height);
        colored_gaussian_kernel_1D kernel = kernels[i].get_1D_kernel(dir, src);
        /*shared_sorted_indexes_[i*2] = index_start_end(i, true, kernel.start());
        shared_sorted_indexes_[i*2+1] = index_start_end(i, false, kernel.end());*/
        float_double min_start = kernel.start();
        float_double max_end = kernel.end();
        /*for(int dx = 0; dx < block_size_x; dx++){
            for(int dy = 0; dy < block_size_y; dy++){
                array_indexes_type my_x = min_x + dx;
                array_indexes_type my_y = min_y + dy;
                if(my_x < width && my_y < height){
                    dir = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
                    kernel = kernels[i].get_1D_kernel(point3d(0, 0, 1), dir);
                    if(kernel.start() < kernel.end()){
                        min_start = min(min_start,kernel.start());
                        max_end = max(max_end, kernel.end());
                    }
                }
            }
        }*/
        shared_sorted_indexes_[i] = index_start_end(i, true, min_start);
        shared_sorted_indexes_[i+n_kernels] = index_start_end(i, false, max_end);
    }
    __syncthreads();
    clock_t end_loading = clock();
    array_indexes_type kernel_block_size = (n_kernels*2)/block_size;
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d min_x %d min_y %d max_x %d max_y %d middle_x %f middle_y %f kernel_block_size %d\n", block_id_x, block_id_y, min_x, min_y, max_x, max_y, middle_x, middle_y, kernel_block_size);
    }*/
    clock_t start_sorting = clock();
    if(kernel_block_size>=2){
        //sort the array using all the threads of the block
        // repartition of kernels is:
        // kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size kernel_block_size kernel_block_size kernel_block_size

        array_indexes_type my_start = thread_id * (kernel_block_size) + min(thread_id, (n_kernels*2)%(block_size));
        array_indexes_type my_end = min((thread_id+1) * (kernel_block_size) + min(thread_id+1, (n_kernels*2)%(block_size))-1, n_kernels*2-1);

        //localy sort the array
        bool local_sorted = false;
        while (!local_sorted)
        {
            local_sorted = true;
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                    local_sorted = false;
                }
            }
        }
        //syncrhonize the threads
        __syncthreads();
        //check if my last val smalest than the first value of the next thread
        bool no_conflict;
        if(my_end < n_kernels*2-1){
            if(shared_sorted_indexes_[my_end] < shared_sorted_indexes_[my_end+1]){
                no_conflict = true;
            }
            else{
                no_conflict = false;
            }
        }
        else{
            no_conflict = true;
        }
        __syncthreads();
        int nb_sorted = __syncthreads_count(no_conflict);

        while (nb_sorted < block_size)
        {
            /*if(thread_id == 0){
                printf("block_id_x %d block_id_y %d nb_sorted %d\n", block_id_x, block_id_y, nb_sorted);
            }*/
            /*if(nb_sorted == block_size-1 && no_conflict== false){
                printf("value which stuck : block_id_x %d block_id_y %d my_start %d my_end %d [my_end].value %f [my_end+1].value %f [my_end].start %d [my_end+1].start %d [my_end].index %d [my_end+1].index %d\n", block_id_x, block_id_y, my_start, my_end, shared_sorted_indexes_[my_end].value, shared_sorted_indexes_[my_end+1].value, shared_sorted_indexes_[my_end].start, shared_sorted_indexes_[my_end+1].start, shared_sorted_indexes_[my_end].index, shared_sorted_indexes_[my_end+1].index);
            }*/
            //check if the my last value have to be swapped with the fist value of the next thread
            if(my_end < n_kernels*2-1){
                if(shared_sorted_indexes_[my_end] > shared_sorted_indexes_[my_end+1]){
                    index_start_end temp = shared_sorted_indexes_[my_end];
                    shared_sorted_indexes_[my_end] = shared_sorted_indexes_[my_end+1];
                    shared_sorted_indexes_[my_end+1] = temp;
                    local_sorted = false;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            /*//make one iteration of bubble sort
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                    local_sorted = false;
                }
                else{
                    break;
                }
            }*/
            //localy sort the array
            /*local_sorted = false;
            while (!local_sorted)
            {
                local_sorted = true;
                for (int i = my_start; i < my_end; i++)
                {
                    if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                        index_start_end temp = shared_sorted_indexes_[i];
                        shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                        shared_sorted_indexes_[i+1] = temp;
                        local_sorted = false;
                    }
                }
            }*/
            //only need 2 stp, one from top to bottom and one from bottom to top
            for(int i = my_start; i < my_end; i++){
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                }
                else{
                    break;
                }
            }
            for(int i = my_end; i > my_start; i--){
                if(shared_sorted_indexes_[i] < shared_sorted_indexes_[i-1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i-1];
                    shared_sorted_indexes_[i-1] = temp;
                }
                else{
                    break;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //check if my last val smalest than the first value of the next thread
            no_conflict = false;
            if(my_end < n_kernels*2-1){
                if(shared_sorted_indexes_[my_end] < shared_sorted_indexes_[my_end+1]){
                    no_conflict = true;
                }
                else{
                    no_conflict = false;
                }
            }
            else{
                no_conflict = true;
            }
            __syncthreads();
            nb_sorted = __syncthreads_count(no_conflict);
        }
        __syncthreads();
    }
    else{
        //sort the array using only the first thread of the block
        if(thread_id == 0){
            bool sorted = false;
            while (!sorted)
            {
                sorted = true;
                for (int i = 0; i < n_kernels*2-1; i++)
                {
                    if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                        index_start_end temp = shared_sorted_indexes_[i];
                        shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                        shared_sorted_indexes_[i+1] = temp;
                        sorted = false;
                    }
                }
            }
        //printf("sorted value0 %f value1 %f value2 %f value3 %f value4 %f value5 %f\n", shared_sorted_indexes_[0].value, shared_sorted_indexes_[1].value, shared_sorted_indexes_[2].value, shared_sorted_indexes_[3].value, shared_sorted_indexes_[4].value, shared_sorted_indexes_[5].value);
        }
        __syncthreads();
    }
    clock_t end_sorting = clock();
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d sorted !\n", block_id_x, block_id_y);
    }*/
    //render the screen
    clock_t start_rendering = clock();
    array_indexes_type my_x = min_x + threadIdx.x;
    array_indexes_type my_y = min_y + threadIdx.y;
    //point3d src = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
    point3d src;
    //float_double pixel_x = ((((float_double)my_x) / (float_double)width)-0.5)*x_scaling;
    //float_double pixel_y = ((((float_double)my_y) / (float_double)height)-0.5)*y_scaling;
    rotation_matrix dir;
    //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
    ray_info_(src_, src, dir_, dir, my_x, my_y, width, height);
    //point3d temp = render_pixel_my_block(dir,src, kernels, n_kernels, shared_sorted_indexes_,share_stack, nb_curent_kernels);
    //point3d temp = render_pixel_my_block_2(dir,src, kernels, n_kernels, shared_sorted_indexes_,share_stack, nb_curent_kernels);
    //point3d temp = render_pixel_my_block_3(dir,src, kernels, n_kernels, shared_sorted_indexes_,share_stack, nb_curent_kernels);
    point3d temp = render_pixel_my_block_4(dir,src, kernels, n_kernels, shared_sorted_indexes_,share_stack, nb_curent_kernels);
    if(my_x < width && my_y < height){
        //if i am in the screen save the pixel
        screen[my_x * height + my_y] = temp;
        //printf("i %d j %d r %f g %f b %f dir %f %f %f\n", my_x, my_y, screen[my_x * height + my_y].x, screen[my_x * height + my_y].y, screen[my_x * height + my_y].z, dir.x, dir.y, dir.z);
    }
    __syncthreads();
    if(thread_id == 0){
        free(nb_curent_kernels);
        //printf("block_id_x %d block_id_y %d finished\n", block_id_x, block_id_y);
    }
    clock_t end_rendering = clock();
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d time_loading %f time_sorting %f time_rendering %f\n", block_id_x, block_id_y, (float)(end_loading-start_loading)/(float)CLOCKS_PER_SEC, (float)(end_sorting-start_sorting)/(float)CLOCKS_PER_SEC, (float)(end_rendering-start_rendering)/(float)CLOCKS_PER_SEC);
        printf("block_id_x %d block_id_y %d cycles_loading %ld cycles_sorting %ld cycles_rendering %ld\n", block_id_x, block_id_y, end_loading-start_loading, end_sorting-start_sorting, end_rendering-start_rendering);
        printf("block_id_x %d block_id_y %d cycle_loading_start %ld cycle_loading_end %ld cycle_sorting_start %ld cycle_sorting_end %ld cycle_rendering_start %ld cycle_rendering_end %ld\n", block_id_x, block_id_y, start_loading, end_loading, start_sorting, end_sorting, start_rendering, end_rendering);
    }*/
}

__global__
void render_screen_block_my_2(point3d *screen, point3d src_,rotation_matrix dir_, visual_gaussian_kernel *kernels, int n_kernels, int width, int height){
    //shared momory for the block to save the kernel order
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const array_indexes_type block_id_x = blockIdx.x;
    const array_indexes_type block_id_y = blockIdx.y;
    const array_indexes_type block_size_x = blockDim.x;
    const array_indexes_type block_size_y = blockDim.y;
    const array_indexes_type block_size = block_size_x * block_size_y;
    const array_indexes_type min_x = block_id_x * block_size_x;
    const array_indexes_type min_y = block_id_y * block_size_y;
    const array_indexes_type max_x = min(min_x + block_size_x, width);
    const array_indexes_type max_y = min(min_y + block_size_y, height);
    const float_double middle_x = (float_double)(min_x + max_x) / 2.;
    const float_double middle_y = (float_double)(min_y + max_y) / 2.;

    extern __shared__ index_start_end shared_sorted_indexes_[];
    

    clock_t start_loading = clock();
    //load the values using every thread of the block
    for (int i = thread_id; i < n_kernels; i+=block_size)
    {
        //point3d src = (point3d(middle_x / (float_double)width, middle_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
        point3d src;
        //float_double pixel_x = ((((float_double)middle_x) / (float_double)width)-0.5)*x_scaling;
        //float_double pixel_y = ((((float_double)middle_y) / (float_double)height)-0.5)*y_scaling;
        rotation_matrix dir;
        //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
        ray_info_(src_, src, dir_, dir, middle_x, middle_y, width, height);
        colored_gaussian_kernel_1D kernel = kernels[i].get_1D_kernel(dir,src);
        /*shared_sorted_indexes_[i*2] = index_start_end(i, true, kernel.start());
        shared_sorted_indexes_[i*2+1] = index_start_end(i, false, kernel.end());*/
        float_double min_start = kernel.start();
        float_double max_end = kernel.end();
        /*for(int dx = 0; dx < block_size_x; dx++){
            for(int dy = 0; dy < block_size_y; dy++){
                array_indexes_type my_x = min_x + dx;
                array_indexes_type my_y = min_y + dy;
                if(my_x < width && my_y < height){
                    dir = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
                    kernel = kernels[i].get_1D_kernel(point3d(0, 0, 1), dir);
                    if(kernel.start() < kernel.end()){
                        min_start = min(min_start,kernel.start());
                        max_end = max(max_end, kernel.end());
                    }
                }
            }
        }*/
        shared_sorted_indexes_[i] = index_start_end(i, true, min_start);
        shared_sorted_indexes_[i+n_kernels] = index_start_end(i, false, max_end);
    }
    __syncthreads();
    clock_t end_loading = clock();
    array_indexes_type kernel_block_size = (n_kernels*2)/block_size;
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d min_x %d min_y %d max_x %d max_y %d middle_x %f middle_y %f kernel_block_size %d\n", block_id_x, block_id_y, min_x, min_y, max_x, max_y, middle_x, middle_y, kernel_block_size);
    }*/
    clock_t start_sorting = clock();
    if(kernel_block_size>=2){
        //sort the array using all the threads of the block
        // repartition of kernels is:
        // kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size+1 kernel_block_size kernel_block_size kernel_block_size kernel_block_size

        array_indexes_type my_start = thread_id * (kernel_block_size) + min(thread_id, (n_kernels*2)%(block_size));
        array_indexes_type my_end = min((thread_id+1) * (kernel_block_size) + min(thread_id+1, (n_kernels*2)%(block_size))-1, n_kernels*2-1);

        //localy sort the array
        bool local_sorted = false;
        while (!local_sorted)
        {
            local_sorted = true;
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                    local_sorted = false;
                }
            }
        }
        //syncrhonize the threads
        __syncthreads();
        //check if my last val smalest than the first value of the next thread
        bool no_conflict;
        if(my_end < n_kernels*2-1){
            if(shared_sorted_indexes_[my_end] < shared_sorted_indexes_[my_end+1]){
                no_conflict = true;
            }
            else{
                no_conflict = false;
            }
        }
        else{
            no_conflict = true;
        }
        __syncthreads();
        int nb_sorted = __syncthreads_count(no_conflict);

        while (nb_sorted < block_size)
        {
            /*if(thread_id == 0){
                printf("block_id_x %d block_id_y %d nb_sorted %d\n", block_id_x, block_id_y, nb_sorted);
            }*/
            /*if(nb_sorted == block_size-1 && no_conflict== false){
                printf("value which stuck : block_id_x %d block_id_y %d my_start %d my_end %d [my_end].value %f [my_end+1].value %f [my_end].start %d [my_end+1].start %d [my_end].index %d [my_end+1].index %d\n", block_id_x, block_id_y, my_start, my_end, shared_sorted_indexes_[my_end].value, shared_sorted_indexes_[my_end+1].value, shared_sorted_indexes_[my_end].start, shared_sorted_indexes_[my_end+1].start, shared_sorted_indexes_[my_end].index, shared_sorted_indexes_[my_end+1].index);
            }*/
            //check if the my last value have to be swapped with the fist value of the next thread
            if(my_end < n_kernels*2-1){
                if(shared_sorted_indexes_[my_end] > shared_sorted_indexes_[my_end+1]){
                    index_start_end temp = shared_sorted_indexes_[my_end];
                    shared_sorted_indexes_[my_end] = shared_sorted_indexes_[my_end+1];
                    shared_sorted_indexes_[my_end+1] = temp;
                    local_sorted = false;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            /*//make one iteration of bubble sort
            for (int i = my_start; i < my_end; i++)
            {
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                    local_sorted = false;
                }
                else{
                    break;
                }
            }*/
            //localy sort the array
            /*local_sorted = false;
            while (!local_sorted)
            {
                local_sorted = true;
                for (int i = my_start; i < my_end; i++)
                {
                    if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                        index_start_end temp = shared_sorted_indexes_[i];
                        shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                        shared_sorted_indexes_[i+1] = temp;
                        local_sorted = false;
                    }
                }
            }*/
            //only need 2 stp, one from top to bottom and one from bottom to top
            for(int i = my_start; i < my_end; i++){
                if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                    shared_sorted_indexes_[i+1] = temp;
                }
                else{
                    break;
                }
            }
            for(int i = my_end; i > my_start; i--){
                if(shared_sorted_indexes_[i] < shared_sorted_indexes_[i-1]){
                    index_start_end temp = shared_sorted_indexes_[i];
                    shared_sorted_indexes_[i] = shared_sorted_indexes_[i-1];
                    shared_sorted_indexes_[i-1] = temp;
                }
                else{
                    break;
                }
            }
            //syncrhonize the threads
            __syncthreads();
            //check if my last val smalest than the first value of the next thread
            no_conflict = false;
            if(my_end < n_kernels*2-1){
                if(shared_sorted_indexes_[my_end] < shared_sorted_indexes_[my_end+1]){
                    no_conflict = true;
                }
                else{
                    no_conflict = false;
                }
            }
            else{
                no_conflict = true;
            }
            __syncthreads();
            nb_sorted = __syncthreads_count(no_conflict);
        }
        __syncthreads();
    }
    else{
        //sort the array using only the first thread of the block
        if(thread_id == 0){
            bool sorted = false;
            while (!sorted)
            {
                sorted = true;
                for (int i = 0; i < n_kernels*2-1; i++)
                {
                    if(shared_sorted_indexes_[i] > shared_sorted_indexes_[i+1]){
                        index_start_end temp = shared_sorted_indexes_[i];
                        shared_sorted_indexes_[i] = shared_sorted_indexes_[i+1];
                        shared_sorted_indexes_[i+1] = temp;
                        sorted = false;
                    }
                }
            }
        //printf("sorted value0 %f value1 %f value2 %f value3 %f value4 %f value5 %f\n", shared_sorted_indexes_[0].value, shared_sorted_indexes_[1].value, shared_sorted_indexes_[2].value, shared_sorted_indexes_[3].value, shared_sorted_indexes_[4].value, shared_sorted_indexes_[5].value);
        }
        __syncthreads();
    }
    clock_t end_sorting = clock();
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d sorted !\n", block_id_x, block_id_y);
    }*/
    //render the screen
    clock_t start_rendering = clock();
    array_indexes_type my_x = min_x + threadIdx.x;
    array_indexes_type my_y = min_y + threadIdx.y;
    //point3d src = (point3d(my_x / (float_double)width, my_y / (float_double)height, 0)-point3d(0.5, 0.5, 0))*30.;
    point3d src;
    //float_double pixel_x = ((((float_double)my_x) / (float_double)width)-0.5)*x_scaling;
    //float_double pixel_y = ((((float_double)my_y) / (float_double)height)-0.5)*y_scaling;
    rotation_matrix dir;
    //ray_info(src_, &src, dir_, &dir, pixel_x, pixel_y);
    ray_info_(src_, src, dir_, dir, my_x, my_y, width, height);
    point3d temp = render_pixel_my_block_5(dir, src, kernels, n_kernels, shared_sorted_indexes_);
    if(my_x < width && my_y < height){
        //if i am in the screen save the pixel
        screen[my_x * height + my_y] = temp;
        //printf("i %d j %d r %f g %f b %f dir %f %f %f\n", my_x, my_y, screen[my_x * height + my_y].x, screen[my_x * height + my_y].y, screen[my_x * height + my_y].z, dir.x, dir.y, dir.z);
    }
    __syncthreads();
    clock_t end_rendering = clock();
    /*if(thread_id == 0){
        printf("block_id_x %d block_id_y %d time_loading %f time_sorting %f time_rendering %f\n", block_id_x, block_id_y, (float)(end_loading-start_loading)/(float)CLOCKS_PER_SEC, (float)(end_sorting-start_sorting)/(float)CLOCKS_PER_SEC, (float)(end_rendering-start_rendering)/(float)CLOCKS_PER_SEC);
        printf("block_id_x %d block_id_y %d cycles_loading %ld cycles_sorting %ld cycles_rendering %ld\n", block_id_x, block_id_y, end_loading-start_loading, end_sorting-start_sorting, end_rendering-start_rendering);
        printf("block_id_x %d block_id_y %d cycle_loading_start %ld cycle_loading_end %ld cycle_sorting_start %ld cycle_sorting_end %ld cycle_rendering_start %ld cycle_rendering_end %ld\n", block_id_x, block_id_y, start_loading, end_loading, start_sorting, end_sorting, start_rendering, end_rendering);
    }*/
}