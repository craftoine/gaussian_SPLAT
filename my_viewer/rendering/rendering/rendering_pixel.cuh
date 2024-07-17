__host__
point3d render_pixel(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels){
    //std::cout << "rendering pixel with " << n_kernels << " kernels" << std::endl;
    colored_gaussian_kernel_1D *sorted_kernels = new colored_gaussian_kernel_1D[n_kernels];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_kernels[i] = kernels[i].get_1D_kernel(dir, src);
        //std::cout << "kernel " << i << " color " << sorted_kernels[i].color.x << " " << sorted_kernels[i].color.y << " " << sorted_kernels[i].color.z << std::endl;
    }
    point3d result = point3d(0, 0, 0);
    array_indexes_type * sorted_indexes = new array_indexes_type[n_kernels];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_indexes[i] = i;
    }
    //sort the array of kernels by the distance to the source
    //bubble sort for testing purposes
    bool not_sorted = false;
    for (int i = 0; i < n_kernels; i++)
    {   
        bool swapped = false;
        for (int j = 0; j < n_kernels - i - 1; j++)
        {
            if (sorted_kernels[j].mu > sorted_kernels[j+1].mu)
            {
                colored_gaussian_kernel_1D temp = sorted_kernels[j];
                sorted_kernels[j] = sorted_kernels[j+1];
                sorted_kernels[j+1] = temp;
                array_indexes_type temp_index = sorted_indexes[j];
                sorted_indexes[j] = sorted_indexes[j+1];
                sorted_indexes[j+1] = temp_index;
                swapped = true;
                not_sorted = true;
            }
        }
        if(!swapped){
            break;
        }
    }
    if (not_sorted)
    {
        //reorder the array : kernels
        visual_gaussian_kernel *kernels_ = new visual_gaussian_kernel[n_kernels];
        for (int i = 0; i < n_kernels; i++)
        {
            kernels_[i] = kernels[i];
        }
        for (int i = 0; i < n_kernels; i++)
        {
            kernels[i] = kernels_[sorted_indexes[i]];
        }
        free(kernels_);       
    }


    float_double T = 1;
    for (int i = 0; i < n_kernels && T >stop_T ; i++)
    {   
        //std::cout << "i " << i << " T " << T << std::endl;
        float_double a = sorted_kernels[i].start();
        float_double b = sorted_kernels[i].end();
        a = max(0.,a);
        if(b>a){
            float_double integral = sorted_kernels[i].get_integral(min((float_double)0,a), b);
            //std::cout << "integral " << integral << std::endl;
            result = result + sorted_kernels[i].color * integral * T;
            T = T * (1 - integral);
            T = max(0.0f,T);
        }
    }
    result =result+ background_color * T;

    return result;
}

__host__
point3d render_pixel_mix(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels){
    //std::cout << "rendering pixel with " << n_kernels << " kernels" << std::endl;
    colored_gaussian_kernel_1D *sorted_kernels = new colored_gaussian_kernel_1D[n_kernels];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_kernels[i] = kernels[i].get_1D_kernel(dir, src);
        //std::cout << "kernel " << i << " color " << sorted_kernels[i].color.x << " " << sorted_kernels[i].color.y << " " << sorted_kernels[i].color.z << std::endl;
    }
    point3d result = point3d(0, 0, 0);
    array_indexes_type * sorted_indexes = new array_indexes_type[n_kernels];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_indexes[i] = i;
    }
    //sort the array of kernels by the distance to the source
    //bubble sort for testing purposes
    bool not_sorted = false;
    for (int i = 0; i < n_kernels; i++)
    {
        bool swapped = false;
        for (int j = 0; j < n_kernels - i - 1; j++)
        {
            if (sorted_kernels[j].mu > sorted_kernels[j+1].mu)
            {
                colored_gaussian_kernel_1D temp = sorted_kernels[j];
                sorted_kernels[j] = sorted_kernels[j+1];
                sorted_kernels[j+1] = temp;
                array_indexes_type temp_index = sorted_indexes[j];
                sorted_indexes[j] = sorted_indexes[j+1];
                sorted_indexes[j+1] = temp_index;
                swapped = true;
                not_sorted = true;
            }
        }
        if(!swapped){
            break;
        }
    }

    if (not_sorted)
    {
        //reorder the array : kernels
        visual_gaussian_kernel *kernels_ = new visual_gaussian_kernel[n_kernels];
        for (int i = 0; i < n_kernels; i++)
        {
            kernels_[i] = kernels[i];
        }
        for (int i = 0; i < n_kernels; i++)
        {
            kernels[i] = kernels_[sorted_indexes[i]];
        }
        free(kernels_);       
    }


    float_double T = 1;
    for (int i = 0; i < n_kernels && T >stop_T; i++)
    {
        /*float_double a = sorted_kernels[i].start();
        float_double b = sorted_kernels[i].end();
    

        //more prceise version
        a = max(0.,a);
        if(b>a){
            float_double eval = sorted_kernels[i].get_integral(a, b);
            float_double alpha = 1- exp(-eval);
            result = result + sorted_kernels[i].color * alpha * T;
            T = T * (1- alpha);
        }*/
        float_double eval = sorted_kernels[i].get_integral(0, INFINITY);
        float_double alpha = 1- exp(-eval);
        result = result + sorted_kernels[i].color * alpha * T;
        T = T * (1- alpha);
    }
    result =result+ background_color * T;
    return result;
}

__host__
point3d render_pixel_my(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels){
    colored_gaussian_kernel_1D *sorted_kernels = new colored_gaussian_kernel_1D[n_kernels];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_kernels[i] = kernels[i].get_1D_kernel(dir, src);
        //std::cout << "kernel " << i << " color " << sorted_kernels[i].color.x << " " << sorted_kernels[i].color.y << " " << sorted_kernels[i].color.z << std::endl;
    }
    index_start_end *sorted_indexes = new index_start_end[n_kernels*2];
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_indexes[i*2] = index_start_end(i, true, sorted_kernels[i].start());
        sorted_indexes[i*2+1] = index_start_end(i, false, sorted_kernels[i].end());
    }
    //sort the array
    //bubble sort for testing purposes
    bool not_sorted = false;
    for (int i = 0; i < n_kernels*2; i++)
    {
        bool swapped = false;
        for (int j = 0; j < n_kernels*2 - i - 1; j++)
        {
            if (sorted_indexes[j] > sorted_indexes[j+1])
            {
                index_start_end temp = sorted_indexes[j];
                sorted_indexes[j] = sorted_indexes[j+1];
                sorted_indexes[j+1] = temp;
                swapped = true;
                if(sorted_indexes[j].start && sorted_indexes[j+1].start){
                    not_sorted = true;
                }
            }
        }
        if(!swapped){
            break;
        }
    }

    if (not_sorted)
    {
        //reorder the array : kernels according to them start
        visual_gaussian_kernel *kernels_ = new visual_gaussian_kernel[n_kernels];
        for (int i = 0; i < n_kernels; i++)
        {
            kernels_[i] = kernels[i];
        }
        int j = 0;
        for (int i = 0; i < n_kernels*2; i++)
        {
            if(sorted_indexes[i].start){
                kernels[j] = kernels_[sorted_indexes[i].index];
                j++;
            }
        }
        free(kernels_);                 
    }

    array_indexes_type nb_curent_kernels = 0;
    array_indexes_type max_nb_curent_kernels = 0;
    array_indexes_type * current_kernels = new array_indexes_type[n_kernels];
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    for (int i = 0; i < n_kernels*2 && T >stop_T; i++)
    {
        float_double b = sorted_indexes[i].value;
        if(b>a && nb_curent_kernels>0){
            //fistly count if we have only one kernel in the window if it is the case we only need one step
            if(nb_curent_kernels == 1){
                float_double eval = sorted_kernels[current_kernels[0]].get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + sorted_kernels[current_kernels[0]].color * alpha * T;
                T = T * (1- alpha);
            }
            else{
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    /*float_double x = a + (b - a) * j / k_steps;
                    float_double sum_eval = 0;
                    for (int l = 0; l < nb_curent_kernels; l++)
                    {
                        float_double eval = sorted_kernels[current_kernels[l]](x);
                        float_double alpha = 1- exp(-eval * (b - a) / k_steps);
                        sum_eval += eval;
                        result = result + sorted_kernels[current_kernels[l]].color * alpha * T;
                    }
                    float_double alpha_tot = 1- exp(-sum_eval * (b - a) / k_steps);
                    T = T * (1 - alpha_tot);*/

                    //more prceise version
                    float_double x = a + (b - a) * j / k_steps;
                    float_double sum_eval = 0.;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < nb_curent_kernels; l++)
                    {
                        float_double eval = sorted_kernels[current_kernels[l]].get_integral(x, x+((b - a) / k_steps));
                        sum_eval += eval;
                        color_add = color_add + sorted_kernels[current_kernels[l]].color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    } 
                }
            }

        }
        else{
            /*if(b<a){
                printf("error in the sorting of the kernels\n");
            }*/
        }
        a = max(0.,b);
        if(sorted_indexes[i].start){
            current_kernels[nb_curent_kernels] = sorted_indexes[i].index;
            nb_curent_kernels++;
            max_nb_curent_kernels = max(max_nb_curent_kernels, nb_curent_kernels);
        }else{
            for (int l = 0; l < nb_curent_kernels; l++)
            {
                if(current_kernels[l] == sorted_indexes[i].index){
                    for (int m = l; m < nb_curent_kernels - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                    }
                    nb_curent_kernels--;
                    break;
                }
            }
        }
    }

    result =result+ background_color * T;
    //std::cout << "max_nb_curent_kernels " << max_nb_curent_kernels << std::endl;
    return result;
}

__host__ __device__
point3d render_pixel_block(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_value* sorted_indexes){
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    for (int i = 0; i < n_kernels && T >stop_T ; i++)
    {
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i].index].get_1D_kernel(dir, src);
        /*float_double a = current_kernel.start();
        float_double b = current_kernel.end();
        a = max(0.,a);
        if(b>a){
            float_double integral = current_kernel.get_integral(min((float_double)0,a), b);
            result = result + current_kernel.color * integral * T;
            T = T * (1 - integral);
            T = max(0.0f,T);
        }*/
        //float_double6 new_sigma = kernels[sorted_indexes[i].index].get_reoriented_sigma(dir);
        //loat_double6 precomputed_reoriented_sigma = kernels[sorted_indexes[i].index].get_reoriented_sigma(dir);
        rotation_matrix new_r = rotation_matrix(kernels[sorted_indexes[i].index].kernel.quaternions4)*(dir.transpose());
        float_double M[3][3];
        float_double6 precomputed_reoriented_sigma;
        M[0][0] = kernels[sorted_indexes[i].index].kernel.scales3.x*new_r[0];
        M[1][0] = kernels[sorted_indexes[i].index].kernel.scales3.y*new_r[3];
        M[2][0] = kernels[sorted_indexes[i].index].kernel.scales3.z*new_r[6];
        precomputed_reoriented_sigma[0] = M[0][0]*M[0][0] + M[1][0]*M[1][0] + M[2][0]*M[2][0];
        M[0][1] = kernels[sorted_indexes[i].index].kernel.scales3.x*new_r[1];
        M[1][1] = kernels[sorted_indexes[i].index].kernel.scales3.y*new_r[4];
        M[2][1] = kernels[sorted_indexes[i].index].kernel.scales3.z*new_r[7];
        precomputed_reoriented_sigma[1] = M[0][0]*M[0][1] + M[1][0]*M[1][1] + M[2][0]*M[2][1];
        M[0][2] = kernels[sorted_indexes[i].index].kernel.scales3.x*new_r[2];
        M[1][2] = kernels[sorted_indexes[i].index].kernel.scales3.y*new_r[5];
        M[2][2] = kernels[sorted_indexes[i].index].kernel.scales3.z*new_r[8];
        precomputed_reoriented_sigma[2] = M[0][0]*M[0][2] + M[1][0]*M[1][2] + M[2][0]*M[2][2];
        precomputed_reoriented_sigma[3] = M[0][1]*M[0][1] + M[1][1]*M[1][1] + M[2][1]*M[2][1];
        precomputed_reoriented_sigma[4] = M[0][1]*M[0][2] + M[1][1]*M[1][2] + M[2][1]*M[2][2];
        precomputed_reoriented_sigma[5] = M[0][2]*M[0][2] + M[1][2]*M[1][2] + M[2][2]*M[2][2];
        
        float_double xx;
        float_double xy;
        float_double yy;
        float_double det = precomputed_reoriented_sigma[0]*precomputed_reoriented_sigma[3]-precomputed_reoriented_sigma[1]*precomputed_reoriented_sigma[1];
        float_double inv_det = 1/det;
        xx = precomputed_reoriented_sigma[3]*inv_det;
        xy = -precomputed_reoriented_sigma[1]*inv_det;
        yy = precomputed_reoriented_sigma[0]*inv_det;
        /*xx = precomputed_reoriented_sigma[0];
        xy = precomputed_reoriented_sigma[1];
        yy = precomputed_reoriented_sigma[3];*/
        
        
        
        
        point3d shifted_mu = kernels[sorted_indexes[i].index].kernel.mu;
        shifted_mu.x = shifted_mu.x - src.x;
        shifted_mu.y = shifted_mu.y - src.y;
        shifted_mu.z = shifted_mu.z - src.z;
        shifted_mu = dir*shifted_mu;
        if(shifted_mu.z < 0){
            continue;
        }
        float_double x,y;
        float_double lx,ly;
        /*lx = sqrt(shifted_mu.x*shifted_mu.x+shifted_mu.z*shifted_mu.z);
        x = shifted_mu.x * lx / shifted_mu.z;
        ly = sqrt(shifted_mu.y*shifted_mu.y+shifted_mu.z*shifted_mu.z);
        y = shifted_mu.y * ly / shifted_mu.z;*/
        x = shifted_mu.x;
        y = shifted_mu.y;
        /*float_double l = sqrt(shifted_mu.x*shifted_mu.x+shifted_mu.y*shifted_mu.y+shifted_mu.z*shifted_mu.z);
        x = shifted_mu.x*l / shifted_mu.z;
        y = shifted_mu.y*l / shifted_mu.z;*/
        
        float_double eval = exp(-0.5*(xx*x*x + yy*y*y + 2*xy*x*y)+kernels[sorted_indexes[i].index].kernel.log_weight);
        //eval *= sqrt(new_sigma[0]*new_sigma[3]-new_sigma[1]*new_sigma[1])/(2*M_PI);
        eval = min(0.99,eval);
        float_double alpha = eval;
        /*float_double T_test = T*(1-alpha);
        if (T_test < stop_T)
        {
            continue;
        }*/
        result = result + current_kernel.color * alpha * T;
        T = T * (1- alpha);

    }
    result =result+ background_color * T;
    return result;
}

__device__
point3d render_pixel_mix_block(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_value* sorted_indexes){
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    for (int i = 0; i < n_kernels && T >stop_T; i++)
    {   
        //printf("sorted_indexes[i].index %d\n", sorted_indexes[i].index);
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i].index].get_1D_kernel(dir, src);
        float_double a = current_kernel.start();
        float_double b = current_kernel.end();
        a = max(0.,a);/*
        if(b>a){
            //dont need many steps as it's only one unicolored stuff the formula is exact
            float_double eval = current_kernel.get_integral(a, b);
            float_double alpha = 1- exp(-eval);
            result = result + current_kernel.color * alpha * T;
            T = T * (1- alpha);
        }*/
        //float_double eval = current_kernel.get_integral(0, INFINITY);
        /*if(current_kernel.mu < 0){
            continue;
        }*/
        if(b>a){
        float_double eval = current_kernel.get_integral(-INFINITY, INFINITY);
        float_double alpha = 1- exp(-eval);
        result = result + current_kernel.color * alpha * T;
        T = T * (1- alpha);
        }
        /*if(threadIdx.x ==0 && threadIdx.y == 0){
            printf("block_x %d block_y %d curent_kernel %d log_weight %f\n", blockIdx.x, blockIdx.y, sorted_indexes[i].index, current_kernel.log_weight);
        }*/
    }
    result =result+ background_color * T;
    return result;
}

__device__
inline point3d render_pixel_my_block(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes,array_indexes_type * current_kernels, array_indexes_type * nb_curent_kernels){
    /*printf("try to alocate %d\n", n_kernels*sizeof(colored_gaussian_kernel_1D));
    colored_gaussian_kernel_1D *sorted_kernels = new colored_gaussian_kernel_1D[n_kernels];
    if(sorted_kernels != NULL){
        printf("memory adress: %p\n", sorted_kernels);
    }
    for (int i = 0; i < n_kernels; i++)
    {
        sorted_kernels[i] = kernels[i].get_1D_kernel(dir, src);
    }*/
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    //printf("thread_id %d waiting count\n", thread_id);
    bool global_done = __syncthreads_count(!done) == 0;
    //printf("thread_id %d count done\n", thread_id);
    /*if(thread_id ==0){
        printf("global_done %d\n", global_done);
    }*/
    index_start_end my_reordering_window[reordering_window_size];
    #pragma unroll
    for (int i = 0; i < reordering_window_size; i++)
    {
        my_reordering_window[i] = sorted_indexes[i];
        if(my_reordering_window[i].start){
            colored_gaussian_kernel_1D current_kernel = kernels[my_reordering_window[i].index].get_1D_kernel(dir, src);
            sorted_indexes[i].value = current_kernel.start();
        }
        else{
            colored_gaussian_kernel_1D current_kernel = kernels[my_reordering_window[i].index].get_1D_kernel(dir, src);
            sorted_indexes[i].value = current_kernel.end();
        }
    }

    //sort the reordering window
    #pragma unroll
    for (int i = 0; i < reordering_window_size; i++)
    {
        #pragma unroll
        for (int j = 0; j < reordering_window_size - i - 1; j++)
        {
            if (my_reordering_window[j] > my_reordering_window[j+1])
            {
                index_start_end temp = my_reordering_window[j];
                my_reordering_window[j] = my_reordering_window[j+1];
                my_reordering_window[j+1] = temp;
            }
        }
    }
    int * counter;
    if(thread_id == 0){
        counter = new int[n_kernels];
        //already push the reordering_window_size first kernels in the stack
        #pragma unroll
        for (int i = 0; i < reordering_window_size; i++)
        {
            if(my_reordering_window[i].start){
                current_kernels[(*nb_curent_kernels)] = my_reordering_window[i].index;
                counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                (*nb_curent_kernels)++;
            }
        }
    }
    
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        //printf("thread_id %d i %d hi\n", thread_id, i);
        //float_double b = sorted_indexes[i].value;
        index_start_end b_index = my_reordering_window[0];
        #pragma unroll
        for (int k = 0; k < reordering_window_size - 1; k++)
        {
            my_reordering_window[k] = my_reordering_window[k+1];
        }
        if(i + reordering_window_size < n_kernels*2){
            my_reordering_window[reordering_window_size - 1] = sorted_indexes[i + reordering_window_size];
            if(my_reordering_window[reordering_window_size - 1].start){
                colored_gaussian_kernel_1D current_kernel = kernels[my_reordering_window[reordering_window_size - 1].index].get_1D_kernel(dir, src);
                sorted_indexes[i + reordering_window_size].value = current_kernel.start();
            }
            else{
                colored_gaussian_kernel_1D current_kernel = kernels[my_reordering_window[reordering_window_size - 1].index].get_1D_kernel(dir, src);
                sorted_indexes[i + reordering_window_size].value = current_kernel.end();
            }
            //sort the reordering window
            #pragma unroll
            for (int l = 0; l < reordering_window_size; l++)
            {
                #pragma unroll
                for (int j = 0; j < reordering_window_size - l - 1; j++)
                {
                    if (my_reordering_window[j] > my_reordering_window[j+1])
                    {
                        index_start_end temp = my_reordering_window[j];
                        my_reordering_window[j] = my_reordering_window[j+1];
                        my_reordering_window[j+1] = temp;
                    }
                }
            }
        }
        //printf("thread_id %d i %d hi2\n", thread_id, i);
        //check that b is smaller than the start of the window
        float_double b = b_index.value;
        if(b_index > my_reordering_window[0]){
            b = my_reordering_window[0].value;
            my_reordering_window[0] = b_index;
        }
        if(b>a&& !done){
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            int c = 0;
            array_indexes_type last_index = 0;
            for (int l = 0; l < (*nb_curent_kernels); l++){
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                /*if(current_kernel.start() <= a && current_kernel.end() >= b){
                    c++;
                }*/
                //c+=(current_kernel.start() <= a && current_kernel.end() >= b);
                if(current_kernel.start() <= a && current_kernel.end() >= b){
                    last_index = l;
                    c++;
                }
            }
            if (c == 1)
            {
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[last_index]].get_1D_kernel(dir, src);
                float_double eval = current_kernel.get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + current_kernel.color * alpha * T;
                T = T * (1- alpha);
            }
            else if(c>1){
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double x_ = x+((b - a) / k_steps);
                    if(j == k_steps - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < (*nb_curent_kernels); l++)
                    {
                        colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                        //float_double eval = sorted_kernels[current_kernels[l]].get_integral(x, x+((b - a) / k_steps));
                        float_double eval;
                        if(x>= current_kernel.start() && x< current_kernel.end()){                      
                            eval= current_kernel.get_integral(x, x_);
                        }
                        else{
                            eval = 0;
                        }
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    }
                }
            }

        }
        a = max(0.,b);
        //printf("thread_id %d a %f b %f T %f wa\n", thread_id, a, b, T);
        __syncthreads();
        //printf("thread_id %d a %f b %f T %f do\n", thread_id, a, b, T);
        /*if(thread_id == 0){
            //printf("a %f \n", a);
            if(sorted_indexes[i].start){
                current_kernels[(*nb_curent_kernels)] = sorted_indexes[i].index;
                (*nb_curent_kernels)++;
            }else{
                for (int l = 0; l < (*nb_curent_kernels); l++)
                {
                    if(current_kernels[l] == sorted_indexes[i].index){
                        for (int m = l; m < (*nb_curent_kernels) - 1; m++)
                        {
                            current_kernels[m] = current_kernels[m+1];
                        }
                        (*nb_curent_kernels)--;
                        break;
                    }
                }
            }
        }*/
        for (int l = (*nb_curent_kernels)-1; l >=0; l--)
        {
            int count_remove = __syncthreads_count(!b_index.start && current_kernels[l] == b_index.index);
            if(thread_id == 0){
                counter[l] -= count_remove;
                if (counter[l] == 0)
                {
                    for (int m = l; m < (*nb_curent_kernels) - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                        counter[m] = counter[m+1];
                    }
                    (*nb_curent_kernels)--;
                }
            }
        }
        if(thread_id == 0){
            if(sorted_indexes[i].start){
                current_kernels[(*nb_curent_kernels)] = sorted_indexes[i].index;
                counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                (*nb_curent_kernels)++;
            }
        }
        done = T<=stop_T;
        //printf("thread_id %d color %f %f %f T %f nb_curent_kernels %d\n", thread_id, result.x, result.y, result.z, T, *nb_curent_kernels);
        global_done = __syncthreads_count(!done) == 0;
        /*if(thread_id ==0 && global_done){
            printf("global_done done at step %d\n", i);
        }*/
    }

    result =result+ background_color * T;
    if(thread_id == 0){
        free(counter);
    }
    return result;
}
__device__ 
point3d render_pixel_my_block_2(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes,array_indexes_type * current_kernels_, array_indexes_type * nb_curent_kernels_){
    index_start_end my_sorted_buffer[reordering_window_size];
    array_indexes_type current_kernels[reordering_window_size];
    int nb_curent_kernels = 0;
    #pragma unroll
    for (int i = 0; i < reordering_window_size; i++)
    {
        if(sorted_indexes[i].start){
            current_kernels[nb_curent_kernels] = sorted_indexes[i].index;
            nb_curent_kernels++;
            my_sorted_buffer[i] = index_start_end(sorted_indexes[i].index, true, kernels[sorted_indexes[i].index].get_1D_kernel(dir, src).start());
        }
        else{
            my_sorted_buffer[i] = index_start_end(sorted_indexes[i].index, false, kernels[sorted_indexes[i].index].get_1D_kernel(dir, src).end());
        }        
    }
    //sort the buffer
    #pragma unroll
    for (int i = 0; i < reordering_window_size; i++)
    {
        #pragma unroll
        for (int j = 0; j < reordering_window_size - i - 1; j++)
        {
            if (my_sorted_buffer[j] > my_sorted_buffer[j+1])
            {
                index_start_end temp = my_sorted_buffer[j];
                my_sorted_buffer[j] = my_sorted_buffer[j+1];
                my_sorted_buffer[j+1] = temp;
            }
        }
    }
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    bool global_done = __syncthreads_count(!done) == 0;
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        index_start_end b_index = my_sorted_buffer[0];
        #pragma unroll
        for (int k = 0; k < reordering_window_size - 1; k++)
        {
            my_sorted_buffer[k] = my_sorted_buffer[k+1];
        }
        if(i + reordering_window_size < n_kernels*2){
            my_sorted_buffer[reordering_window_size - 1] = sorted_indexes[i + reordering_window_size];
            if(my_sorted_buffer[reordering_window_size - 1].start){
                colored_gaussian_kernel_1D current_kernel = kernels[my_sorted_buffer[reordering_window_size - 1].index].get_1D_kernel(dir, src);
                sorted_indexes[i + reordering_window_size].value = current_kernel.start();
            }
            else{
                colored_gaussian_kernel_1D current_kernel = kernels[my_sorted_buffer[reordering_window_size - 1].index].get_1D_kernel(dir, src);
                sorted_indexes[i + reordering_window_size].value = current_kernel.end();
            }
            //sort the buffer
            #pragma unroll
            for (int l = 0; l < reordering_window_size; l++)
            {
                #pragma unroll
                for (int j = 0; j < reordering_window_size - l - 1; j++)
                {
                    if (my_sorted_buffer[j] > my_sorted_buffer[j+1])
                    {
                        index_start_end temp = my_sorted_buffer[j];
                        my_sorted_buffer[j] = my_sorted_buffer[j+1];
                        my_sorted_buffer[j+1] = temp;
                    }
                }
            }
        }
        float_double b = b_index.value;
        if(b_index > my_sorted_buffer[0]){
            b = my_sorted_buffer[0].value;
            my_sorted_buffer[0] = b_index;
        }
        if(b>a&& !done){
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            int c = 0;
            array_indexes_type last_index = 0;
            for (int l = 0; l < nb_curent_kernels; l++){
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                //c+=(current_kernel.start() <= a && current_kernel.end() >= b);
                if(current_kernel.start() <= a && current_kernel.end() >= b){
                    last_index = l;
                    c++;
                }
            }
            if (c == 1)
            {
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[last_index]].get_1D_kernel(dir, src);
                float_double eval = current_kernel.get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + current_kernel.color * alpha * T;
                T = T * (1- alpha);
            }
            else if(c>1){
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double x_ = a + (b - a) * (j+1) / k_steps;
                    if(j == k_steps - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < nb_curent_kernels; l++)
                    {
                        colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                        float_double eval= current_kernel.get_integral(x, x_);
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    }
                }
            }
        }
        a = max(0.,b);
        if(b_index.start){
            current_kernels[nb_curent_kernels] = b_index.index;
            nb_curent_kernels++;
        }else{
            for (int l = 0; l < nb_curent_kernels; l++)
            {
                if(current_kernels[l] == b_index.index){
                    for (int m = l; m < nb_curent_kernels - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                    }
                    nb_curent_kernels--;
                    break;
                }
            }
        }
        done = T<=stop_T;
        global_done = __syncthreads_count(!done) == 0;        
    }
    result =result+ background_color * T;
    return result;
}
__device__ 
inline point3d render_pixel_my_block_3(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes,array_indexes_type * current_kernels, array_indexes_type * nb_curent_kernels){
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    if(thread_id == 0){
        (*nb_curent_kernels) = 0;
    }
    bool global_done = __syncthreads_count(!done) == 0;
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        //printf("thread_id %d i %d hi\n", thread_id, i);
        float_double b = sorted_indexes[i].value;
        if(b>a&& !done){
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            int c = 0;
            array_indexes_type last_index = 0;
            for (int l = 0; l < (*nb_curent_kernels); l++){
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                c+=(current_kernel.start() <= a && current_kernel.end() >= b);
            }
            if (c == 1)
            {
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[last_index]].get_1D_kernel(dir, src);
                float_double eval = current_kernel.get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + current_kernel.color * alpha * T;
                T = T * (1- alpha);
            }
            else if(c>1){
                for(int j = 0; j < k_steps_2 && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps_2;
                    float_double x_ = a + (b - a) * (j+1) / k_steps_2;
                    if(j == k_steps_2 - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < (*nb_curent_kernels); l++)
                    {
                        array_indexes_type current_kernel_index = current_kernels[l];
                        /*if(current_kernel_index >= n_kernels){
                            printf("error current_kernel_index %d l %d nb_curent_kernels %d\n", current_kernel_index, l, (*nb_curent_kernels));
                        }*/
                        colored_gaussian_kernel_1D current_kernel = kernels[current_kernel_index].get_1D_kernel(dir, src);
                        //float_double eval = sorted_kernels[current_kernels[l]].get_integral(x, x+((b - a) / k_steps_2));
                        float_double eval;
                        if(x>= current_kernel.start() && x< current_kernel.end()){                      
                            eval= current_kernel.get_integral(x, x_);
                        }
                        else{
                            eval = 0;
                        }
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    }
                }
            }
        }
        a = max(0.,b);
        //printf("thread_id %d a %f b %f T %f wa\n", thread_id, a, b, T);
        __syncthreads();
        //printf("thread_id %d a %f b %f T %f do\n", thread_id, a, b, T);
        if(thread_id == 0){
            //printf("a %f \n", a);
            if(sorted_indexes[i].start){
                current_kernels[(*nb_curent_kernels)] = sorted_indexes[i].index;
                /*if(current_kernels[(*nb_curent_kernels)] >= n_kernels){
                    printf("error current_kernel_index %d, nb_curent_kernels %d in insert\n", current_kernels[(*nb_curent_kernels)], (*nb_curent_kernels));
                }*/
                (*nb_curent_kernels)++;
            }else{
                for (int l = 0; l < (*nb_curent_kernels); l++)
                {
                    if(current_kernels[l] == sorted_indexes[i].index){
                        for (int m = l; m < (*nb_curent_kernels) - 1; m++)
                        {
                            current_kernels[m] = current_kernels[m+1];
                        }
                        (*nb_curent_kernels)--;
                        break;
                    }
                }
            }
        }
        done = T<=stop_T;
        //printf("thread_id %d color %f %f %f T %f nb_curent_kernels %d\n", thread_id, result.x, result.y, result.z, T, *nb_curent_kernels);
        global_done = __syncthreads_count(!done) == 0;
        /*if(thread_id ==0 && global_done){
            printf("global_done done at step %d\n", i);
        }*/
    }

    result =result+ background_color * T;
    return result;
}

__device__
inline point3d render_pixel_my_block_4(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes,array_indexes_type * current_kernels, array_indexes_type * nb_curent_kernels){
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    index_start_end buffer[reordering_window_size];
    array_indexes_type length_buffer = min(n_kernels*2, reordering_window_size);
    //load the first values in the buffer
    //#pragma unroll
    for (int i = 0; i < length_buffer; i++)
    {
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i].index].get_1D_kernel(dir, src);
        if(sorted_indexes[i].start){
            buffer[i] = index_start_end(sorted_indexes[i].index, true, current_kernel.start());
        }
        else{
            buffer[i] = index_start_end(sorted_indexes[i].index, false, current_kernel.end());
        }
    }

    //debugging, check that all indexes are here:
    /*if(thread_id == 0){
        for (int i = 0; i < n_kernels*2; i++)
        {
            bool found = false;
            for (int j = 0; j < length_buffer; j++)
            {
                if(sorted_indexes[i].index == buffer[j].index){
                    found = true;
                    break;
                }
            }
            if(!found){
                printf("error index %d not found\n", sorted_indexes[i].index);

            }
        }                    float_double alpha = 1- exp(-eval);
    }*/
    array_indexes_type* counter;
    if(thread_id == 0){
        //load the first values in the current kernels
        counter = new array_indexes_type[n_kernels];
        (*nb_curent_kernels) = 0;
        //#pragma unroll
        for (int i = 0; i < length_buffer; i++)
        {
            if(buffer[i].start){
                current_kernels[(*nb_curent_kernels)] = buffer[i].index;
                counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                (*nb_curent_kernels)++;
            }
        }
    }
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    bool global_done = __syncthreads_count(!done) == 0;
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        index_start_end b_index = buffer[0];
        array_indexes_type b_index_index = 0;
        //#pragma unroll
        for (int k = 1; k < length_buffer; k++)
        {
            if(buffer[k]< b_index){
                b_index = buffer[k];
                b_index_index = k;
            }
            /*if(buffer[k].value < b_index.value){
                printf("error in the min extraction : buffer %d %d %f b_index %d %d %f\n", buffer[k].index, buffer[k].start, buffer[k].value, b_index.index, b_index.start, b_index.value);
            }*/
        }
        /*if(thread_id == 0 && blockIdx.x == 15 && blockIdx.y == 27){
            //printf("block_id_x %d block_id_y %d b_index_index %d\n", blockIdx.x, blockIdx.y, b_index_index);
            printf("b_index_index %d b_index is val: %f, index: %d , start: %d\n", b_index_index, b_index.value, b_index.index, b_index.start);
        }*/
        if(i + length_buffer < n_kernels*2){
            //printf("ADDING");
            colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i + length_buffer].index].get_1D_kernel(dir, src);
            if(sorted_indexes[i + length_buffer].start){
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, true, current_kernel.start());
            }
            else{
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, false, current_kernel.end());
            }
        }
        else{
            buffer[b_index_index] = buffer[length_buffer - 1];
            length_buffer--;
        }
        float_double b = b_index.value;
        if(b>a&& !done){
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d i %d a %f b %f, (b-a)/k_steps %f\n", blockIdx.x, blockIdx.y, i, a, b, (b - a) / k_steps);
                //printf("nb_curent_kernels %d\n", *nb_curent_kernels);
                for (int l = 0; l < (*nb_curent_kernels); l++)
                {
                    colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                    //printf("current_kernels %d origninal mu %f %f %f\n", current_kernels[l],kernels[current_kernels[l]].kernel.mu.x, kernels[current_kernels[l]].kernel.mu.y, kernels[current_kernels[l]].kernel.mu.z);
                    //printf("current_kernels %d, tacken = %d\n", current_kernels[l], current_kernel.start() <= a && current_kernel.end() >= b);
                    //printf("current_kernel %f %f\n", current_kernel.start(), current_kernel.end());
                    //printf("current_kernel %d, log_weight %f, color %f %f %f\n", current_kernels[l], current_kernel.log_weight, current_kernel.color.x, current_kernel.color.y, current_kernel.color.z);
                }
                //printf("dir %f %f %f %f \n", dir.quaternions4.x, dir.quaternions4.y, dir.quaternions4.z, dir.quaternions4.w);
                //printf("src %f %f %f\n", src.x, src.y, src.z);
                //printf("T %f\n", T);
                //printf("1result %f %f %f\n", result.x, result.y, result.z);
                //printf("b_index %d %d %f\n", b_index.index, b_index.start, b_index.value);
                //result = point3d(0, 0, 1);
                //T = 0;
            }*/
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            int c = 0;
            array_indexes_type last_index = 0;
            for (int l = 0; l < (*nb_curent_kernels); l++){
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                //c+=(current_kernel.start() <= a && current_kernel.end() >= b);
                if(current_kernel.start() <= a && current_kernel.end() >= b){
                    last_index = l;
                    c++;
                }
            }
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d c %d\n", blockIdx.x, blockIdx.y, c);
            }*/
            if (c == 1)
            {
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[last_index]].get_1D_kernel(dir, src);
                float_double eval = current_kernel.get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                    printf("block_id_x %d block_id_y %d result would be %f %f %f T would be %f last_index %d kernel color %f %f %f \n", blockIdx.x, blockIdx.y, result.x + current_kernel.color.x * alpha * T, result.y + current_kernel.color.y * alpha * T, result.z + current_kernel.color.z * alpha * T, T * (1- alpha),last_index, current_kernel.color.x, current_kernel.color.y, current_kernel.color.z);
                }*/
                result = result + current_kernel.color * alpha * T;
                T = T * (1- alpha);
            }
            else if(c>1){
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double x_ = x+((b - a) / k_steps);
                    if(j == k_steps - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                        printf("block_id_x %d block_id_y %d x %f, x+((b - a) / k_steps) %f\n", blockIdx.x, blockIdx.y, x, x+((b - a) / k_steps));
                    }*/
                    for (int l = 0; l < (*nb_curent_kernels); l++)
                    {
                        colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                        float_double eval;
                        if(x>= current_kernel.start() && x< current_kernel.end()){                      
                            eval= current_kernel.get_integral(x, x_);
                            /*if(c == 1 && thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                                printf("block_id_x %d block_id_y %d current_kernel %d kernel color %f %f %f \n", blockIdx.x, blockIdx.y, l, current_kernel.color.x, current_kernel.color.y, current_kernel.color.z);
                            }*/
                        }
                        else{
                            eval = 0;
                        }
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                        printf("block_id_x %d block_id_y %d sum_eval %f, sum_eval>min_sum_eval %d\n", blockIdx.x, blockIdx.y, sum_eval, sum_eval>min_sum_eval);
                    }*/
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + (color_add /sum_eval)*alpha*T;
                        T = T * (1- alpha);
                    }
                    /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                        printf("T %f\n", T);
                        printf("result %f %f %f\n", result.x, result.y, result.z);
                    }*/
                }
            }
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d result %f %f %f T %f\n", blockIdx.x, blockIdx.y, result.x, result.y, result.z, T);
            }*/
        }
        else{
            /*if(b<a && a!=0.){
                printf("block_id_x %d block_id_y %d error b %f smaller than a %f, b_index was val: %f, index: %d , start: %d\n", blockIdx.x, blockIdx.y, b, a, b_index.value, b_index.index);
            }*/
        }
        __syncthreads();
        a = max(0.,max(a,b));
        for (int l = (*nb_curent_kernels)-1; l >=0; l--)
        {
            int count_remove = __syncthreads_count(!b_index.start && current_kernels[l] == b_index.index);
            if(thread_id == 0){
                counter[l] -= count_remove;
                if (counter[l] == 0)
                {
                    for (int m = l; m < (*nb_curent_kernels) - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                        counter[m] = counter[m+1];
                    }
                    (*nb_curent_kernels)--;
                }
            }
        }
        if(thread_id == 0){
            if(reordering_window_size+i < n_kernels*2){
                if(sorted_indexes[i + reordering_window_size].start){
                    current_kernels[(*nb_curent_kernels)] = sorted_indexes[i + reordering_window_size].index;
                    counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                    (*nb_curent_kernels)++;
                }
            }
        }
        done = T<=stop_T;
        global_done = __syncthreads_count(!done) == 0;
    }
    result =result+ background_color * T;
    /*if(blockIdx.x == 38 && blockIdx.y == 45 && thread_id >= 204 && thread_id <=205){
        printf("block_id_x %d block_id_y %d thread_id %d result %f %f %f\n", blockIdx.x, blockIdx.y, thread_id, result.x, result.y, result.z);
        result = point3d(1, 0, 0);
    }*/
    if(thread_id == 0){
        free(counter);
    }
    return result;
}
__device__
inline point3d render_pixel_my_block_4_2(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes,array_indexes_type * current_kernels, array_indexes_type * nb_curent_kernels){
    const array_indexes_type thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    index_start_end buffer[reordering_window_size];
    array_indexes_type length_buffer = min(n_kernels*2, reordering_window_size);
    //load the first values in the buffer
    //#pragma unroll
    for (int i = 0; i < length_buffer; i++)
    {
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i].index].get_1D_kernel(dir, src);
        if(sorted_indexes[i].start){
            buffer[i] = index_start_end(sorted_indexes[i].index, true, current_kernel.start());
        }
        else{
            buffer[i] = index_start_end(sorted_indexes[i].index, false, current_kernel.end());
        }
    }

    //debugging, check that all indexes are here:
    /*if(thread_id == 0){
        for (int i = 0; i < n_kernels*2; i++)
        {
            bool found = false;
            for (int j = 0; j < length_buffer; j++)
            {
                if(sorted_indexes[i].index == buffer[j].index){
                    found = true;
                    break;
                }
            }
            if(!found){
                printf("error index %d not found\n", sorted_indexes[i].index);

            }
        }                    float_double alpha = 1- exp(-eval);
    }*/
    array_indexes_type* counter;
    if(thread_id == 0){
        //load the first values in the current kernels
        counter = new array_indexes_type[n_kernels];
        (*nb_curent_kernels) = 0;
        //#pragma unroll
        for (int i = 0; i < length_buffer; i++)
        {
            if(buffer[i].start){
                current_kernels[(*nb_curent_kernels)] = buffer[i].index;
                counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                (*nb_curent_kernels)++;
            }
        }
    }
    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    bool global_done = __syncthreads_count(!done) == 0;
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        index_start_end b_index = buffer[0];
        array_indexes_type b_index_index = 0;
        //#pragma unroll
        for (int k = 1; k < length_buffer; k++)
        {
            if(buffer[k]< b_index){
                b_index = buffer[k];
                b_index_index = k;
            }
            /*if(buffer[k].value < b_index.value){
                printf("error in the min extraction : buffer %d %d %f b_index %d %d %f\n", buffer[k].index, buffer[k].start, buffer[k].value, b_index.index, b_index.start, b_index.value);
            }*/
        }
        /*if(thread_id == 0 && blockIdx.x == 15 && blockIdx.y == 27){
            //printf("block_id_x %d block_id_y %d b_index_index %d\n", blockIdx.x, blockIdx.y, b_index_index);
            printf("b_index_index %d b_index is val: %f, index: %d , start: %d\n", b_index_index, b_index.value, b_index.index, b_index.start);
        }*/
        if(i + length_buffer < n_kernels*2){
            //printf("ADDING");
            colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i + length_buffer].index].get_1D_kernel(dir, src);
            if(sorted_indexes[i + length_buffer].start){
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, true, current_kernel.start());
            }
            else{
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, false, current_kernel.end());
            }
        }
        else{
            buffer[b_index_index] = buffer[length_buffer - 1];
            length_buffer--;
        }
        float_double b = b_index.value;
        if(b>a&& !done){
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d i %d a %f b %f, (b-a)/k_steps %f\n", blockIdx.x, blockIdx.y, i, a, b, (b - a) / k_steps);
                //printf("nb_curent_kernels %d\n", *nb_curent_kernels);
                for (int l = 0; l < (*nb_curent_kernels); l++)
                {
                    colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                    //printf("current_kernels %d origninal mu %f %f %f\n", current_kernels[l],kernels[current_kernels[l]].kernel.mu.x, kernels[current_kernels[l]].kernel.mu.y, kernels[current_kernels[l]].kernel.mu.z);
                    //printf("current_kernels %d, tacken = %d\n", current_kernels[l], current_kernel.start() <= a && current_kernel.end() >= b);
                    //printf("current_kernel %f %f\n", current_kernel.start(), current_kernel.end());
                    //printf("current_kernel %d, log_weight %f, color %f %f %f\n", current_kernels[l], current_kernel.log_weight, current_kernel.color.x, current_kernel.color.y, current_kernel.color.z);
                }
                //printf("dir %f %f %f %f \n", dir.quaternions4.x, dir.quaternions4.y, dir.quaternions4.z, dir.quaternions4.w);
                //printf("src %f %f %f\n", src.x, src.y, src.z);
                //printf("T %f\n", T);
                //printf("1result %f %f %f\n", result.x, result.y, result.z);
                //printf("b_index %d %d %f\n", b_index.index, b_index.start, b_index.value);
                //result = point3d(0, 0, 1);
                //T = 0;
            }*/
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            int c = 0;
            array_indexes_type last_index = 0;
            for (int l = 0; l < (*nb_curent_kernels); l++){
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                //c+=(current_kernel.start() <= a && current_kernel.end() >= b);
                if(current_kernel.start() <= a && current_kernel.end() >= b){
                    last_index = l;
                    c++;
                }
            }
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d c %d\n", blockIdx.x, blockIdx.y, c);
            }*/
            if (c == 1)
            {
                colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[last_index]].get_1D_kernel(dir, src);
                float_double eval = current_kernel.get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                    printf("block_id_x %d block_id_y %d result would be %f %f %f T would be %f last_index %d kernel color %f %f %f \n", blockIdx.x, blockIdx.y, result.x + current_kernel.color.x * alpha * T, result.y + current_kernel.color.y * alpha * T, result.z + current_kernel.color.z * alpha * T, T * (1- alpha),last_index, current_kernel.color.x, current_kernel.color.y, current_kernel.color.z);
                }*/
                result = result + current_kernel.color * alpha * T;
                T = T * (1- alpha);
            }
            else if(c>1){
                /*for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < (*nb_curent_kernels); l++)
                    {
                        colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                        float_double eval;
                        if(a>= current_kernel.start() && b<= current_kernel.end()){                      
                            eval= current_kernel.get_integral(x, x+((b - a) / k_steps));
                        }
                        else{
                            eval = 0;
                        }
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + (color_add /sum_eval)*alpha*T;
                        T = T * (1- alpha);
                    }
                }*/
                float_double sum_eval[k_steps];
                point3d color_add[k_steps];
                for(int j = 0; j < k_steps; j++){
                    sum_eval[j] = 0;
                    color_add[j] = point3d(0, 0, 0);
                }
                for (int l = 0; l < (*nb_curent_kernels); l++)
                {
                    colored_gaussian_kernel_1D current_kernel = kernels[current_kernels[l]].get_1D_kernel(dir, src);
                    if(current_kernel.start() <= a && current_kernel.end() >= b){
                        for(int j = 0; j < k_steps; j++){
                            float_double x = a + (b - a) * j / k_steps;
                            float_double x_ = a + (b - a) * (j+1) / k_steps;
                            if(j == k_steps-1){
                                x_ = b;
                            }
                            float_double eval = current_kernel.get_integral(x, x_);
                            sum_eval[j] += eval;
                            color_add[j] = color_add[j] + current_kernel.color * eval;
                        }
                    }
                }
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    if(sum_eval[j]>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval[j]);
                        result = result + (color_add[j] /sum_eval[j])*alpha*T;
                        T = T * (1- alpha);
                    }
                }
            }
            /*if(thread_id == 0 && n_kernels == 2 && ((false && blockIdx.y == 23 && blockIdx.x ==22)||(blockIdx.y == 22 && blockIdx.x ==22))){
                printf("block_id_x %d block_id_y %d result %f %f %f T %f\n", blockIdx.x, blockIdx.y, result.x, result.y, result.z, T);
            }*/
        }
        else{
            /*if(b<a && a!=0.){
                printf("block_id_x %d block_id_y %d error b %f smaller than a %f, b_index was val: %f, index: %d , start: %d\n", blockIdx.x, blockIdx.y, b, a, b_index.value, b_index.index);
            }*/
        }
        __syncthreads();
        a = max(0.,max(a,b));
        for (int l = (*nb_curent_kernels)-1; l >=0; l--)
        {
            int count_remove = __syncthreads_count(!b_index.start && current_kernels[l] == b_index.index);
            if(thread_id == 0){
                counter[l] -= count_remove;
                if (counter[l] == 0)
                {
                    for (int m = l; m < (*nb_curent_kernels) - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                        counter[m] = counter[m+1];
                    }
                    (*nb_curent_kernels)--;
                }
            }
        }
        if(thread_id == 0){
            if(reordering_window_size+i < n_kernels*2){
                if(sorted_indexes[i + reordering_window_size].start){
                    current_kernels[(*nb_curent_kernels)] = sorted_indexes[i + reordering_window_size].index;
                    counter[(*nb_curent_kernels)] = blockDim.x*blockDim.y;
                    (*nb_curent_kernels)++;
                }
            }
        }
        done = T<=stop_T;
        global_done = __syncthreads_count(!done) == 0;
    }
    result =result+ background_color * T;
    /*if(blockIdx.x == 38 && blockIdx.y == 45 && thread_id >= 204 && thread_id <=205){
        printf("block_id_x %d block_id_y %d thread_id %d result %f %f %f\n", blockIdx.x, blockIdx.y, thread_id, result.x, result.y, result.z);
        result = point3d(1, 0, 0);
    }*/
    if(thread_id == 0){
        free(counter);
    }
    return result;
}
__device__
inline point3d render_pixel_my_block_5(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes){
    index_start_end buffer[reordering_window_size];
    colored_gaussian_kernel_1D current_kernels[max_overlap];
    array_indexes_type current_kernel_indexes[max_overlap];
    array_indexes_type length_buffer = min(n_kernels*2, reordering_window_size);
    array_indexes_type nb_curent_kernels = 0;
    //load the first values in the buffer
    //#pragma unroll
    for (int i = 0; i < length_buffer; i++)
    {
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i].index].get_1D_kernel(dir, src);
        if(sorted_indexes[i].start){
            buffer[i] = index_start_end(sorted_indexes[i].index, true, current_kernel.start());
        }
        else{
            buffer[i] = index_start_end(sorted_indexes[i].index, false, current_kernel.end());
        }
    }


    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    bool done = T<=stop_T;
    bool global_done = __syncthreads_count(!done) == 0;
    for (int i = 0; i < n_kernels*2 && !global_done; i++)
    {
        index_start_end b_index = buffer[0];
        array_indexes_type b_index_index = 0;
        //#pragma unroll
        for (int k = 1; k < length_buffer; k++)
        {
            if(buffer[k]< b_index){
                b_index = buffer[k];
                b_index_index = k;
            }
        }
        if(i + length_buffer < n_kernels*2){
            colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[i + length_buffer].index].get_1D_kernel(dir, src);
            if(sorted_indexes[i + length_buffer].start){
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, true, current_kernel.start());
            }
            else{
                buffer[b_index_index] = index_start_end(sorted_indexes[i + length_buffer].index, false, current_kernel.end());
            }
        }
        else{
            buffer[b_index_index] = buffer[length_buffer - 1];
            length_buffer--;
        }
        float_double b = b_index.value;
        if(b>a&& !done){
            if(nb_curent_kernels == 1)
            {
                float_double eval = current_kernels[0].get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + current_kernels[0].color * alpha * T;
                T = T * (1- alpha);
            }
            else if(nb_curent_kernels>1){
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double x_ = x+((b - a) / k_steps);
                    if(j == k_steps - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < nb_curent_kernels; l++){
                        colored_gaussian_kernel_1D current_kernel = current_kernels[l];
                        float_double eval;            
                        eval= current_kernel.get_integral(x, x_);
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                    }
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    }

                }
            }
        }
        a = max(0.,max(a,b));
        if(b_index.start){
            if(nb_curent_kernels < max_overlap){
                current_kernels[nb_curent_kernels] = kernels[b_index.index].get_1D_kernel(dir, src);
                current_kernel_indexes[nb_curent_kernels] = b_index.index;
                nb_curent_kernels++;
            }
            else{
                //printf("error nb_curent_kernels %d is bigger than max_overlap %d\n", nb_curent_kernels, max_overlap);
                /*//find the one which will finish the earlier and replace it
                float_double min_end = current_kernels[0].end();
                array_indexes_type min_end_index = 0;
                for (int l = 1; l < nb_curent_kernels; l++)
                {
                    if(current_kernels[l].end() < min_end){
                        min_end = current_kernels[l].end();
                        min_end_index = l;
                    }
                }
                colored_gaussian_kernel_1D b_kernel = kernels[b_index.index].get_1D_kernel(dir, src);
                if(b_kernel.end() > min_end){
                    current_kernels[min_end_index] = b_kernel;
                    current_kernel_indexes[min_end_index] = b_index.index;
                }*/
                //find the one which have the smalest integral left and replace it
                float_double min_integral = current_kernels[0].get_integral(a, current_kernels[0].end());
                array_indexes_type min_integral_index = 0;
                for (int l = 1; l < nb_curent_kernels; l++)
                {
                    float_double integral = current_kernels[l].get_integral(a, current_kernels[l].end());
                    if(integral < min_integral){
                        min_integral = integral;
                        min_integral_index = l;
                    }
                }
                colored_gaussian_kernel_1D b_kernel = kernels[b_index.index].get_1D_kernel(dir, src);
                float_double b_integral = b_kernel.get_integral(a, b_kernel.end());
                if(b_integral > min_integral){
                    current_kernels[min_integral_index] = b_kernel;
                    current_kernel_indexes[min_integral_index] = b_index.index;
                }

            }
        }
        else{
            for (int l = 0; l < nb_curent_kernels; l++)
            {
                if(current_kernel_indexes[l] == b_index.index){
                    for (int m = l; m < nb_curent_kernels - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                        current_kernel_indexes[m] = current_kernel_indexes[m+1];
                    }
                    nb_curent_kernels--;
                    break;
                }
            }
        }
        done = T<=stop_T;
        global_done = __syncthreads_count(!done) == 0;
    }

    result =result+ background_color * T;
    return result;
}
__device__
inline point3d render_pixel_my_block_5_b(rotation_matrix dir, point3d src, visual_gaussian_kernel *kernels, int n_kernels, index_start_end * sorted_indexes){
    /*if(threadIdx.x == 0 && threadIdx.y == 0){
        printf("src %f %f %f dir %f %f %f %f %f %f %f %f %f\n", src.x, src.y, src.z, dir[0],dir[1],dir[2],dir[3],dir[4],dir[5],dir[6],dir[7],dir[8]);
    }*/
    //printf("sorted_indexes %d %d %f\n", sorted_indexes[0].index, sorted_indexes[0].start, sorted_indexes[0].value);
    index_start_end buffer[reordering_window_size];
    colored_gaussian_kernel_1D current_kernels[max_overlap];
    array_indexes_type current_kernel_indexes[max_overlap];
    array_indexes_type length_buffer = 0;
    array_indexes_type nb_curent_kernels = 0;
    //load the first values in the buffer
    //#pragma unroll
    array_indexes_type last_tacken = 0;
    for (int i = 0; i < min(n_kernels*2, reordering_window_size) && (last_tacken < n_kernels*2); i++)
    {
        last_tacken++;
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[last_tacken-1].index].get_1D_kernel(dir, src);
        float_double st, en;
        st = current_kernel.start();
        en = current_kernel.end();
        //en+=0.0001;
        while((st == en|| current_kernel.get_integral() <= min_integral_v) && last_tacken < n_kernels*2){
            last_tacken++;
            current_kernel = kernels[sorted_indexes[last_tacken-1].index].get_1D_kernel(dir, src);
            st = current_kernel.start();
            en = current_kernel.end();
            //en+=0.0001;
        }
        if(last_tacken <= n_kernels*2 && st < en && current_kernel.get_integral() > min_integral_v){
            //printf("%f\n",current_kernel.get_integral());
            //if(current_kernel.get_integral() > 1.){
            //printf("block_id_x %d block_id_y %d, log_weight %f, sigma %f , mu %f, old scale %f %f %f\n", blockIdx.x, blockIdx.y, current_kernel.log_weight, current_kernel.sigma, current_kernel.mu, kernels[sorted_indexes[last_tacken-1].index].kernel.scales3.x, kernels[sorted_indexes[last_tacken-1].index].kernel.scales3.y, kernels[sorted_indexes[last_tacken-1].index].kernel.scales3.z);
            //}//
            //float_double6 sigma_origin = kernels[sorted_indexes[last_tacken-1].index].get_reoriented_sigma(dir);
            //printf("sigma_origin %f %f %f %f %f %f\n", sigma_origin[0], sigma_origin[1], sigma_origin[2], sigma_origin[3], sigma_origin[4], sigma_origin[5]);
            /*if((threadIdx.x >= 8 && threadIdx.x <= 9) && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
                float_double6 sigma_origin = kernels[sorted_indexes[last_tacken-1].index].get_reoriented_sigma(dir);
                printf("threadIdx.x %d , sigma_origin %f %f %f %f %f %f\n", threadIdx.x, sigma_origin[0], sigma_origin[1], sigma_origin[2], sigma_origin[3], sigma_origin[4], sigma_origin[5]);
                //printf("threadIdx.x %d , log_weight %f \n", threadIdx.x, current_kernel.log_weight);
                //printf("threadIdx.x %d ,integral %f \n", threadIdx.x, current_kernel.get_integral());
            }*/
            
            length_buffer++;
            if(sorted_indexes[last_tacken-1].start){
                buffer[i] = index_start_end(sorted_indexes[last_tacken-1].index, true, st);
            }
            else{
                buffer[i] = index_start_end(sorted_indexes[last_tacken-1].index, false, en);
            }
        }
        
    }


    point3d result = point3d(0, 0, 0);
    float_double T = 1;
    float_double a = 0;
    /*if(((threadIdx.x <= 1 && threadIdx.y <= 0)|| false) && ((blockIdx.x == 30 && blockIdx.y == 5)|| false)){
        printf("threadIdx.x %d start %d index %d value %f\n", threadIdx.x, buffer[0].start, buffer[0].index, buffer[0].value);
        colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[0].index].get_1D_kernel(dir, src);
        printf("threadIdx.x %d start %f end %f mu %f\n", threadIdx.x, current_kernel.start(true), current_kernel.end(), current_kernel.mu);


    }*/
    /*if(n_kernels == 2 && threadIdx.x == 6 && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
        printf("block_id_x %d block_id_y %d\n", blockIdx.x, blockIdx.y);
        result = point3d(1, 0, 0);
        T = 0;
    }
    if(n_kernels == 2 && threadIdx.x == 7 && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
        printf("block_id_x %d block_id_y %d\n", blockIdx.x, blockIdx.y);
        result = point3d(1, 0, 0);
        T = 0;
    }*/
    bool done = T<=stop_T || length_buffer == 0;
    for (int i = 0; i < n_kernels*2 && !done; i++)
    {
        index_start_end b_index = buffer[0];
        array_indexes_type b_index_index = 0;
        //#pragma unroll
        for (int k = 1; k < length_buffer; k++)
        {
            if(buffer[k]< b_index){
                b_index = buffer[k];
                b_index_index = k;
            }
        }
        if(last_tacken < n_kernels*2){
            last_tacken++;
            colored_gaussian_kernel_1D current_kernel = kernels[sorted_indexes[last_tacken-1].index].get_1D_kernel(dir, src);
            float_double st, en;
            st = current_kernel.start();
            en = current_kernel.end();
            //en+=0.0001;
            while((st == en|| current_kernel.get_integral() <= min_integral_v) && last_tacken < n_kernels*2){
                last_tacken++;
                current_kernel = kernels[sorted_indexes[last_tacken-1].index].get_1D_kernel(dir, src);
                st = current_kernel.start();
                en = current_kernel.end();
                //en+=0.0001;
            }
            if(last_tacken <= n_kernels*2 && st < en && current_kernel.get_integral() > min_integral_v){
                if(sorted_indexes[last_tacken-1].start){
                    buffer[b_index_index] = index_start_end(sorted_indexes[last_tacken-1].index, true, st);
                }
                else{
                    buffer[b_index_index] = index_start_end(sorted_indexes[last_tacken-1].index, false, en);
                }
            }
            else{
                buffer[b_index_index] = buffer[length_buffer - 1];
                length_buffer--;
            }
        }
        else{
            buffer[b_index_index] = buffer[length_buffer - 1];
            length_buffer--;
        }
        float_double b = b_index.value;
        /*if((threadIdx.x >= 8 && threadIdx.x <= 9) && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
            printf("threadIdx.x %d , a %f b %f, b_index %d %d %f\n", threadIdx.x, a, b, b_index.index, b_index.start, b_index.value);
        }*/
        if(b>a&& !done){
            //firstly count if we have only one kernel in the window if it is the case we only need one step
            if(nb_curent_kernels == 1)
            {
                //printf("yes, i %d\n", i);
                /*if(threadIdx.x ==0 && threadIdx.y == 0){
                    printf("block_x %d block_y %d curent_kernel log_weight %f\n", blockIdx.x, blockIdx.y, current_kernels[0].log_weight);
                }*/
                float_double eval = current_kernels[0].get_integral(a, b);
                float_double alpha = 1- exp(-eval);
                result = result + current_kernels[0].color * alpha * T;
                T = T * (1- alpha);
            }
            else if(nb_curent_kernels>1){
                for(int j = 0; j < k_steps && T >stop_T; j++){
                    float_double x = a + (b - a) * j / k_steps;
                    float_double x_ = x+((b - a) / k_steps);
                    if(j == k_steps - 1){
                        x_ = b;
                    }
                    float_double sum_eval = 0;
                    point3d color_add = point3d(0, 0, 0);
                    for (int l = 0; l < nb_curent_kernels; l++)
                    {
                        colored_gaussian_kernel_1D current_kernel = current_kernels[l];
                        float_double eval;            
                        eval= current_kernel.get_integral(x, x_);
                        sum_eval += eval;
                        color_add = color_add + current_kernel.color * eval;
                        }
                    /*if((threadIdx.x >= 8 && threadIdx.x <= 9) && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
                        printf("threadIdx.x %d , sum_eval %f\n", threadIdx.x, sum_eval);
                    }*/
                    if(sum_eval>min_sum_eval){
                        float_double alpha = 1- exp(-sum_eval);
                        result = result + color_add * (T/sum_eval)*alpha;
                        T = T * (1- alpha);
                    }
                    
                }
            }
        }
        a = max(0.,max(a,b));
        if(b_index.start){
            if(nb_curent_kernels < max_overlap){
                /*if(threadIdx.x ==0 && threadIdx.y == 0){
                    //printf("pushed %d in %d\n", b_index.index, nb_curent_kernels);
                }*/
                current_kernels[nb_curent_kernels] = kernels[b_index.index].get_1D_kernel(dir, src);
                /*if(threadIdx.x ==0 && threadIdx.y == 0){
                    //printf("block_x %d block_y %d curent_kernel log_weight %f\n", blockIdx.x, blockIdx.y, current_kernels[nb_curent_kernels].log_weight);
                }*/
                current_kernel_indexes[nb_curent_kernels] = b_index.index;
                nb_curent_kernels++;
            }
            else{
                //printf("error nb_curent_kernels %d is bigger than max_overlap %d\n", nb_curent_kernels, max_overlap);
                /*//find the one which will finish the earlier and replace it
                float_double min_end = current_kernels[0].end();
                array_indexes_type min_end_index = 0;
                for (int l = 1; l < nb_curent_kernels; l++)
                {
                    if(current_kernels[l].end() < min_end){
                        min_end = current_kernels[l].end();
                        min_end_index = l;
                    }
                }
                colored_gaussian_kernel_1D b_kernel = kernels[b_index.index].get_1D_kernel(dir, src);
                if(b_kernel.end() > min_end){
                    current_kernels[min_end_index] = b_kernel;
                    current_kernel_indexes[min_end_index] = b_index.index;
                }*/
                //find the one which have the smalest integral left and replace it
                float_double min_integral = current_kernels[0].get_integral(a, current_kernels[0].end());
                array_indexes_type min_integral_index = 0;
                for (int l = 1; l < nb_curent_kernels; l++)
                {
                    float_double integral = current_kernels[l].get_integral(a, current_kernels[l].end());
                    if(integral < min_integral){
                        min_integral = integral;
                        min_integral_index = l;
                    }
                }
                colored_gaussian_kernel_1D b_kernel = kernels[b_index.index].get_1D_kernel(dir, src);
                float_double b_integral = b_kernel.get_integral(a, b_kernel.end());
                if(b_integral > min_integral){
                    current_kernels[min_integral_index] = b_kernel;
                    current_kernel_indexes[min_integral_index] = b_index.index;
                }

            }
        }
        else{
            for (int l = 0; l < nb_curent_kernels; l++)
            {
                if(current_kernel_indexes[l] == b_index.index){
                    for (int m = l; m < nb_curent_kernels - 1; m++)
                    {
                        current_kernels[m] = current_kernels[m+1];
                        current_kernel_indexes[m] = current_kernel_indexes[m+1];
                    }
                    nb_curent_kernels--;
                    /*if(threadIdx.x ==0 && threadIdx.y == 0){
                        printf("popped %d in %d\n", b_index.index, l);
                    }*/
                    break;
                }
            }
        }
        done = T<=stop_T || length_buffer == 0;
    }
    /*if(((threadIdx.x <= 1 && threadIdx.y <= 0)|| false) && ((blockIdx.x == 30 && blockIdx.y == 5)|| false)){
        printf("threadIdx.x %d , result %f %f %f T %f\n", threadIdx.x, result.x, result.y, result.z, T);
    }*/
    result =result+ background_color * T;
    /*if((threadIdx.x >= 8 && threadIdx.x <= 9) && threadIdx.y == 0 && blockIdx.y == 34 && blockIdx.x == 23){
        printf("threadIdx.x %d , result %f %f %f T %f\n", threadIdx.x, result.x, result.y, result.z, T);
    }*/
    return result;
}