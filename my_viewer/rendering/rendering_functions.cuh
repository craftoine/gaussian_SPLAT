#include "ray_gestion.cuh"
#include "pre_filtering/gaussian_filtering.cuh"
__host__ __device__ inline void get_full_block_from_scren_cord_(float_double pixel_x_min, float_double pixel_y_min, float_double pixel_x_max, float_double pixel_y_max, int& block_x_min, int& block_x_max, int& block_y_min, int& block_y_max, int width, int height){
    int pixel_x_min_int = (int) pixel_x_min;
    int pixel_y_min_int = (int) pixel_y_min;
    int pixel_x_max_int = (int) pixel_x_max;
    int pixel_y_max_int = (int) pixel_y_max;
    if(pixel_y_max - pixel_y_min >= 1 && pixel_x_max - pixel_x_min >= 1){
        pixel_x_min_int = max(0, pixel_x_min_int);
        pixel_y_min_int = max(0, pixel_y_min_int);
        pixel_x_max_int = min(width-1, pixel_x_max_int);
        pixel_y_max_int = min(height-1, pixel_y_max_int);
        block_x_min = pixel_x_min_int/rendering_block_size;
        block_x_max = pixel_x_max_int/rendering_block_size;
        block_y_min = pixel_y_min_int/rendering_block_size;
        block_y_max = pixel_y_max_int/rendering_block_size;
    }
    else{
        //printf("too small gaussian kernel\n");
        block_x_min = 1;
        block_x_max = 0;
        block_y_min = 1;
        block_y_max = 0;
    }
}
template <class T>
__host__ __device__ inline void sort_index_start_end(T* array, array_indexes_type start, array_indexes_type end){
    //do an heap sort
    T* arr = array+start;
    array_indexes_type n = end-start+1;

    //build max heap
    for(array_indexes_type i = 1;i<n;i++){
        if(arr[i]>arr[(i-1)/2]){
            array_indexes_type j = i;
            while(j >0 && arr[j]>arr[(j-1)/2]){
                T temp = arr[j];
                arr[j] = arr[(j-1)/2];
                arr[(j-1)/2] = temp;
                j = (j-1)/2;
            }
        }
    }

    //sort

        
    for(array_indexes_type i = n-1;i>0;i--){
        T temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        array_indexes_type j = 0;
        array_indexes_type k;
        do
        {
            k = 2*j+1;

            if(k<i-1 && arr[k]<arr[k+1]){
                k++;
            }
            if(k<i && arr[j]<arr[k]){
                T temp = arr[k];
                arr[k] = arr[j];
                arr[j] = temp;
                j = k;
            }
            j = k;
        }while(k < i);
    }
}
#include "pre_filtering/block_gestion.cuh"
#include "rendering/rendering_pixel.cuh"
#include "rendering/rendering_screen.cuh"
#include "rendering/rendering_screen_prefiltered.cuh"