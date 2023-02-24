/*-----------------------------------------------------------------------------
* Author  : Adam Young
*
* Class   : CS-525, Assignment 1
*
* Date    : October 8, 2011
*
* File    : cs_525_a1.cu
* 
* Purpose : 
*
* Notes   : None
*-----------------------------------------------------------------------------*/

#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "heat_distrobution_matrix.h"
 
int main( void ) {
  int N = 0, num_blocks_x, num_blocks_y, num_threads_x, num_threads_y ;
  
  printf("Enter value of N for heat distrobution matrix -> ");
  scanf("%d",&N);
  printf("Enter num blocks (x) for value N -> ");
  scanf("%d",&num_blocks_x);
  printf("Enter num blocks (y) for value N -> ");
  scanf("%d",&num_blocks_y);
  printf("Enter num threads (x) for value N -> ");
  scanf("%d",&num_threads_x);
  printf("Enter num threads (y) for value N -> ");
  scanf("%d",&num_threads_y);
  printf("Enter error value e -> ");
  
  int total_x = num_blocks_x * num_threads_x;
  int total_y = num_blocks_y * num_threads_y;
  // Validate arguments
  int count;
  int device_set = 0;

  cudaDeviceProp prop;
  cudaGetDeviceCount(&count);
  for(int i =0; i< count; i++){
    cudaGetDeviceProperties(&prop,i);
    if (num_threads_x <= prop.maxThreadsDim[0] && 
        num_threads_y <= prop.maxThreadsDim[1] &&
        num_blocks_x  <= prop.maxGridSize[0] &&
        num_blocks_y  <= prop.maxGridSize[1] && 
        num_threads_x * num_threads_y <= prop.maxThreadsPerBlock)
      printf("Max threads dim size x %d\n",prop.maxThreadsDim[0]);
      printf("Max threads dim size y %d\n",prop.maxThreadsDim[1]);
      printf("Max grid size x %d\n",prop.maxGridSize[0]);
      printf("Max grid size y %d\n",prop.maxGridSize[1]);
      printf("Max threads per block %d\n",prop.maxThreadsPerBlock);
      cudaSetDevice(i);
      device_set = 1; 
  }
  if (!device_set){
    printf("Error - No Cuda card(s) support the specified input");
  }
  else if (total_x < N)
    printf("Error - Invalid number of threads [%d] for specified N [%d]",
           total_x, N);
  else if (total_y < N)
    printf("Error - Invalid number of threads [%d] for specified N [%d]",
           total_y, N);
  else if (N <= 0 || N > 4000){
    
    printf("Error - Invalid dimensions [%d x %d] for heat distrobution matrix",
           N);
  }
  else{
    //N=N+1;
    // Host vars
    float *heat_matrix1 = (float*) calloc(N * N, sizeof(float));
    float *heat_matrix2 = (float*) calloc(N * N, sizeof(float));

    // Device vars
    float *dev_heat_matrix1;
    float *dev_heat_matrix2;

    float start, stop;
    float time;
    
    dim3 num_blocks(num_blocks_x,num_blocks_y);
    dim3 num_threads(num_threads_x,num_threads_y);

    // Allocate appropriate space for the matrices 
    cudaMalloc((void**) &dev_heat_matrix1,   N * N*sizeof(float));
    cudaMalloc((void**) &dev_heat_matrix2,   N * N*sizeof(float));

    populate_room_with_tempturature(heat_matrix1, N);

    cudaMemcpy(dev_heat_matrix2, heat_matrix1, N * N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_heat_matrix1, heat_matrix1, N * N*sizeof(float),cudaMemcpyHostToDevice);

    start = clock();
    printf("Running cpu version\n");
    cpu_compute_heat_distrobution(heat_matrix1, N);
    stop = clock();
    time = (stop-start)/(float)CLOCKS_PER_SEC;
   
    print_matrix(heat_matrix1,N,time);
   
    printf("Running gpu version\n");
    
    // Start the timer.
    cudaEvent_t dev_start, dev_stop;

    cudaEventCreate(&dev_start);
    cudaEventCreate(&dev_stop);

    cudaEventRecord(dev_start,0);
    //start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++){

      // Compute all the values for the matrix.

      global_compute_heat_distrobution<<<num_blocks,num_threads>>>(dev_heat_matrix2, dev_heat_matrix1, N);

      cudaThreadSynchronize();

      // Copy all the heat values.
      copy_heat_values<<<num_blocks,num_threads>>>(dev_heat_matrix1, dev_heat_matrix2, N);
      cudaThreadSynchronize();
    }

    cudaMemcpy(heat_matrix2, dev_heat_matrix1,N * N*sizeof(float),cudaMemcpyDeviceToHost);
   // stop = clock();

    // Stop the timer, computed the elapsed time, and destroy the timer vars.
    cudaEventRecord(dev_stop,0);
    cudaEventSynchronize(dev_stop);

    cudaEventElapsedTime(&time,dev_start,dev_stop);
    cudaEventDestroy(dev_start);
    cudaEventDestroy(dev_stop);
    //time = (dev_stop-start)/(float)CLOCKS_PER_SEC;
    printf("%f\n",time);
    print_matrix(heat_matrix2,N,time); 

    // Release host memory
    free(heat_matrix1);
    free(heat_matrix2);

    // Release device memory
    cudaFree(dev_heat_matrix1);
    cudaFree(dev_heat_matrix2);

  }
  printf("Select <enter> to exit");
	//scanf("%c",&user_input);
  //scanf("%c",&user_input);
  printf("\n");
  
	return 0;

}