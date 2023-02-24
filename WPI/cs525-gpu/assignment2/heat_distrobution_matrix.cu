/*-----------------------------------------------------------------------------
* Author  : Adam Young
*
* Class   : CS-525, Assignment 2
*
* Date    : October 22, 2011
*
* File    : heat_distrobution_matrix.cu
* 
* Purpose : Defines operations to generate and print heat distrobution 
*           matrices on the CPU and GPU based on the following room:
*
*                      __________=======__________    
*                     |             |             | |
*                     |             |             | |
*                     |           [N/4]           | |
*                     |                           | |
*                     |                           | |
*                     |                           | |
*                     |                           | N
*                     |                           | |
*                     |                           | |
*                     |                           | |
*                     |                           | |
*                     |                           | |
*                     |___________________________| |
*                     <------------ N ------------>
*
*           Heat distrobution for each point in the room is computed by the 
*           following calculation:
*
*             H[i][j] = H[i-1][j] + H[i+1][j] + H[i][j-1] + H[i][j+1]
*                       ---------------------------------------------
*                                            4
*
* Notes   : None
*-----------------------------------------------------------------------------*/

#include <math.h>
#include "heat_distrobution_matrix.h"

/*-----------------------------------------------------------------------------
* Function: populate_room_with_tempturature
* 
* Purpose : Poputlates room with known temperatures for the room wall and 
*           fireplace.
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*           for a matrix of dimensions N * N. Points in the room are based on
*           the fireplace and wall temperatures defined above.
*-----------------------------------------------------------------------------*/
void populate_room_with_tempturature(float *matrix, int N){
  int fireplace_indeces = (N*4)/10; 

  int fireplace_start = N/2 - fireplace_indeces/2;
  int fireplace_end = fireplace_start + fireplace_indeces;

  // Top wall with fireplace
  for (int column=0; column<N; column++)
    if(column >= fireplace_start && column < fireplace_end)
      matrix[column] = FIREPLACE_TEMP;
    else
      matrix[column] = WALL_TEMP;

  // Bottom wall
  for (int column = 0; column<N; column++)
    matrix[ (N*(N-1))+column] = WALL_TEMP;

  // Left/Right wall
  for (int row = 0; row < N; row++){
    matrix[N * row] = WALL_TEMP;
    matrix[N * row + (N - 1)] = WALL_TEMP;
  }
}

/*-----------------------------------------------------------------------------
* Function: cpu_jacobi_iteration
* 
* Purpose : Takes in a pointer to a matrix as the parameter and computes 
*           values for each index on the CPU based on the rotation matrix
*           described in the file description.
*           
*            x,y,z coordinates are computed from the arbitrary axis parameter.
*            Angle value (degrees) is used to compute c and s values 
*            described above.
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*           for a matrix of dimensions - MATRIX_ROWS x MATRIX_COLUMNS
*           Assumes the axis passed in has space allocated and accessable for
*           three coordinates in the order (x,y,x).
*-----------------------------------------------------------------------------*/
void cpu_compute_heat_distrobution(float *matrix, int N){
  float *g  = (float*) calloc(N * N, sizeof(float));

  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++){
    for(int i = 1; i < N-1; i++)
      for(int j = 1; j < N-1; j++){
      
        g[i * N + j] = 0.25 * ( matrix[((i-1) * N) +   j  ] + 
                                matrix[((i+1) * N) +   j  ] + 
                                     matrix[(  i   * N) + (j-1)] + 
                                     matrix[(  i   * N) + (j+1)] );
      }
    for(int i = 1; i < N-1; i++)
      for(int j = 1; j < N-1; j++)
        matrix[i * N + j] = g[i * N + j];

  }
  free(g);
}

/*-----------------------------------------------------------------------------
* Function: gpu_compute_heat_distrobution
* 
* Purpose : 
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*            for a matrix of dimensions N * N. Points in the room are based on
*            the fireplace and wall temperatures defined above.
*-----------------------------------------------------------------------------*/
__global__ 
void global_compute_heat_distrobution(float *dst_matrix, float *src_matrix, int N){

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int tid       = x + y *blockDim.x *gridDim.x;
  int tid_up    = x + (y-1) *blockDim.x *gridDim.x;
  int tid_down  = x + (y+1) *blockDim.x *gridDim.x;
  int tid_left  = (x-1) + y *blockDim.x *gridDim.x;
  int tid_right = (x+1) + y *blockDim.x *gridDim.x;

  if (x > 0 && x < (N-1) && y > 0 && y < (N-1)){ 
      
      //dst_matrix[y*N+x] = 0.25 * ( src_matrix[(( y -1) * N) +  x   ] + 
      //                             src_matrix[(( y +1) * N) +  x   ] + 
      //                             src_matrix[(  y     * N) +  x - 1 ] + 
      //                             src_matrix[(  y     * N) + x + 1 ] );
      dst_matrix[tid] = 0.25 * ( src_matrix[tid_up   ] + 
                                 src_matrix[tid_down  ] + 
                                 src_matrix[tid_left ] + 
                                 src_matrix[tid_right] );
  }
}

/*-----------------------------------------------------------------------------
* Function: gpu_compute_heat_distrobution
* 
* Purpose : 
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*            for a matrix of dimensions N * N. Points in the room are based on
*            the fireplace and wall temperatures defined above.
*-----------------------------------------------------------------------------*/
__global__ 
void shared_compute_heat_distrobution(float *dst_matrix, float *src_matrix, int N, const int threadsPerBlock){
  //__shared__ float cache[BLOCK_SIZE][BLOCK_SIZE];

  //read into shared.

  __syncthreads();
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int tid       = x + y *blockDim.x *gridDim.x;
  int tid_up    = x + (y-1) *blockDim.x *gridDim.x;
  int tid_down  = x + (y+1) *blockDim.x *gridDim.x;
  int tid_left  = (x-1) + y *blockDim.x *gridDim.x;
  int tid_right = (x+1) + y *blockDim.x *gridDim.x;

  if (x > 0 && x < (N-1) && y > 0 && y < (N-1)){ 
      
      //cache[tid] = 0.25 * ( cache[tid_up   ] + 
      //                           cache[tid_down  ] + 
      //                           cache[tid_left ] + 
      //                           cache[tid_right] );
      __syncthreads();
  }
  
  //copy shared back to global
}

__global__ 
void copy_heat_values(float *dst_matrix, float *src_matrix, int N){

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int tid       = x + y *blockDim.x *gridDim.x;

  //dst_matrix[y*N+x] = src_matrix[y*N+x];
  dst_matrix[tid] = src_matrix[tid];
}



/*-----------------------------------------------------------------------------
* Function: print_matrix
* 
* Purpose : Takes in a pointer to a matrix as the parameter and print it to
*           standard output.
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*           for a matrix of dimensions - MATRIX_ROWS x MATRIX_COLUMNS
*-----------------------------------------------------------------------------*/
void print_matrix(float *matrix, int N, float time){

	printf("N = %d \n",N);

  printf("Elapsed_Time = %f \n",N);

  if (N < 7){ 
	  for (int row = 0; row < N; row++){
      for (int column = 0; column < N; column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");
  }
  else{
    
    // Top left
    printf("Top left corner\n");
    for (int row = 0; row < 6; row++){
      for (int column = 0; column < 6; column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");

    // Top right
    printf("Top right corner\n");
    for (int row = 0; row < 6; row++){
      for (int column = N-6; column < N; column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");

    // Middle 
    printf("Middle\n");
    for (int row = (N/2 - 3); row < (N/2+3); row++){
      for (int column = (N/2 - 3); column < (N/2 + 3); column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");
    
    // Bottom left corner
    printf("Bottom left corner\n");
    for (int row = N - 6; row < N; row++){
      for (int column = 0; column < 6; column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");

    // Bottom right corner
    printf("Bottom right corner\n");
    for (int row = N-6; row < N; row++){
      for (int column = N-6; column < N; column++){
			  printf("%10f ", matrix[row*N+column]);
     
      }
		  printf("\n");
	  }
    printf("\n");
  }

}