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
*                                (100 *C)     (20 *C)   
*                      __________=======______/___    
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
*           following calculation (i.e. Jacobi Algorithm):
*
*             H[i][j] = H[i-1][j] + H[i+1][j] + H[i][j-1] + H[i][j+1]
*                       ---------------------------------------------
*                                            4
*
* Notes   : None
*-----------------------------------------------------------------------------*/

#ifndef HEAT_DISTROBUTION_MATRIX_H
#define HEAT_DISTROBUTION_MATRIX_H

#include <stdio.h>
#include <cuda.h>

// Constant temperatures (celsius)
#define FIREPLACE_TEMP 100
#define WALL_TEMP 20
#define NUM_ITERATIONS 50000
#define BLOCK_SIZE 16;

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
void populate_room_with_tempturature(float *matrix, int N);

/*-----------------------------------------------------------------------------
* Function: cpu_jacobi_iteration
* 
* Purpose : Takes in a pointer to a matrix as well as square dimension 'N' as 
*           the parameters and computes values for each index based on on the
*           jacobi iteration algorithm described above, to compute the heat
*           distrobution matrix for the room.
*
* Returns : None
*
* Notes   : Assumes the matrix passed in has space allocated and accessable
*           for a matrix of dimensions N * N. Points in the room are based on
*           the fireplace and wall temperatures defined above.
*-----------------------------------------------------------------------------*/
void cpu_compute_heat_distrobution(float *matrix, int N);

/*-----------------------------------------------------------------------------
* Function: gpu_bae_rotation2
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
void global_compute_heat_distrobution(float *dst_matrix, float *src_matrix, int N);

/*-----------------------------------------------------------------------------
* Function: gpu_bae_rotation2
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
void shared_compute_heat_distrobution(float *dst_matrix, float *src_matrix, int N);


/*-----------------------------------------------------------------------------
* Function: gpu_bae_rotation2
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
void copy_heat_values(float *dst_matrix, float *src_matrix, int N);

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
void print_matrix(float *matrix, int N, float time);

#endif HEAT_DISTROBUTION_MATRIX_H