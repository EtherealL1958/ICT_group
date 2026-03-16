
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

int ComputeSYMGS_TDG ( const SparseMatrix & A, const Vector & r, Vector & x ) {

	assert( x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = currentNumberOfNonzeros-1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	return 0;
}

int ComputeSYMGS_BLOCK( const SparseMatrix & A, const Vector & r, Vector & x ) {

	assert(x.localLength >= A.localNumberOfColumns);
	
#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;

	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) {
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) {
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize*A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = first; i < last; i+= (A.chunkSize/2)) {
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];
					double sum2 = rv[i+2];
					double sum3 = rv[i+3];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
						sum2 -= A.matrixValues[i+2][j] * xv[A.mtxIndL[i+2][j]];
						sum3 -= A.matrixValues[i+3][j] * xv[A.mtxIndL[i+3][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
					sum2 += matrixDiagonal[i+2][0] * xv[i+2];
					xv[i+2] = sum2 / matrixDiagonal[i+2][0];
					sum3 += matrixDiagonal[i+3][0] * xv[i+3];
					xv[i+3] = sum3 / matrixDiagonal[i+3][0];
				} else if ( A.chunkSize == 2 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i+0];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) {
			local_int_t firstRow = ((block+1) * A.blockSize) - 1; // this is the last row of the last block
			local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
			local_int_t lastChunk = firstChunk - A.blockSize;

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = last-1; i >= first; i -= (A.chunkSize/2)) {
				local_int_t i = last-1;
				if ( A.chunkSize == 4 ) {
					double sum3 = rv[i-3];
					double sum2 = rv[i-2];
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum3 -= A.matrixValues[i-3][j] * xv[A.mtxIndL[i-3][j]];
						sum2 -= A.matrixValues[i-2][j] * xv[A.mtxIndL[i-2][j]];
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}
					sum3 += matrixDiagonal[i-3][0] * xv[i-3];
					xv[i-3] = sum3 / matrixDiagonal[i-3][0];

					sum2 += matrixDiagonal[i-2][0] * xv[i-2];
					xv[i-2] = sum2 / matrixDiagonal[i-2][0];

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else if ( A.chunkSize == 2 ) {
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				}
			}
		}
	}

	return 0;
}
//#endif

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

	// This function is just a stub right now which decides which implementation of the SYMGS will be executed (TDG or block coloring)
	if ( A.TDG )
		return ComputeSYMGS_TDG(A, r, x);
	return ComputeSYMGS_BLOCK(A, r, x);

}