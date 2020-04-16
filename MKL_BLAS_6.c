//module load intel/clusterxe-2016
// icc -o MKL_BLAS MKL_BLAS_6.c -mkl

/*
   DGESVD Example.
   ==============

   Program computes the singular value decomposition of a general
   rectangular matrix A:

     8.79   9.93   9.83   5.45   3.16
     6.11   6.91   5.04  -0.27   7.98
    -9.15  -7.93   4.86   4.85   3.01
     9.57   1.64   8.83   0.74   5.80
    -3.49   4.02   9.80  10.00   4.27
     9.84   0.15  -8.99  -6.02  -5.31

   Description.
   ============

   The routine computes the singular value decomposition (SVD) of a real
   m-by-n matrix A, optionally computing the left and/or right singular
   vectors. The SVD is written as

   A = U*SIGMA*VT

   where SIGMA is an m-by-n matrix which is zero except for its min(m,n)
   diagonal elements, U is an m-by-m orthogonal matrix and VT (V transposed)
   is an n-by-n orthogonal matrix. The diagonal elements of SIGMA
   are the singular values of A; they are real and non-negative, and are
   returned in descending order. The first min(m, n) columns of U and V are
   the left and right singular vectors of A.

   Note that the routine returns VT, not V.

   Example Program Results.
   ========================

 DGESVD Example Program Results

 Singular values
  27.47  22.64   8.56   5.99   2.01

 Left singular vectors (stored columnwise)
  -0.59   0.26   0.36   0.31   0.23
  -0.40   0.24  -0.22  -0.75  -0.36
  -0.03  -0.60  -0.45   0.23  -0.31
  -0.43   0.24  -0.69   0.33   0.16
  -0.47  -0.35   0.39   0.16  -0.52
   0.29   0.58  -0.02   0.38  -0.65

 Right singular vectors (stored rowwise)
  -0.25  -0.40  -0.69  -0.37  -0.41
   0.81   0.36  -0.25  -0.37  -0.10
  -0.26   0.70  -0.22   0.39  -0.49
   0.40  -0.45   0.25   0.43  -0.62
  -0.22   0.14   0.59  -0.63  -0.44
*/

#include <stdlib.h>
#include <stdio.h>
#include "mkl.h"

/* DGESVD prototype */
//extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
//                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
//                double* work, int* lwork, int* info );
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

/* Parameters */
#define M 6
#define N 5
#define LDA M
#define LDU M
#define LDVT N

/* Main program */
int main() {
        /* Locals */
        int k,m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork,matrix_layout;
        double wkopt;
	double* superb;
        double* work;
        /* Local arrays */
        double s[N], u[LDU*M], vt[LDVT*N],a0[LDA*N];
	
	
      /*  double a[LDA*N] = {
            8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
            9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
            9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
            5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
            3.16,  7.98,  3.01,  5.80,  4.27, -5.31
        };*/
	
	for(k=0; k<LDA*N; k++)
	{
	  a[k] =  1 + k + drand48();
	  a0[k] = a[k]; 
	}
	
	print_matrix( "Matrix A", m, n, a, ldu );

        /* Executable statements */
        printf( " \t\tSVD Example \n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        dgesvd("All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Compute SVD */
        dgesvd("All", "All", &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info );
	
        /* Check for convergence */
    //    if( info > 0 ) {
   //             printf( "The algorithm computing SVD failed to converge.\n" );
   //             exit( 1 );
   //      }

     /* Print singular values */
        print_matrix( "S", 1, n, s, 1 );
        /* Print left singular vectors */
        print_matrix( "U=[ ", m, n, u, ldu );
        /* Print right singular vectors */
        print_matrix( "VT=[ ", n, n, vt, ldvt );
	
	// check results
	int r,c;
	double  at[LDA*N] ;
	// S*VT
	/*for(r=0; r<N; r++)
	{ 
	  for(c=0; c<N; c++)
	  {
	    vt[r+c*ldvt] = vt[r+c*ldvt]*s[r];
	  }
	}
        print_matrix( "VT 1", n, n, vt, ldvt );*/
        // U*S*VT
	/*for(r=0; r<M; r++)
	{ 
	  for(c=0; c<N; c++)
	  {
	    at[r+c*lda] = 0;
	    for(k=0; k<N; k++)
	    at[r+c*lda] = at[r+c*lda] + u[r+k*ldu]*vt[k+c*ldvt];
	  }
	}

     	print_matrix( "Matrix At", m, n, at, ldu );*/
      /////////////////////////////////////////////////////////////////////////////
      // Mess with LS SVD solution
      // Ax= b; A=U.S.VT
      // x = V * S-1 * UT
      double p[N],b0[M],x[N], x0[N];
      for(c=0; c<N; c++)
      {
	x0[c]=1;
      }
      // A*X
      for(r=0; r<M; r++)
      {   b0[r] = 0;
	  for(c=0; c<N; c++)
	  {
	    b0[r] = b0[r] + a0[r+c*lda]*x0[c];
	  }
      }
      print_matrix( "B=[", 1, m, b0, 1 );
      
      // Calc p=S.UT.b
      for(c=0; c<N; c++)
      {
	  p[c] = 0;
	  for(r=0; r<M; r++)
	  {     
	    p[c] = p[c] + u[c*M+r]*b0[r];
	  }
	  p[c] = p[c]/s[c];
      }
      print_matrix( "P=[", 1, n, p, 1 );
    
      // Calc x = V.p
        for(c=0; c<N; c++)
	{
	  x[c] = 0;
	  for(r=0; r<N; r++)
	  {
	    x[c] =  x[c] + vt[r+c*ldvt]*p[r];
	  }
	}
       print_matrix( "SVD LS X", 1, n, x, 1 );

      /////////////////////////////////////////////////////////////////////////////
      
        /* Free workspace */
        free( (void*)work );
        exit( 0 );
	
} /* End of DGESVD Example */

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) 
{
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) 
	{
                for( j = 0; j < n; j++ ) printf( " %6.2f\t", a[i+j*lda] );
                printf( "\n" );
        }
}







