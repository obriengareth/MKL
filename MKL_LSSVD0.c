// Gareth O'Brien from intel examples 03/04/2018
//module load intel/clusterxe-2016
// icc -o MKL_BLAS MKL_SVD.c -mkl

#include <stdlib.h>
#include <stdio.h>
#include "mkl.h"

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );

/* Parameters */
#define M 12
#define N 10
#define LDA M
#define LDU M
#define LDVT N

/* Main program */
int main() 
{
        /* Locals */
        int k,m = M, n = N, lda = LDA, ldu = LDU, ldvt = LDVT, info, lwork,matrix_layout;
        double wkopt;
	double* superb;
        double* work;
        /* Local arrays */
        double a[LDA*N],s[N], u[LDU*M], vt[LDVT*N],a0[LDA*N];

	/////////////////////////////////////////////////////////////////////////////////////
	for(k=0; k<LDA*N; k++)
	{
	  a[k] =  1 + k + 0.01*drand48();
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
	/*double  at[LDA*N] ;
	// S*VT
	for(r=0; r<N; r++)
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
      // x = V * (S^-1) * (U')
      
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

      /////////////////////////////////////////////////////////////////////////////
      
      
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
	
} // 

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







