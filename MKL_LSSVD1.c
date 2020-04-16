///////////////////////////////////////////////////////////////////////////////////////////////////
//module load intel/clusterxe-2016
// icc -o MKL_BLAS MKL_LSSVD1.c -mkl
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#define Max(x,y) ( (x)>=(y)  ?  (x) : (y) )
#define Min(x,y) ( (x)<=(y)  ?  (x) : (y) )

extern void print_matrix( char* desc, int m, int n, double* a, int lda );
extern void print_vector_norm( char* desc, int m, int n, double* a, int lda );
///////////////////////////////////////////////////////////////////////////////////////////////////

/* Parameters */
#define M 12
#define N 10
#define NRHS 1
#define LDA M
#define LDB M
#define QR_SVD 2	// 1 for QR otherwise SVD

///////////////////////////////////////////////////////////////////////////////////////////////////

/* Main program */
int main() 
{
        /* Locals */
        int k,m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork;
        double wkopt;
        double* work;
	double a[LDA*N], a0[LDA*N],b[LDB*NRHS] ;

	/////////////////////////////////////////////////////////////////////////////////////
	// test parameters to run code
	for(k=0; k<LDA*N; k++)
	{
	  a[k] = ( 1 + k ) * drand48() + 5*drand48();
	  a0[k] = a[k]; 
	}
	print_matrix( "Matrix A=[", m, n, a, lda );

	for(k=0; k<LDB*NRHS; k++)
	  b[k]=5 + drand48();
	
	/////////////////////////////////////////////////////////////////////////////////////
        double s[Max(1, Min(N,M))],rcond=0;  // need to check size of s = max(1, min(N,M))
	int rank ;
	lwork = -1;

	if(QR_SVD==1)
	{
	      // QR Solution
	      dgels( "No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork, &info );
	      lwork = (int)wkopt;
	      work = (double*)malloc( lwork*sizeof(double) );
	      /* Solve the equations A*X = B */
	      dgels( "No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info );
	}
	else
	{
	      // SVD Solution
	      dgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, &wkopt, &lwork, &info  );
	      lwork = (int)wkopt;
	      work = (double*)malloc( lwork*sizeof(double) );
	      /* Solve the equations A*X = B */
	      dgelss(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, work, &lwork, &info  );
	}
	
	/* Check for the full rank */
        if( info > 0 ) {
                printf( "The diagonal element %i of the triangular factor ", info );
                printf( "of A is zero, so that A does not have full rank;\n" );
                printf( "the least squares solution could not be computed.\n" );
                exit( 1 );
        }
        
        /* Print least squares solution */
        print_matrix( " Least squares solution", n, nrhs, b, ldb );

        /* Free workspace */
        free( (void*)work );
        exit( 0 );

} /* End of DGELSS Example */


///////////////////////////////////////////////////////////////////////////////////////////////////
/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.4f", a[i+j*lda] );
                printf( "\n" );
        }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/* Auxiliary routine: printing norms of matrix columns */
void print_vector_norm( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        double norm;
        printf( "\n %s\n", desc );
        for( j = 0; j < n; j++ ) {
                norm = 0.0;
                for( i = 0; i < m; i++ ) norm += a[i+j*lda] * a[i+j*lda];
                printf( " %6.2f", norm );
        }
        printf( "\n" );
}
///////////////////////////////////////////////////////////////////////////////////////////////////
