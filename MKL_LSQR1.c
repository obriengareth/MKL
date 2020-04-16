//module load intel/clusterxe-2016
// icc -o MKL_BLAS MKL_LSSVD0.c -mkl

#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>

/* DGELS prototype */
/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );
extern void print_vector_norm( char* desc, int m, int n, double* a, int lda );

/* Parameters */
#define M 6
#define N 4
#define NRHS 2
#define LDA M
#define LDB M

/* Main program */
int main() 
{
        /* Locals */
        int m = M, n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info, lwork;
        double wkopt;
        double* work;
        /* Local arrays */
        double a[LDA*N] = {
            1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
           -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
           -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
            4.53,  3.83, -6.64,  2.06, -2.47,  4.70
        };
        double b[LDB*NRHS] = {
            8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
            9.35, -4.43, -0.70, -0.26, -7.36, -2.52
        };
        /* Executable statements */
        printf( " DGELS Example Program Results\n" );
        /* Query and allocate the optimal workspace */
        lwork = -1;
        dgels( "No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, &wkopt, &lwork, &info );
        lwork = (int)wkopt;
        work = (double*)malloc( lwork*sizeof(double) );
        /* Solve the equations A*X = B */
        dgels( "No transpose", &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info );
        /* Check for the full rank */
        if( info > 0 ) {
                printf( "The diagonal element %i of the triangular factor ", info );
                printf( "of A is zero, so that A does not have full rank;\n" );
                printf( "the least squares solution could not be computed.\n" );
                exit( 1 );
        }
        /* Print least squares solution */
        print_matrix( "Least squares solution", n, nrhs, b, ldb );
        /* Print residual sum of squares for the solution */
        print_vector_norm( "Residual sum of squares for the solution", m-n, nrhs,
                        &b[n], ldb );
        /* Print details of QR factorization */
        print_matrix( "Details of QR factorization", m, n, a, lda );
        /* Free workspace */
        free( (void*)work );
        exit( 0 );
} /* End of DGELS Example */


//lapack_int LAPACKE_dgels ( int matrix_layout, char trans, lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb);
//lapack_int LAPACKE_dgelss( int matrix_layout,             lapack_int m, lapack_int n, lapack_int nrhs, double* a, lapack_int lda, double* b, lapack_int ldb, double* s, double rcond, lapack_int* rank );


/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
}

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