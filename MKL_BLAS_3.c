//module load intel/clusterxe-2016

// icc -o MKL_BLAS MKL_BLAS_3.c -mkl

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main()
{
      MKL_INT         m, n, lda, incx, incy, i, j;
      MKL_INT         rmaxa, cmaxa;
      float           alpha;
      float          *a, *x, *y;
      CBLAS_LAYOUT    layout;
      MKL_INT         len_x, len_y;
      m = 2;
      n = 3;
      lda = 5;
      incx = 2;
      incy = 1;
      alpha = 0.5;
                        layout = CblasRowMajor;
      len_x = 10;
      len_y = 10;
      rmaxa = m + 1;
      cmaxa = n;
      a = (float *)calloc( rmaxa*cmaxa, sizeof(float) );
      x = (float *)calloc( len_x, sizeof(float) );
      y = (float *)calloc( len_y, sizeof(float) );
      if( a == NULL || x == NULL || y == NULL ) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }
      if( layout == CblasRowMajor )
        lda=cmaxa;
      else
         lda=rmaxa;
      for (i = 0; i < 10; i++) {
          x[i] = 1.0;
          y[i] = 1.0;
      }
     
      for (i = 0; i < m; i++) {
          for (j = 0; j < n; j++) {
              a[i + j*lda] = j + 1;
          }
      }
      cblas_sger(layout, m, n, alpha, x, incx, y, incy, a, lda);
      
      PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &m, &n, a, &lda, "A");
      
      free(a);
      free(x);
      free(y);
      return 0;
      
}
