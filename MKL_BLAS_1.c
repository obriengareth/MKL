//module load intel/clusterxe-2016

// icc -o MKL_BLAS MKL_BLAS_1.c -mkl


#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main()
{
      MKL_INT  n, incx, incy, i;
      float   *x, *y;
      float    res;
      MKL_INT  len_x, len_y;
      n = 5;
      incx = 2;
      incy = 1;
      len_x = 1+(n-1)*abs(incx);
      len_y = 1+(n-1)*abs(incy);
      x    = (float *)calloc( len_x, sizeof( float ) );
      y    = (float *)calloc( len_y, sizeof( float ) );
      if( x == NULL || y == NULL ) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }
      for (i = 0; i < n; i++) {
          x[i*abs(incx)] = 2.0;
          y[i*abs(incy)] = 1.0;
      }
      res = cblas_sdot(n, x, incx, y, incy);

      printf("\n       SDOT = %7.3f\n", res);
      free(x);
      free(y);
      return 0;
      
}

