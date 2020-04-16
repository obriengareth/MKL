// icc -o MKL_BLAS MKL_BLAS_2.c -mkl


#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main()
{
      MKL_INT  n, incx, incy, i;
      float   *x, *y;
      MKL_INT  len_x, len_y;
      n = 3;
      incx = 3;
      incy = 1;
      len_x = 10;
      len_y = 10;
      x    = (float *)calloc( len_x, sizeof( float ) );
      y    = (float *)calloc( len_y, sizeof( float ) );
      if( x == NULL || y == NULL ) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }
      for (i = 0; i < 10; i++) {
          x[i] = i + 1;
      }
      cblas_scopy(n, x, incx, y, incy);
/*       Print output data                                     */
      printf("\n\n     OUTPUT DATA");
      PrintVectorS(FULLPRINT, n, y, incy, "Y");
     
      free(x);
      free(y);
      return 0;
}
