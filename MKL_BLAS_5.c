/*
 module load intel/clusterxe-2016
 icc -o MKL_BLAS MKL_BLAS_5.c -mkl
*/

//#include <cstdio>
//#include "mkl_blas.h"

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#define N 5

void main()
{
  int n, inca = 1, incb = 1, i;
  MKL_Complex16 a[N], b[N], c;
  
  void zdotc();
  
  n = N;
  for( i = 0; i < n; i++ ){
    a[i].real = (double)i; a[i].imag = (double)i * 2.0;
    b[i].real = (double)(n - i); b[i].imag = (double)i * 2.0;
  }
  
  zdotc( &c, &n, a, &inca, b, &incb );
  printf( "The complex dot product is: ( %6.2f, %6.2f )\n", c.real, c.imag );
}