/*
Solve Ax=b
The p? function solves overdetermined or underdetermined real linear systems 
involving an m-by-n matrix sub(A) = A(ia:ia+m-1,ja:ja+n-1), or its transpose, 
using a QTQ or LQ factorization of sub(A). It is assumed that sub(A) has full 
rank.

Compile and Run
module load intel/clusterxe-2016
mpicc -o MKL_Scalapack_LSQR MKL_Scalapack_LSQR.c -mkl -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64
mpirun -np 4 ./MKL_Scalapack_LSQR

Gareth 6/6/18
*/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "mpi.h"
#define DS 11  // DESCRIPTOR_SIZE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define Mg 10 // Number of rows in global Matrix
#define Ng 8 // Number of columns in global Matrix
#define Nr 1 // Number of rhs columns

#define Mp 2 // Number of splits of A in M direction
#define Np 2 // Number of splits of A in N direction

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// functions
extern void print_matrix( char* desc, int m, int n, float* a, int lda );

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gloabl Variables
int mysize, myrank;
int icon;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc ,char *argv[])
{
  int myproc,noproc;
  //int mp,np;	//Number of splits of A in M and N direction
  int Mp_ret,Np_ret,myrow, mycol;
  float *Al,*bl; 
  float *x0,*x,*b0;
  int *ipiv;

    /// //////////////////////////////////////////////////////////////////////////////////////////
    /// Initialize MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mysize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if(myrank==0)
    {
      printf("Test Program for LS QR\n");
      printf("Using %d procs: Proc=%d\n",mysize,myrank);
    }
    
    /// //////////////////////////////////////////////////////////////////////////////////////////   
    /// set up basic linear alegbra comms
    // Returns the number of processes available for use.
    Cblacs_pinfo( &myproc, &noproc ); 
    // Gets values that BLACS use for internal defaults. 
    Cblacs_get( -1, 0, &icon );
    // Assigns available processes into BLACS process grid  MP & NP = 2**x
    Cblacs_gridinit( &icon,"c", Mp, Np ); // default, column major, nrows, ncols
    // Returns information on the current grid.
    Cblacs_gridinfo( icon, &Mp_ret, &Mp_ret, &myrow, &mycol);
    // Check the setup via pritn to screen
    //printf("proc=%d\tMP=%d\tNp=%d\trow=%d\tcol=%d myrank=%d\n",myrank,Mp_ret, Mp_ret, myrow, mycol,myrank);
    //printf("Cblacs %d procs: Proc=%d myrank=%d\n",noproc,myproc,myrank);

    /// ///////////////////////////////////////////////////////////////////////////////////////////
    /// Initializes the array descriptor for distributed matrix
    int info=0,zero=0,one=1;
    int nrhs=Nr;
    int descal[DS], descbl[DS], descxl[DS];
    int mg=Mg,ng=Ng;
    int mla=mg/Mp,nla=ng/Np;
    int mb=1,nb=1;
    descinit_(descal, &mg, &ng  , &mb, &nb , &zero, &zero, &icon, &mla, &info);
    info=0;
    descinit_(descbl, &mg, &nrhs, &mb, &one, &zero, &zero, &icon, &mla, &info);
    info=0;
    descinit_(descxl, &ng, &nrhs, &nb, &one, &zero, &zero, &icon, &nla, &info);

    Al = (float*)malloc(sizeof(float)*mla*nla);
    bl = (float*)malloc(sizeof(float)*mla*nrhs);
    b0 = (float*)malloc(sizeof(float)*mla*nrhs);
    x0 = (float*)malloc(sizeof(float)*nla*nrhs);

    ipiv=(int*)malloc(sizeof(int)*(mla+mb));
    
    /// //////////////////////////////////////////////////////////////////////////////////////////
    /// Build Matrix A
    int r,c,i,j;
    for(r=0; r<mla; r++)
    {
      for(c=0; c<nla; c++)
      { 
	i=r + (mla*myrow); j=(c+mycol*nla);
	Al[r+c*mla] = 1 ;
	if(i==j)
	  Al[r+c*mla] = 3;
	if(i==j-1)
	  Al[r+c*mla] = 2;
	if(i==j+1)
	  Al[r+c*mla] = 2;

	//Al[r+c*mla] = 1 + 0*drand48() + r + (mla*myrow)+(c+mycol*nla)*mg; 
      }
    }
    
    if(myrank==0)
    {
      printf( "Matrix Size Ag (%d x %d) Al (%d x %d)\n", mg, ng, mla, nla );
      printf( "Matrix Size Bg (%d x %d) Al (%d x %d)\n\n", mg, nrhs, mla, nrhs );
      //printf("Proc=%d\n",myrank);
     // print_matrix( "Matrix A=[", mla, nla, Al, mla );
    }
    
   /* for(r=1; r<mysize; r++)
    {
      if(r==myrank)
      {
      printf("Proc=%d\n",myrank);
      print_matrix( "Matrix A=[", mla, nla, Al, mla );
      }
    }*/
   
    /// //////////////////////////////////////////////////////////////////////////////////////////
    /// Build Matrix X nrhs
    for(r=0; r<nla; r++)
    {
      for(c=0; c<nrhs; c++)
      { 
        x0[r+c*nla]= 1 + r + (nla*myrow);
      }
    }
    if(mycol==0)
    {
      //printf("Proc=%d\n",myrank);
   //   if(myrank==0) printf("Matrix X0=[\n");
   //   print_matrix("", nla, nrhs, x0, nla );
    }
    
    /// //////////////////////////////////////////////////////////////////////////////////////////
    /// Cal B = alpha*A.X + betaY; 
    
    float alpha=1, beta=0;
    psgemm_("N" , "N" , &mg , &nrhs , &ng ,&alpha , Al , &one , &one , descal , 
	                            x0 ,  &one , &one ,descxl , 
				    &beta , b0 , &one , &one , descbl );
    // copy B to bl for inversion 
    for(r=0; r<mla; r++)
    {
      for(c=0; c<nrhs; c++)
      { 
        bl[r+c*mla]=  b0[r+c*mla];
      }
    }
    if(mycol==0)
    {
  //    if(myrank==0) printf("Matrix B=[\n");
  //    print_matrix("", mla, nrhs, bl, mla );
    }
    
    /// ////////////////////////////////////////////////////////////////////////////////////////// 
    /// Computes the solution to the system of linear equations with a square distributed 
    /// matrix and multiple right-hand sides.
    /// need mg==ng in this case
    /*
     psgesv_(&ng, &one, Al, &one, &one,descal, ipiv, bl, &one,&one,descbl,  &info);
     //printf("info=%d \n",info);
     if(mycol==0)
     {
       if(myrank==0) printf("Matrix X=[\n");
       print_matrix("", nla, nrhs, bl, nla );
     }
     */
    
    /// ////////////////////////////////////////////////////////////////////////////////////////// 
    /// Solves overdetermined or underdetermined linear systems involving a matrix of full rank.
    if(mg<ng)
    {
	int lwork=-1;  float wkopt;   float* work;      
	psgels_("N" , &mg , &mg , &nrhs , Al , &one, &one, descal, bl, 
			    &one,&one,descbl, &wkopt, &lwork, &info ); 
	lwork = (int)wkopt;
	work = (float*)malloc( lwork*sizeof(float) );
	// Solve the equations A*X = B 
	psgels_("N" , &mg , &mg , &nrhs , Al , &one, &one,descal,bl, 
			  &one,&one,descbl, work, &lwork, &info ); 
	//printf("info=%d \n",info);
	}
    else
    {
        int lwork=-1;  float wkopt;   float* work;      
	psgels_("N" , &mg , &ng , &nrhs , Al , &one, &one, descal, bl, 
			&one,&one,descbl, &wkopt, &lwork, &info ); 
	lwork = (int)wkopt;
	work = (float*)malloc( lwork*sizeof(float) );
	// Solve the equations A*X = B 
	psgels_("N" , &mg , &ng , &nrhs , Al , 
		&one, &one,descal,bl, &one,&one,descbl, work, &lwork, &info ); 
	//printf("info=%d \n",info);
    }
 
    if(mycol==0)
    {
       if(myrank==0) printf("LS Matrix X=[\n");
       print_matrix("", nla, nrhs, bl, nla );
    } 
    /// ////////////////////////////////////////////////////////////////////////////////////////// 
    /// finish and leave
    free(Al);
    free(ipiv);
    free(bl);
    Cblacs_exit( 0 );

    return 0;
    
  }
  

//////////////////////////////////////////////////////////////////////
void print_matrix( char* desc, int m, int n, float* a, int lda ) 
{
        int i, j;
       // printf( " %s\n", desc );
        for( i = 0; i < m; i++ ) 
	{
                for( j = 0; j < n; j++ ) printf( " %6.4f", a[i+j*lda] );
                printf( "\n" );
        }
}

//////////////////////////////////////////////////////////////////////
//  void psgels (char *trans , MKL_INT *m , MKL_INT *n , MKL_INT *nrhs , float *a , MKL_INT *ia , MKL_INT *ja , 
//                 MKL_INT *desca , float *b , MKL_INT *ib , MKL_INT *jb , MKL_INT *descb ,
//                 float *work , MKL_INT *lwork , MKL_INT *info );
//////////////////////////////////////////////////////////////////////
// void psgemm (const char *transa , const char *transb , const MKL_INT *m , const MKL_INT *n , const MKL_INT *k , 
//                const float *alpha , const float *a , const MKL_INT *ia , const MKL_INT *ja , const MKL_INT *desca , 
//                const float *b , const MKL_INT *ib , const MKL_INT *jb , const MKL_INT *descb , const float *beta , 
//                float *c , const MKL_INT *ic , const MKL_INT *jc , const MKL_INT *descc );
//////////////////////////////////////////////////////////////////////
