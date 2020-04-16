/******
program ples
!       Parallel Linear Equation solver
!       Array A  = ( MP*MLA x NP*NLA )
!       Proc. grid is MP     x NP
!       Local mat. (al) MLA  x    NLA
!       array A(ig,jg) = 0.1*ig + 0.001*jg  (ig not= jg)
!                      = ig                 (ig    = jg)
!       Solve Ax=b, b=1
******/
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <mkl_cblas.h>
//#include <mkl_blas.h>
//#include <mkl_scalapack.h>
#include <mkl.h>
#include "mpi.h"
#define DS 11  // DESCRIPTOR_SIZE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define Mg 8 // Number of rows in global Matrix
#define Ng 8 // Number of columns in global Matrix

#define Mp 2 // Number of splits of A in M direction
#define Np 2 // Number of splits of A in N direction

extern void print_matrix( char* desc, int m, int n, float* a, int lda );
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
extern void   Cblacs_pinfo( int* mypnum, int* nprocs);
extern void   Cblacs_get( int context, int request, int* value);
extern int    Cblacs_gridinit( int* context, char * order, int np_row, int np_col);
extern void   Cblacs_gridinfo( int context, int*  np_row, int* np_col, int*  my_row, int*  my_col);
extern void   Cblacs_gridexit( int context);
extern void   Cblacs_exit( int error_code);
extern void   Cblacs_gridmap( int* context, int* map, int ld_usermap, int np_row, int np_col);
*/
//void setarray(float *a, int myrow, int mycol, int lda_x, int lda_y);
//void descinit_(int *idescal, int *m,int *n,int *mb,int *nb, int *dummy1 , int *dummy2 , int *icon, int *mla, int *info);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int icon;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------
// MPI Variables
//MPI_Comm MPI_COMM_WORLD;
int mysize, myrank;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc ,char *argv[])
{
  int myproc,noproc;
  int Mp_ret,Np_ret,myrow, mycol;
  float *Al,*bl;
  int *ipiv;

//	m=mla*mp;
//	n=nla*np;
	/*int mype,npe;
	
	float *al, *b;
	int *ipiv;
	int nprow=2 , npcol=2;
	int ib;
	int info;
	int zero=0;
	int one=1;
	int  m,n;
	m=mla*mp;
	n=nla*np;*/
	
/////////////////////////////////////////////////////////////////////////////////////////////
// Initialize MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mysize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if(myrank==0)
    {
      printf("Test Program for LS QR\n");
      printf("Using %d procs: Proc=%d\n",mysize,myrank);
    }
/////////////////////////////////////////////////////////////////////////////////////////////   
// set up basic linear alegbra comms
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

/////////////////////////////////////////////////////////////////////////////////////////////
    // Initializes the array descriptor for distributed matrix
    int info=0,zero=0,one=1;
    int descal[DS], descb[DS];
    int mg=Mg,ng=Ng;
    int mla=mg/Mp,nla=ng/Np;
    int mb=2,nb=2;
    descinit_(descal, &mg, &ng  , &mb, &nb , &zero, &zero, &icon, &mla, &info);
    info=0;
    descinit_(descb,  &mg, &one, &nb, &one, &zero, &zero, &icon, &mla, &info);

    Al = (float*)malloc(sizeof(float)*mla*nla);
    bl = (float*)malloc(sizeof(float)*mla);
    ipiv=(int*)malloc(sizeof(int)*(mla+mb));
 
    int r,c;
    for(r=0; r<mla; r++)
    {
      for(c=0; c<nla; c++)
      {
	Al[r+c*mla] = 2 + 5*drand48() + r + (mla*myrow)+(c+mycol*nla)*mg; 
      }
    }
    if(myrank==0)
    {
      printf( "\nMatrix Size Ag (%d x %d) Al (%d x %d)\n", mg, ng, mla, nla );
      printf( "\nMatrix Size Bg (%d x %d) Al (%d x %d)\n", mg, one, mla, one );
      printf("Proc=%d\n",myrank);
      print_matrix( "Matrix A=[", mla, nla, Al, mla );
    }
   /* for(r=1; r<mysize; r++)
    {
      if(r==myrank)
      {
      printf("Proc=%d\n",myrank);
      print_matrix( "Matrix A=[", mla, nla, Al, mla );
      }
    }*/

    for(r=0;r<mla;r++) 
    bl[r]=1.0;
    
    // Computes the solution to the system of linear equations with a square distributed 
    // matrix and multiple right-hand sides.
    psgesv_(&ng, &one, Al, &one, &one,descal, ipiv, bl, &one,&one,descb,  &info);
    
    printf("info=%d \n",info);

  /*  if(mycol == 0 )
    {    
      printf("x=(%2d%2d) %8.4f %8.4f\n",myrow,mycol,bl[0],bl[1]);
    }*/
    
    for(r=0;r<mla;r++) 
    printf("myrank=%d r=%d rg=%d x=%8.4f\n",myrank,r,r+(mla*myrow),bl[r]);

    
    free(Al);
    free(ipiv);
    free(bl);
    Cblacs_exit( 0 );

    //MPI_Finalize();
    return 0;
    
  }
  
//////////////////////////////////////////////////////////////////////
//  void psgesv_(MKL_INT *n , MKL_INT *nrhs , float *a , MKL_INT *ia , MKL_INT *ja , 
//               MKL_INT *desca , MKL_INT *ipiv , float *b , MKL_INT *ib , 
//               MKL_INT *jb , MKL_INT *descb , MKL_INT *info );  
//  void psgels (char *trans , MKL_INT *m , MKL_INT *n , MKL_INT *nrhs , float *a , MKL_INT *ia , MKL_INT *ja , 
//                 MKL_INT *desca , float *b , MKL_INT *ib , MKL_INT *jb , MKL_INT *descb ,
//                 float *work , MKL_INT *lwork , MKL_INT *info );

//////////////////////////////////////////////////////////////////////
void print_matrix( char* desc, int m, int n, float* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.4f", a[i+j*lda] );
                printf( "\n" );
        }
}
//////////////////////////////////////////////////////////////////////

 /*
float **matrix(int nrl,int nrh,int ncl,int nch);
int mod(int i, int j);
*/
 /*
void setarray(float *a,int myrow, int mycol, int lda_x, int lda_y){
      float **aa;
      float ll,mm,cr,cc;
      int ii,jj,i,j,pr,pc,h,g;
      int nprow = 2, npcol = 2;
      int n=8,m=8,nb=2,mb=2,rsrc=0,csrc=0;
      int n_b = 1;
      int index;
      aa=matrix(1,8,1,8);
      for (i=1;i<=8;i++) {
      	for (j=1;j<=8;j++) {
         if(i == j){
           aa[i][i]=i;
         }
         else {
            aa[i][j]=0.1*i+0.001*j; 
         }
         }
      }
      for (i=1;i<=m;i++) {
      	for (j=1;j<=n;j++) {
      // finding out which pe gets this i,j element
              cr = (float)( (i-1)/mb );
              h = rsrc+(int)(cr);
              pr = mod( h,nprow);
              cc = (float)( (j-1)/mb );
              g = csrc+(int)(cc);
              pc = mod(g,nprow);
      // check if on this pe and then set a
              if (myrow == pr && mycol==pc){
      // ii,jj coordinates of local array element
      // ii = x + l*mb
      // jj = y + m*nb
                  ll = (float)( ( (i-1)/(nprow*mb) ) );
                  mm = (float)( ( (j-1)/(npcol*nb) ) );
                  ii = mod(i-1,mb) + 1 + (int)(ll)*mb;
                  jj = mod(j-1,nb) + 1 + (int)(mm)*nb;
                  index=(jj-1)*lda_x+ii;
                  index=index-1;
//                  a(ii,jj) = aa(i,j) 
                  a[index] = aa[i][j];
              }
          }
		}

  	}
*/

 /*
int mod(int i, int j) 
{
	return (i % j);
}*/
 
 /*
float **matrix(int nrl,int nrh,int ncl,int nch)
{
    int i;
	float **m;
	m=(float **) malloc((unsigned) (nrh-nrl+1)*sizeof(float*));
	if (!m){
	     printf("allocation failure 1 in matrix()\n");
	     exit(1);
	}
	m -= nrl;
	for(i=nrl;i<=nrh;i++) {
	    if(i == nrl){ 
		    m[i]=(float *) malloc((unsigned) (nrh-nrl+1)*(nch-ncl+1)*sizeof(float));
		    if (!m[i]){
		         printf("allocation failure 2 in matrix()\n");
		         exit(1);
		    }		
		    m[i] -= ncl;
	    }
	    else {
	        m[i]=m[i-1]+(nch-ncl+1);
	    }
	}
	return m;
}
*/
 
