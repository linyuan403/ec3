#include <iostream>
//#include <math.h>
//#include <queue>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//#include <fstream>

#define SIZE 32

typedef struct{
  int n;
  bool a[SIZE][SIZE];
} _ADJ;

__host__ void print(_ADJ t){
  std::cout<<"n="<<t.n<<": "<<std::endl;
  for(int i=0;i<t.n;i++){
    std::cout<<"(";
    for(int j=0;j<t.n;j++){
      std::cout<<t.a[i][j];
    }
    std::cout<<")"<<std::endl;
  }
  std::cout<<std::endl;
}

__host__ __device__  void copy(_ADJ *to, _ADJ *from){
  to->n=from->n;
  for(int i=0;i<from->n;i++){
     for(int j=0;j<from->n;j++){
       to->a[i][j]=from->a[i][j];
     }
  }
}


__host__ __device__ int BadTriples(_ADJ t){
  int n=t.n;	    
  int count=0;
  for(int i=0; i<n-2; i++){
    for(int j=i+1; j<n-1;j++){
      for(int k=j+1; k <n; k++){
	bool b[8];
	int s;
	for(s=0;s<8;s++){
	  b[s]=0;
	}
	for(int l=0;l<n;l++){
	  if(l!=i && l!=j && l!=k){
	    s=0;
	    if(t.a[i][l]) s=s+4;
	    if(t.a[j][l]) s=s+2;
	    if(t.a[k][l]) s=s+1;
	    b[s]=1;
	  }
	}
	if(not (b[0] && b[1] && b[2] && b[3] && b[4] && b[5] && b[6] && b[7])) count++;                 
      }
    }
  }
  return count;
}
     

__global__ void VecBadTriples(_ADJ* t, int* y, int N){
  int i = blockDim.x * blockIdx.x + threadIdx.x; 
  if (i < N)  {
    y[i]=BadTriples(t[i]);
  }
}


void checkCUDAError(const char *msg){

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
    {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
              cudaGetErrorString( err) );
      exit(EXIT_FAILURE);
    }                         
}


int main(int argc, char* argv[]){

  if(argc <2){
    std::cout<<"Usage:"<<std::endl;
    std::cout<<argv[0]<<" n target"<<std::endl;
    return 0;
  }

  int n;
  n=atoi(argv[1]);
  
  int target;
  target=atoi(argv[2]);
 
  
  _ADJ cur;
  _ADJ tmp;
  
  
  srand(time(0));
  cur.n=n;
  for(int i=0;i<n;i++){
    for(int j=i;j<n;j++){
      
      
      if(rand()%2==0){
	cur.a[i][j]=cur.a[j][i]=0;
      }
      else{
	cur.a[i][j]=cur.a[j][i]=1;
      }
    }
  }
 
  int bt_cur= BadTriples(cur);
  int bt_tmp;
  int bt_next=bt_cur;
  std::cout<<"Bad trples ="<<bt_cur<<std::endl;
  
  int e=n*(n-1)/2;
  int N=e*(e-1)/2;
  
  std::cout<<"e="<<e<<" N="<<N<<std::endl;

  int memsize = N * sizeof(_ADJ);
  _ADJ * t0 = (_ADJ*) malloc(memsize);

  _ADJ * dev_t0;
  cudaMalloc((void**)&dev_t0, memsize);
  

  int * bt = (int *) malloc(N * sizeof(int));
  int * dev_bt;
  cudaMalloc((void**)&dev_bt, N* sizeof(int));
  checkCUDAError("cudaMalloc dev_bt");

  while(bt_cur>target){
    std::cout<<"Reshaffling..."<<std::endl;
    //flip a small portion 1/n of edges
    for(int i=0;i<n;i++){ 
    for(int j=i;j<n;j++){ 
      cur.n=n;
      if(rand()%(n)==0){
	cur.a[j][i]=1-cur.a[j][i]; //flip
	cur.a[i][j]=cur.a[j][i];
      }
    }
    }
    
    bt_cur= BadTriples(cur);
    copy(&tmp, &cur);
    bt_tmp= bt_cur;
    
    while(true){
      
    std::cout<<"Enter while loop..."<<std::endl;
    std::cout<<"bad triples="<<bt_cur<<std::endl;
    int s=0;
    for(int i=0;i<n-1;i++){
      for(int j=i+1; j<n;j++){
	for(int k=i;k<n-1; k++){
	  for(int l=k+1;l<n;l++){
	    if(k>i || l>j){
	      for(int p=k;p<n-1;p++){
		for(int q=p+1;q<n;q++){
		  if(p>k || q>l){		  
		    copy(&t0[s], &cur); 
		    t0[s].a[i][j]=1-t0[s].a[i][j]; //flip edge ij
		    t0[s].a[j][i]=t0[s].a[i][j];
		    
		    t0[s].a[k][l]=1-t0[s].a[k][l]; //flip edge kl
		    t0[s].a[l][k]=t0[s].a[k][l];
		    
		    t0[s].a[p][q]=1-t0[s].a[p][q]; //flip edge pq
		    t0[s].a[q][p]=t0[s].a[p][q];
		    s++;

		    if(s==N){
		      cudaMemcpy(dev_t0, t0, memsize,  cudaMemcpyHostToDevice);
		      checkCUDAError("cudaMemcpyHostToDevice");
  
		      VecBadTriples<<<(N+1023)/1024, 1024>>>(dev_t0, dev_bt, N);
		      checkCUDAError("kernel VecBadTriples");

		      cudaMemcpy(bt, dev_bt, sizeof(int)*N,  cudaMemcpyDeviceToHost);
		      checkCUDAError("cudaMemcpyDeviceToHost");

		      bt_next=bt_tmp;
		      int min_x=-1;
		      for(int x=0; x<N;x++){
			if(bt[x]<bt_next){
			  min_x=x;
			  bt_next=bt[x];
			}
		      }

		      if(min_x>0){
			copy(&tmp, &t0[min_x]);
			bt_tmp=bt_next;
			//std::cout<<"bad triples="<<bt_tmp<<std::endl;
		      }
		      s=0;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    if(s>0){
      cudaMemcpy(dev_t0, t0, memsize,  cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpyHostToDevice");
    
      VecBadTriples<<<(s+1023)/1024, 1024>>>(dev_t0, dev_bt, s);
      checkCUDAError("kernel VecBadTriples");

      cudaMemcpy(bt, dev_bt, sizeof(int)*s,  cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpyDeviceToHost");

      bt_next=bt_tmp;
      int min_x=-1;
      for(int x=0;x<s;x++){
    
	if(bt[x]<bt_next){
	  min_x=x;
	  bt_next=bt[x];
	}
      }

      if(min_x>0){
	copy(&tmp, &t0[min_x]);
	bt_tmp=bt_next;
	//std::cout<<"bad triples="<<bt_tmp<<std::endl;
      }
    }
    
    if(bt_tmp<bt_cur){
      copy(&cur, &tmp);
      bt_cur=bt_tmp;
    }
    else{
      break;
    }
    }
    std::cout<<"bad triples="<<bt_cur<<std::endl;
  }

  print(cur);
  std::cout<<"Bad trples ="<<BadTriples(cur)<<std::endl;

  cudaFree(dev_t0);
  cudaFree(dev_bt);
  free(t0);
  free(bt);
}
