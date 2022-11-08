#include <iostream>
#include <climits>

#define BS 1024

using namespace std;

__global__ void bfs(int* dflag, int* ddist, int* dvertex_pointers, int* dedges, int n, int m){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<n){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(ddist[e]>(ddist[tid]+1)){
                ddist[e]=ddist[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void initialize(int* ddist, int n){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid<n){
        ddist[tid]=INT_MAX/2;
    }
}

int main(){
    int n=11;
    int m=20;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    int vertex_pointers[12];
    int edges[20]={7,1,9,10,0,3,2,6,7,5,7,8,4,3,3,0,4,4,0,0};
    vertex_pointers[0]=0;
    vertex_pointers[1]=4;
    vertex_pointers[2]=5;
    vertex_pointers[3]=6;
    vertex_pointers[4]=9;
    vertex_pointers[5]=12;
    vertex_pointers[6]=13;
    vertex_pointers[7]=14;
    vertex_pointers[8]=17;
    vertex_pointers[9]=18;
    vertex_pointers[10]=19;
    vertex_pointers[11]=20;

    int root=7;

    int* dvertex_pointers;
    cudaMalloc((void**)&dvertex_pointers,sizeof(int)*(n+1));
    cudaMemcpy(dvertex_pointers,vertex_pointers,sizeof(int)*(n+1),cudaMemcpyHostToDevice);

    int* dedges;
    cudaMalloc((void**)&dedges,sizeof(int)*m);
    cudaMemcpy(dedges,edges,sizeof(int)*m,cudaMemcpyHostToDevice);

    int dist[11];

    int* ddist;
    cudaMalloc((void**) &ddist,sizeof(int)*n);
    initialize<<<blocksPerGrid,threadsPerBlock>>>(ddist,n);
    cudaMemcpy(dist,ddist,sizeof(int)*n,cudaMemcpyDeviceToHost);
    
    dist[root]=0;

    int flag=1;
    int* dflag;
    cudaMalloc(&dflag,sizeof(int));

    while(flag){
        flag=0;
        cudaMemcpy(ddist,dist,sizeof(int)*n,cudaMemcpyHostToDevice);
        cudaMemcpy(dflag,&flag,sizeof(int),cudaMemcpyHostToDevice);
        bfs<<<blocksPerGrid,threadsPerBlock>>>(dflag,ddist,dvertex_pointers,dedges,n,m);
        cudaMemcpy(dist,ddist,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
    }

    for(int i=0;i<n;i++){
        cout<<(i+1)<<" "<<dist[i]<<endl;
    }
}