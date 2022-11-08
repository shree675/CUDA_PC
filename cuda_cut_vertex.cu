#include <iostream>
#include <climits>

#define BS 17

using namespace std;

__global__ void bfs(int* dvertex_pointers, int* dedges, int* ddist, int* dvertices, int n, int m, int u, int* dflag){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<n && tid!=u){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(e==u){
                continue;
            }
            if(ddist[e]>(ddist[tid]+1)){
                ddist[e]=ddist[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void find_cut_vertices(int* dvertex_pointers, int* dedges, int* dcut_vertex, int n, int m){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    if(tid<n){
        int* dvertices;
        cudaMalloc((void**) &dvertices,sizeof(int)*n);
        for(int i=0;i<n;i++){
            dvertices[i]=0;
        }
        int* dflag;
        int* ddist;
        cudaMalloc((void**) &dflag,sizeof(int));
        // printf("%d ",dflag);
        cudaMalloc((void**) &ddist,sizeof(int)*n);
        bool done=false;
        for(int i=0;i<n;i++){
            ddist[i]=INT_MAX/2;
            if(i!=tid && !done){
                ddist[i]=0;
                done=true;
            }
        }
        *dflag=1;
        while(*dflag){
            *dflag=0;
            bfs<<<1,n>>>(dvertex_pointers,dedges,ddist,dvertices,n,m,tid,dflag);            
            if (cudaSuccess != cudaDeviceSynchronize()) { 
                return; 
            } 
        }
        for(int i=0;i<n;i++){
            if(ddist[i]>=INT_MAX/2 && i!=tid){
                dcut_vertex[tid]=1;
                break;
            }
        }
    }
}

int main(){
    int n=17;
    int m=49;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    int vertex_pointers[18];
    int edges[50]={1,2,0,2,5,0,1,3,5,4,2,12,11,2,5,1,2,4,6,7,7,5,5,6,8,10,7,10,10,7,8,9,12,3,13,3,11,13,14,11,12,14,15,13,12,16,14,16,14,15};
    vertex_pointers[0]=0;
    vertex_pointers[1]=2;
    vertex_pointers[2]=5;
    vertex_pointers[3]=10;
    vertex_pointers[4]=13;
    vertex_pointers[5]=15;
    vertex_pointers[6]=20;
    vertex_pointers[7]=22;
    vertex_pointers[8]=26;
    vertex_pointers[9]=28;
    vertex_pointers[10]=29;
    vertex_pointers[11]=32;
    vertex_pointers[12]=35;
    vertex_pointers[13]=39;
    vertex_pointers[14]=42;
    vertex_pointers[15]=46;
    vertex_pointers[16]=48;
    vertex_pointers[17]=50;

    int root=0;

    int* dvertex_pointers;
    cudaMalloc((void**)&dvertex_pointers,sizeof(int)*(n+1));
    cudaMemcpy(dvertex_pointers,vertex_pointers,sizeof(int)*(n+1),cudaMemcpyHostToDevice);

    int* dedges;
    cudaMalloc((void**)&dedges,sizeof(int)*m);
    cudaMemcpy(dedges,edges,sizeof(int)*m,cudaMemcpyHostToDevice);

    int cut_vertex[17]={0};
    int* dcut_vertex;
    cudaMalloc((void**) &dcut_vertex,sizeof(int)*n);
    cudaMemcpy(dcut_vertex,cut_vertex,sizeof(int)*n,cudaMemcpyHostToDevice);

    find_cut_vertices<<<1,n>>>(dvertex_pointers,dedges,dcut_vertex,n,m);
    cudaDeviceSynchronize();
    cudaMemcpy(&cut_vertex,dcut_vertex,sizeof(int)*n,cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        if(cut_vertex[i]==1){
            cout<<i<<" ";
        }
    }

}
