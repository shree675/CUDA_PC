#include <iostream>
#include <climits>

#define BS 17

using namespace std;

__global__ void bfs(int* dflag, int* dlevel, int* dvertex_pointers, int* dedges, int n){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<n){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(dlevel[e]>(dlevel[tid]+1)){
                dlevel[e]=dlevel[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void truncated_bfs(int* dlevel, int* dvertex_pointers, int* dedges, int* ddist, int n, int u, int* dflag, int* dreached){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<n && tid!=u){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(e==u){
                continue;
            }
            
            if(ddist[e]>(ddist[tid]+1)){
                if(dlevel[e]<=dlevel[u]){
                    *dreached=1;
                }
                ddist[e]=ddist[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void find_cut_vertices(int* dlevel, int* dvertex_pointers, int* dedges, int* dcut_vertex, int n, int root){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    if(tid<n && tid!=root){
        // TODO: separate check for root vertex
        for(int j=dvertex_pointers[tid];j<dvertex_pointers[tid+1];j++){
            
            int e=dedges[j];

            int* dflag;
            int* ddist;
            cudaMalloc((void**) &dflag,sizeof(int));
            int* dreached;
            cudaMalloc((void**) &dreached,sizeof(int));
            cudaMalloc((void**) &ddist,sizeof(int)*n);
            for(int i=0;i<n;i++){
                ddist[i]=INT_MAX/2;
            }
            ddist[e]=0;
            *dflag=1;
            *dreached=0;
            while(*dflag && !(*dreached)){
                *dflag=0;
                truncated_bfs<<<1,n>>>(dlevel,dvertex_pointers,dedges,ddist,n,tid,dflag,dreached);
                if (cudaSuccess != cudaDeviceSynchronize()) {
                    return;
                }
            }
            if (cudaSuccess != cudaDeviceSynchronize()) {
                return;
            }
            if(!(*dreached)){
                dcut_vertex[tid]=1;
                break;
            }
            if(tid==root){
                break;
            }
        }
    }
    else if(tid==root){
        int e=dedges[dvertex_pointers[tid]];
        int* dflag;
        int* ddist;
        cudaMalloc((void**) &dflag,sizeof(int));
        cudaMalloc((void**) &ddist,sizeof(int)*n);
        int* dreached;
        cudaMalloc((void**) &dreached,sizeof(int));
        bool done=false;
        for(int i=0;i<n;i++){
            ddist[i]=INT_MAX/2;
        }
        ddist[e]=0;
        *dflag=1;
        *dreached=0;
        while(*dflag){
            *dflag=0;
            truncated_bfs<<<1,n>>>(ddist,dvertex_pointers,dedges,ddist,n,tid,dflag,dreached);
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
    int m=50;
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

    int level[17];
    for(int i=0;i<n;i++){
        level[i]=INT_MAX/2;
    }
    level[root]=0;
    int* dlevel;
    cudaMalloc((void**) &dlevel,sizeof(int)*n);

    int flag=1;
    int* dflag;
    cudaMalloc((void**) &dflag,sizeof(int));

    while(flag){
        flag=0;
        cudaMemcpy(dlevel,level,sizeof(int)*n,cudaMemcpyHostToDevice);
        cudaMemcpy(dflag,&flag,sizeof(int),cudaMemcpyHostToDevice);
        bfs<<<1,n>>>(dflag,dlevel,dvertex_pointers,dedges,n);
        cudaMemcpy(level,dlevel,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(dlevel,level,sizeof(int)*n,cudaMemcpyHostToDevice);
    find_cut_vertices<<<1,n>>>(dlevel,dvertex_pointers,dedges,dcut_vertex,n,root);
    cudaDeviceSynchronize();
    cudaMemcpy(&cut_vertex,dcut_vertex,sizeof(int)*n,cudaMemcpyDeviceToHost);

    for(int i=0;i<n;i++){
        if(cut_vertex[i]==1){
            cout<<i<<" ";
        }
    }

}
