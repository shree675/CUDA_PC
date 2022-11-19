/*
* Program to obtain the cut vertices of a graph given in CSR format. 
* Output is the list of cut vertices.
* To execute, run the following command on a system with CUDA support enabled GPUs
* bash run.sh cuda_cut_vertex.cu
*/

#include <iostream>
#include <climits>
#include <chrono>

#define BS 17
#define N 17
#define M 50
#define ROOT 0

using namespace std;

__global__ void bfs(int* dflag, int* dlevel, int* dvertex_pointers, int* dedges){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<N){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(dlevel[e]>(dlevel[tid]+1)){
                dlevel[e]=dlevel[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void truncated_bfs(int* dlevel, int* dvertex_pointers, int* dedges, int* ddist, int u, int* dflag, int* dreached){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;
    
    if(tid<N && tid!=u){
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

__global__ void find_cut_vertices(int* dlevel, int* dvertex_pointers, int* dedges, int* dcut_vertex){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(N+BS-1)/BS;

    if(tid<N && tid!=ROOT){
        for(int j=dvertex_pointers[tid];j<dvertex_pointers[tid+1];j++){
            
            int e=dedges[j];

            int* dflag;
            int* ddist;
            cudaMalloc((void**) &dflag,sizeof(int));
            int* dreached;
            cudaMalloc((void**) &dreached,sizeof(int));
            cudaMalloc((void**) &ddist,sizeof(int)*N);
            for(int i=0;i<N;i++){
                ddist[i]=INT_MAX/2;
            }
            ddist[e]=0;
            *dflag=1;
            *dreached=0;
            while(*dflag && !(*dreached)){
                *dflag=0;
                truncated_bfs<<<blocksPerGrid,N>>>(dlevel,dvertex_pointers,dedges,ddist,tid,dflag,dreached);
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
            if(tid==ROOT){
                break;
            }
        }
    }
    else if(tid==ROOT){
        int e=dedges[dvertex_pointers[tid]];
        int* dflag;
        int* ddist;
        cudaMalloc((void**) &dflag,sizeof(int));
        cudaMalloc((void**) &ddist,sizeof(int)*N);
        int* dreached;
        cudaMalloc((void**) &dreached,sizeof(int));
        for(int i=0;i<N;i++){
            ddist[i]=INT_MAX/2;
        }
        ddist[e]=0;
        *dflag=1;
        *dreached=0;
        while(*dflag){
            *dflag=0;
            truncated_bfs<<<blocksPerGrid,N>>>(ddist,dvertex_pointers,dedges,ddist,tid,dflag,dreached);
            if (cudaSuccess != cudaDeviceSynchronize()) {
                return;
            }
        }
        for(int i=0;i<N;i++){
            if(ddist[i]>=INT_MAX/2 && i!=tid){
                dcut_vertex[tid]=1;
                break;
            }
        }
    }
}

__global__ void initialize(int* dlevel){
    for(int i=0;i<N;i++){
        dlevel[i]=INT_MAX/2;
    }
    dlevel[ROOT]=0;
}

int main(){
    int threadsPerBlock=BS;
    int blocksPerGrid=(N+BS-1)/BS;

    int vertex_pointers[N+1];
    int edges[M]={1,2,0,2,5,0,1,3,5,4,2,12,11,2,5,1,2,4,6,7,7,5,5,6,8,10,7,10,10,7,8,9,12,3,13,3,11,13,14,11,12,14,15,13,12,16,14,16,14,15};
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

    // Start Timer
    float time,total_time=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    auto start_t = chrono::high_resolution_clock::now();

    int* dvertex_pointers;
    cudaMalloc((void**)&dvertex_pointers,sizeof(int)*(N+1));
    cudaMemcpy(dvertex_pointers,vertex_pointers,sizeof(int)*(N+1),cudaMemcpyHostToDevice);

    int* dedges;
    cudaMalloc((void**)&dedges,sizeof(int)*M);
    cudaMemcpy(dedges,edges,sizeof(int)*M,cudaMemcpyHostToDevice);

    int cut_vertex[N]={0};
    int* dcut_vertex;
    cudaMalloc((void**) &dcut_vertex,sizeof(int)*N);
    cudaMemcpy(dcut_vertex,cut_vertex,sizeof(int)*N,cudaMemcpyHostToDevice);

    int level[N];

    int* dlevel;
    cudaMalloc((void**) &dlevel,sizeof(int)*N);

    cudaEventRecord(start,0);
    initialize<<<blocksPerGrid,threadsPerBlock>>>(dlevel);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;

    cudaMemcpy(level,dlevel,sizeof(int)*N,cudaMemcpyDeviceToHost);

    int flag=1;
    int* dflag;
    cudaMalloc((void**) &dflag,sizeof(int));

    while(flag){
        flag=0;
        cudaMemcpy(dlevel,level,sizeof(int)*N,cudaMemcpyHostToDevice);
        cudaMemcpy(dflag,&flag,sizeof(int),cudaMemcpyHostToDevice);

        cudaEventRecord(start,0);
        bfs<<<blocksPerGrid,N>>>(dflag,dlevel,dvertex_pointers,dedges);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);
        total_time+=time;
        
        cudaMemcpy(level,dlevel,sizeof(int)*N,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(dlevel,level,sizeof(int)*N,cudaMemcpyHostToDevice);

    cudaEventRecord(start,0);
    find_cut_vertices<<<blocksPerGrid,N>>>(dlevel,dvertex_pointers,dedges,dcut_vertex);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;
    
    cudaDeviceSynchronize();
    cudaMemcpy(&cut_vertex,dcut_vertex,sizeof(int)*N,cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventElapsedTime(&time,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto stop_t = chrono::high_resolution_clock::now();

    cout<<"Cut vertices: ";
    for(int i=0;i<N;i++){
        if(cut_vertex[i]==1){
            cout<<i<<" ";
        }
    }
    cout<<endl;

    auto duration = chrono::duration_cast<chrono::microseconds>(stop_t - start_t);
    printf("CPU Time Taken: %f ms\n", ((float) duration.count())/1000.0);

    printf("GPU Time Taken: %f ms\n", total_time);
}
