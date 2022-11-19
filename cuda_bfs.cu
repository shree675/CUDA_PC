/*
* Program to perform Parallel BFS on a graph given in CSR format. 
* Output is the levels of each node in the graph with root vertex at level 0.
* To execute, run the following command on a system with CUDA support enabled GPUs
* bash run.sh cuda_bfs.cu
*/


#include <iostream>
#include <climits>
#include <chrono>

#define BS 1024
#define N 11
#define M 20
#define ROOT 0

using namespace std;

__global__ void bfs(int* dflag, int* ddist, int* dvertex_pointers, int* dedges){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int e;

    int *global_now;
    cudaMalloc(&global_now,sizeof(int));

    clock_t start = clock();
    clock_t now;
    for (;;) {
    now = clock();
    clock_t cycles = now > start ? now - start : now + (0xffffffff - start);
        if (cycles >= 10000) {
            break;
        }
    }
    // Stored "now" in global memory here to prevent the
    // compiler from optimizing away the entire loop.
    *global_now = now;
    
    if(tid<N){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            e=dedges[i];
            if(ddist[e]>(ddist[tid]+1)){
                ddist[e]=ddist[tid]+1;
                *dflag=1;
            }
        }
    }
}

__global__ void initialize(int* ddist){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid<N){
        ddist[tid]=INT_MAX/2;
    }
}

int main(){

    int threadsPerBlock=BS;
    int blocksPerGrid=(N+BS-1)/BS;

    int vertex_pointers[N+1];
    int edges[M]={7,1,9,10,0,3,2,6,7,5,7,8,4,3,3,0,4,4,0,0};
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

    // Start timer
    float time,total_time=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    auto start_t = chrono::high_resolution_clock::now();

    int* dvertex_pointers;
    cudaMalloc((void**)&dvertex_pointers,sizeof(int)*(N+1));
    cudaMemcpy(dvertex_pointers,vertex_pointers,sizeof(int)*(N+1),cudaMemcpyHostToDevice);

    int* dedges;
    cudaMalloc((void**)&dedges,sizeof(int)*M);
    cudaMemcpy(dedges,edges,sizeof(int)*M,cudaMemcpyHostToDevice);

    int dist[N];

    int* ddist;
    cudaMalloc((void**) &ddist,sizeof(int)*N);

    cudaEventRecord(start,0);
    initialize<<<blocksPerGrid,threadsPerBlock>>>(ddist);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;

    cudaMemcpy(dist,ddist,sizeof(int)*N,cudaMemcpyDeviceToHost);
    
    dist[ROOT]=0;

    int flag=1;
    int* dflag;
    cudaMalloc(&dflag,sizeof(int));

    while(flag){
        flag=0;
        cudaMemcpy(ddist,dist,sizeof(int)*N,cudaMemcpyHostToDevice);
        cudaMemcpy(dflag,&flag,sizeof(int),cudaMemcpyHostToDevice);

        cudaEventRecord(start,0);
        bfs<<<blocksPerGrid,threadsPerBlock>>>(dflag,ddist,dvertex_pointers,dedges);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);

        cudaMemcpy(dist,ddist,sizeof(int)*N,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
        total_time+=time;
    }
    
    auto stop_t = chrono::high_resolution_clock::now();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout<<"Vertex Distance\n";
    for(int i=0;i<N;i++){
        printf("%4d  %7d\n",i+1,dist[i]);
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(stop_t - start_t);
    printf("CPU Time Taken: %f ms\n", ((float) duration.count())/1000.0);

    printf("GPU Time Taken: %f ms\n", total_time);
    
}