/*
* Program to obtain the biconnected components of a graph given in CSR format. 
* Output is the list of components identified by assigning a uniquer number to all edges in a component.
* To execute, run the following command on a system with CUDA support enabled GPUs
* bash run.sh cuda_bcc.cu
*/

#include <iostream>
#include <thrust/device_vector.h>
#include <chrono>

#define ROOT 0

// Test Case 1
#define BS 17
#define N 17
#define M 50

// Test Case 2
// #define BS 7
// #define N 7
// #define M 20

// Test Case 3
// #define BS 22
// #define N 22
// #define M 54

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

__global__ void find_cut_vertices(int* dlevel, int* dvertex_pointers, int* dedges, int* dcut_vertex, int* dunsafe_vertex){
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
                dunsafe_vertex[e]=tid;
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
                dunsafe_vertex[e]=tid;
                break;
            }
        }
    }
}

__device__ void set_bcc_id(int* dvertex_pointers, int* dedges, int* dvisited, int* dcurrent_cut_vertex, int u, int* dminimum, int* dbcc){
    for(int i=dvertex_pointers[u];i<dvertex_pointers[u+1];i++){
        if(dvisited[i]){
            return;
        }
        int v=dedges[i];

        dvisited[i]=1;
        // for(int j=dvertex_pointers[v];j<dvertex_pointers[j+1];j++){
        //     if(dedges[j]==u){
        //         dvisited[j]=1;
        //         dbcc[j]=*dminimum;
        //         break;
        //     }
        // }
        dbcc[i]=*dminimum;
        
        if(v!=*dcurrent_cut_vertex){
            set_bcc_id(dvertex_pointers,dedges,dvisited,dcurrent_cut_vertex,v,dminimum,dbcc);
        }
    }
}

__device__ void dfs(int* dvertex_pointers, int* dedges, int* lcl_dvisited, int* dcurrent_cut_vertex, int* dminimum, int u, int* dcut_vertex){
    for(int i=dvertex_pointers[u];i<dvertex_pointers[u+1];i++){
        if(lcl_dvisited[i]){
            return;
        }
        int v=dedges[i];

        if(v!=*dcurrent_cut_vertex){
            atomicMin(dminimum,v);
        }

        lcl_dvisited[i]=1;
        
        if(v!=*dcurrent_cut_vertex){
            dfs(dvertex_pointers,dedges,lcl_dvisited,dcurrent_cut_vertex,dminimum,v,dcut_vertex);
        }
    }
}

__global__ void copy_visited(int* lcl_dvisited, int* dvisited){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid<M){
        lcl_dvisited[tid]=dvisited[tid];
    }
}

__global__ void find_bcc(int* dlevel, int* dvertex_pointers, int* dedges, int* dunsafe_vertex, int cur_level, int* dbcc, int* dvisited, int* dcut_vertex){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(N+BS-1)/BS;

    if(dlevel[tid]==cur_level && dunsafe_vertex[tid]!=-1){
        int* dminimum;
        cudaMalloc(&dminimum,sizeof(int));
        *dminimum=tid;

        int* dcurrent_cut_vertex;
        cudaMalloc(&dcurrent_cut_vertex,sizeof(int));
        *dcurrent_cut_vertex=dunsafe_vertex[tid];
        int* ddist;
        cudaMalloc((void**) &ddist,sizeof(int)*N);
        for(int i=0;i<N;i++){
            ddist[i]=INT_MAX/2;
        }
        ddist[tid]=0;

        int* lcl_dvisited;
        cudaMalloc(&lcl_dvisited,sizeof(int)*M);

        copy_visited<<<blocksPerGrid,M>>>(lcl_dvisited,dvisited);
        dfs(dvertex_pointers,dedges,lcl_dvisited,dcurrent_cut_vertex,dminimum,tid,dcut_vertex);

        set_bcc_id(dvertex_pointers,dedges,dvisited,dcurrent_cut_vertex,tid,dminimum,dbcc);
    }
}

__global__ void initialize(int* dlevel, int* dunsafe_vertex){
    for(int i=0;i<N;i++){
        dlevel[i]=INT_MAX/2;
        dunsafe_vertex[i]=-1;
    }
    dlevel[ROOT]=0;
}

__global__ void initialize_bcc(int* dbcc){
    for(int i=0;i<M;i++){
        dbcc[i]=-1;
    }
}

int main(){
    
    // Please note: On changing test case, change the #define to change the value of BS the same value as N
    // Test Case 1:----------------------
    // int n=17;
    // int m=50;
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
    // End of Test Case 1--------------------

    // Test Case 2: 
    // int n=7;
    // int m=20;

    // int vertex_pointers[N+1] = {0,1,7,9,12,15,18,20};
    // int edges[M]={1,0,2,3,4,5,6,1,3,1,2,4,1,3,5,1,4,6,1,5};
    // End of Test Case 2 --------------------

    // Test Case 3:
    // int n=22;
    // int m=54;

    // int vertex_pointers[N+1] = {0,2,3,5,8,11,13,16,18,20,25,27,29,31,33,36,40,42,44,46,50,52,54};
    // int edges[M]={2,7,19,0,3,2,4,5,3,6,7,3,6,4,5,8,0,4,6,9,8,10,13,14,19,9,11,10,12,11,13,9,12,9,15,18,14,16,17,18,15,17,16,15,14,15,1,9,20,21,19,21,19,20};


    // End of Test Case 3 --------------------

    float time,total_time=0;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock=BS;
    int blocksPerGrid=(N+BS-1)/BS;

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
    int unsafe_vertex[N];

    int* dlevel;
    cudaMalloc((void**) &dlevel,sizeof(int)*N);

    int* dunsafe_vertex;
    cudaMalloc((void**) &dunsafe_vertex,sizeof(int)*N);

    cudaEventRecord(start,0);
    initialize<<<blocksPerGrid,threadsPerBlock>>>(dlevel,dunsafe_vertex);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;

    cudaMemcpy(level,dlevel,sizeof(int)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(unsafe_vertex,dunsafe_vertex,sizeof(int)*N,cudaMemcpyDeviceToHost);

    int flag=1;
    int* dflag;
    cudaMalloc((void**) &dflag,sizeof(int));

    int max_level=0;
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

        max_level++;
        cudaMemcpy(level,dlevel,sizeof(int)*N,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(dunsafe_vertex,unsafe_vertex,sizeof(int)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(dlevel,level,sizeof(int)*N,cudaMemcpyHostToDevice);
    // At this point the levels of the tree obtained from performing BFS from the root node are available.

    cudaEventRecord(start,0);
    find_cut_vertices<<<blocksPerGrid,N>>>(dlevel,dvertex_pointers,dedges,dcut_vertex,dunsafe_vertex);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;

    // List of cut vertices and their corresponding unsafe vertices are available
    cudaDeviceSynchronize();
    cudaMemcpy(&cut_vertex,dcut_vertex,sizeof(int)*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(unsafe_vertex,dunsafe_vertex,sizeof(int)*N,cudaMemcpyDeviceToHost);

    // for(int i=0;i<N;i++){
    //     if(unsafe_vertex[i]!=-1){
    //         cout<<i<<" "<<unsafe_vertex[i]<<endl;
    //     }
    // }

    int bcc[M];
    int* dbcc;
    cudaMalloc(&dbcc,sizeof(int)*M);

    cudaEventRecord(start,0);
    initialize_bcc<<<blocksPerGrid,threadsPerBlock>>>(dbcc);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    total_time+=time;

    cudaMemcpy(bcc,dbcc,sizeof(int)*N,cudaMemcpyDeviceToHost);    

    int visited[M]={0};
    int* dvisited;
    cudaMalloc(&dvisited,sizeof(int)*M);

    for(int i=max_level+1;i>=0;i--){
        cudaMemcpy(dvisited,visited,sizeof(int)*M,cudaMemcpyHostToDevice);
        cudaMemcpy(dbcc,bcc,sizeof(int)*M,cudaMemcpyHostToDevice);

        cudaEventRecord(start,0);
        find_bcc<<<blocksPerGrid,N>>>(dlevel,dvertex_pointers,dedges,dunsafe_vertex,i,dbcc,dvisited,dcut_vertex);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time,start,stop);
        total_time+=time;

        cudaDeviceSynchronize();
        cudaMemcpy(bcc,dbcc,sizeof(int)*M,cudaMemcpyDeviceToHost);
        cudaMemcpy(visited,dvisited,sizeof(int)*M,cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop,0);
    cudaEventElapsedTime(&time,start,stop);

    auto stop_t = chrono::high_resolution_clock::now();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Edge    BCC\n");
    for(int i=0;i<N;i++){
        for(int j=vertex_pointers[i];j<vertex_pointers[i+1];j++){
            if(level[i]>level[edges[j]]){
                printf("%d-%d %6d\n",i,edges[j],bcc[j]);
            }
            else if(level[i]==level[edges[j]] && i<edges[j]){
                printf("%d-%d %6d\n",i,edges[j],bcc[j]);
            }
            // cout<<i<<"-"<<edges[j]<<" "<<bcc[j]<<endl;
        }
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(stop_t - start_t);
    printf("CPU Time Taken: %f ms\n", ((float) duration.count())/1000.0);

    printf("GPU Time Taken: %f ms\n", total_time);
}