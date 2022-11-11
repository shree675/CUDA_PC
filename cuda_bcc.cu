#include <iostream>

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

__global__ void find_cut_vertices(int* dlevel, int* dvertex_pointers, int* dedges, int* dcut_vertex, int n, int root, int* dunsafe_vertex){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    if(tid<n && tid!=root){
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
                dunsafe_vertex[e]=tid;
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
                dunsafe_vertex[e]=tid;
                break;
            }
        }
    }
}

__global__ void find_minimum(int* dlevel, int* dvertex_pointers, int* dedges, int* ddist, int n, int* dflag, int* dcurrent_cut_vertex, int* dminimum, int cur_level, int* dvisited){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    int e;
    if(dlevel[tid]>=(cur_level-1) && tid!=*dcurrent_cut_vertex){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            if(dvisited[i]){
                continue;
            }
            e=dedges[i];

            if(tid!=*dcurrent_cut_vertex){
                atomicMin(dminimum,tid);
            }
            
            if(ddist[e]>(ddist[tid]+1)){
                if(dlevel[e]<dlevel[*dcurrent_cut_vertex]){
                    continue;
                }
                ddist[e]=ddist[tid]+1;
                if(e!=*dcurrent_cut_vertex){
                    atomicMin(dminimum,e);
                }
                *dflag=1;
            }
        }
    }
}

__global__ void set_bcc_id(int* dlevel, int* dvertex_pointers, int* dedges, int* ddist, int n, int* dflag, int cur_level, int* dminimum, int* dcurrent_cut_vertex, int* dvisited, int* dbcc){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    int e;
    if(dlevel[tid]>=(cur_level-1)){
        for(int i=dvertex_pointers[tid];i<dvertex_pointers[tid+1];i++){
            if(dvisited[i]){
                continue;
            }
            dvisited[i]=1;
            e=dedges[i];
            
            if(ddist[e]>(ddist[tid]+1)){
                if(dlevel[e]<dlevel[*dcurrent_cut_vertex]){
                    continue;
                }
                ddist[e]=ddist[tid]+1;
                dbcc[i]=*dminimum;
                // printf("%d %d\n",*dminimum,i);
                *dflag=1;
            }
        }
    }
}

__global__ void find_bcc(int* dlevel, int* dvertex_pointers, int* dedges, int* dunsafe_vertex, int cur_level, int n, int* dbcc, int* dvisited){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int threadsPerBlock=BS;
    int blocksPerGrid=(n+BS-1)/BS;

    if(dlevel[tid]==cur_level && dunsafe_vertex[tid]!=-1){
        int* dminimum;
        cudaMalloc(&dminimum,sizeof(int));
        *dminimum=INT_MAX/2;

        int* dcurrent_cut_vertex;
        cudaMalloc(&dcurrent_cut_vertex,sizeof(int));
        *dcurrent_cut_vertex=dunsafe_vertex[tid];
        int* dflag;
        int* ddist;
        cudaMalloc((void**) &dflag,sizeof(int));
        cudaMalloc((void**) &ddist,sizeof(int)*n);
        for(int i=0;i<n;i++){
            ddist[i]=INT_MAX/2;
        }
        ddist[tid]=0;
        *dflag=1;
        while(*dflag){
            *dflag=0;
            find_minimum<<<1,n>>>(dlevel,dvertex_pointers,dedges,ddist,n,dflag,dcurrent_cut_vertex,dminimum,cur_level,dvisited);
            if (cudaSuccess != cudaDeviceSynchronize()) {
                return;
            }
        }
        printf("%d\n",*dminimum);

        for(int i=0;i<n;i++){
            ddist[i]=INT_MAX/2;
        }
        ddist[tid]=0;
        *dflag=1;
        while(*dflag){
            *dflag=0;
            set_bcc_id<<<1,n>>>(dlevel,dvertex_pointers,dedges,ddist,n,dflag,cur_level,dminimum,dcurrent_cut_vertex,dvisited,dbcc);
            if (cudaSuccess != cudaDeviceSynchronize()) {
                return;
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
    int unsafe_vertex[17];
    for(int i=0;i<n;i++){
        level[i]=INT_MAX/2;
        unsafe_vertex[i]=-1;
    }
    level[root]=0;
    int* dlevel;
    cudaMalloc((void**) &dlevel,sizeof(int)*n);

    int* dunsafe_vertex;
    cudaMalloc((void**) &dunsafe_vertex,sizeof(int)*n);

    int flag=1;
    int* dflag;
    cudaMalloc((void**) &dflag,sizeof(int));

    int max_level=0;
    while(flag){
        flag=0;
        cudaMemcpy(dlevel,level,sizeof(int)*n,cudaMemcpyHostToDevice);
        cudaMemcpy(dflag,&flag,sizeof(int),cudaMemcpyHostToDevice);
        bfs<<<1,n>>>(dflag,dlevel,dvertex_pointers,dedges,n);
        max_level++;
        cudaMemcpy(level,dlevel,sizeof(int)*n,cudaMemcpyDeviceToHost);
        cudaMemcpy(&flag,dflag,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(dunsafe_vertex,unsafe_vertex,sizeof(int)*n,cudaMemcpyHostToDevice);
    cudaMemcpy(dlevel,level,sizeof(int)*n,cudaMemcpyHostToDevice);
    find_cut_vertices<<<1,n>>>(dlevel,dvertex_pointers,dedges,dcut_vertex,n,root,dunsafe_vertex);
    cudaDeviceSynchronize();
    cudaMemcpy(&cut_vertex,dcut_vertex,sizeof(int)*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(unsafe_vertex,dunsafe_vertex,sizeof(int)*n,cudaMemcpyDeviceToHost);

    // for(int i=0;i<n;i++){
    //     if(unsafe_vertex[i]!=-1){
    //         cout<<i<<" "<<unsafe_vertex[i]<<endl;
    //     }
    // }

    int bcc[50];
    for(int i=0;i<n;i++){
        bcc[i]=-1;
    }
    int* dbcc;
    cudaMalloc(&dbcc,sizeof(int)*m);

    int visited[50]={0};
    int* dvisited;
    cudaMalloc(&dvisited,sizeof(int)*m);

    for(int i=max_level;i>=0;i--){
        cudaMemcpy(dvisited,visited,sizeof(int)*m,cudaMemcpyHostToDevice);
        cudaMemcpy(dbcc,bcc,sizeof(int)*m,cudaMemcpyHostToDevice);
        find_bcc<<<1,n>>>(dlevel,dvertex_pointers,dedges,dunsafe_vertex,i,n,dbcc,dvisited);
        cudaDeviceSynchronize();
        cudaMemcpy(bcc,dbcc,sizeof(int)*m,cudaMemcpyDeviceToHost);
        cudaMemcpy(visited,dvisited,sizeof(int)*m,cudaMemcpyDeviceToHost);
    }

    // for(int i=0;i<m;i++){
    //     cout<<i<<" "<<bcc[i]<<endl;
    // }
}