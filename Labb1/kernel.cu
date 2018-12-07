#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

#define TRUE 1
#define FALSE 0

#define SEQ_LENGHT  100000
#define BLOCK_SIZE  1024
#define USE_GPU     TRUE
#define USE_CPU     FALSE
#define PRINTARRAY  FALSE

__host__
bool isCorrect(int *array);

__host__
void printArray(int *array);

__host__
void generateArray(int *array);

__host__
void presentResult(int *array, long long elapsedTime);

#if USE_GPU
    __host__
    bool useGPU();

    __host__
    bool sortOnGPU(int *array);
#endif

#if USE_CPU
    __host__
    void useCPU();

    __host__
    void sortOnCPU(int *array);
#endif

// ***********************************
//          MAIN LOOP
// ***********************************


int main()
{
    srand((unsigned int)time(NULL));
    
    #if USE_CPU
        useCPU();
    #endif

    #if USE_GPU
        if(!useGPU())
            return 1;
    #endif

    return 0;
}

// ***********************************
//          HELPER FUNCTIONS
// ***********************************


__host__
bool isCorrect(int *array)
{
    for(int i = 0; i < SEQ_LENGHT-1; i++)
    {
        if(array[i] > array[i+1])
            return false;
    }
    return true;
}

__host__
void printArray(int *array)
{
    for(int i = 0; i < SEQ_LENGHT; i++)
    {
        std::cout << array[i] << std::endl;
    }
}

__host__
void generateArray(int *array)
{
    for(int i = 0; i < SEQ_LENGHT; i++)
    {
        array[i] = rand();
    }

    #if PRINTARRAY
        printArray(array);
        std::cout << "\n\n";
    #endif
}

__host__
void presentResult(int *array, long long elapsedTime)
{
    #if PRINTARRAY
        printArray(array);
        std::cout << "\n\n";
    #endif

    if (isCorrect(array))
        std::cout << "The array was correctly sorted!\n";
    else
        std::cout << "Not correct!\n";

    std::cout << "Time elapsed during sorting: "
        << elapsedTime
        << "ms\n";
}

#if USE_GPU
    __host__
    bool useGPU()
    {
        cudaError_t cudaStatus;
        int array[SEQ_LENGHT];

        generateArray(array);

        std::cout << "Starting GPU sorting method now.\n";
        auto timeStart = std::chrono::high_resolution_clock::now();

        if(!sortOnGPU(array))
        {
            std::cerr << "sortOnGPU unsuccessful, terminating.\n";
            if (cudaDeviceReset() != cudaSuccess)
                std::cerr << "cudaDeviceReset failed!\n";
            return false;
        }

        auto timeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "GPU sorting method finished!\n";

        presentResult(
            array,
            std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count()
        );

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
			std::cerr << "cudaDeviceReset failed!\n";
            return false;
        }
		return true;
    }
#endif

#if USE_CPU
    __host__
    void useCPU()
    {
        int array[SEQ_LENGHT];

        generateArray(array);

        std::cout << "Starting CPU sorting method now.\n";
        auto timeStart = std::chrono::high_resolution_clock::now();

        sortOnCPU(array);

        auto timeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "CPU sorting method finished!\n";

        presentResult(
            array,
            std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count()
        );
    }
#endif


// ***********************************
//          HOST METHOD ON CPU
// ***********************************

#if USE_CPU
    __host__ inline
    void hSwap(int* a, int* b)
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }

    __host__ inline
    void cpuSortEven(int *array)
    {
        for(int i = 0; i < SEQ_LENGHT - 1; i+=2)
        {
            if(array[i] > array[i+1])
                hSwap(array+i, array+i+1);
        }
    }

    __host__ inline
    void cpuSortOdd(int *array)
    {
        for(int i = 1; i < SEQ_LENGHT - 1; i+=2)
        {
            if(array[i] > array[i+1])
                hSwap(array+i, array+i+1);
        }
    }

    __host__
    void sortOnCPU(int *array)
    {
        for(int i = 0; i < SEQ_LENGHT / 2; i++)
        {
            cpuSortEven(array);
            cpuSortOdd(array);
        }
    }
#endif

// ***********************************
//          DEVICE METHOD ON GPU
// ***********************************

#if USE_GPU
    __device__
    void dSwap(int* a, int* b)
    {
        int temp = *a;
        *a = *b;
        *b = temp;
    }

    __global__
    void gpuSortEven(int *array)
    {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        if(i < SEQ_LENGHT - 1)
        {
            if(array[i] > array[i+1])
            {
                dSwap(array+i, array+i+1);
            }
        }
    }

    __global__
    void gpuSortOdd(int *array)
    {
        int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1;
        if(i < SEQ_LENGHT - 1)
        {
            if(array[i] > array[i+1])
            {
                dSwap(array+i, array+i+1);
            }
        }
    }


    __host__
    bool sortOnGPU(int *array)
    {
        cudaError_t cudaStatus;
        int *d_array = nullptr;

        cudaStatus = cudaMalloc((void**)&d_array, SEQ_LENGHT * sizeof(int));
        if(cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed!\n";
            goto Error;
        }

        cudaStatus = cudaMemcpy(d_array, array, SEQ_LENGHT * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy failed!\n";
            goto Error;
        }

        int nrOfBlocks = (SEQ_LENGHT / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for(int i = 0; i < SEQ_LENGHT / 2; i++)
        {
            gpuSortEven<<<nrOfBlocks, BLOCK_SIZE>>>(d_array);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                std::cerr << "gpuSortEven kernel call failed at iteration " << i << "!\n"
                    << cudaGetErrorString(cudaStatus) << std::endl;
                goto Error;
            }

            gpuSortOdd<<<nrOfBlocks, BLOCK_SIZE>>>(d_array);
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                std::cerr << "gpuSortOdd kernel call failed at iteration " << i << "!\n"
                    << cudaGetErrorString(cudaStatus) << std::endl;
                goto Error;
            }
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus <<" after launching addKernel!\n";
            goto Error;
        }

        cudaStatus = cudaMemcpy(array, d_array, SEQ_LENGHT * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "cudaMemcpy (cudaMemcpyDeviceToHost) failed!\n";
            goto Error;
        }

        return true;

        Error:
        cudaFree(d_array);
        return false;
    }
#endif