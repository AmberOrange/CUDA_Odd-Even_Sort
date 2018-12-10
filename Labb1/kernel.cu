#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <ctime>
#include <chrono>

#define TRUE 1
#define FALSE 0

#define MAX_TOTAL_ELEMENTS  100000
#define TOTAL_RUNS 10
#define BLOCK_SIZE  256
#define USE_GPU     TRUE
#define USE_CPU     TRUE
#define PRINTARRAY  FALSE

__host__
bool isCorrect(int *array, int totalElements);

__host__
void printArray(int *array, int totalElements);

__host__
void generateArray(int *array, int totalElements);

__host__
void presentResult(int *array, int totalElements, long long elapsedTime);

#if USE_GPU
	__host__
	bool doGPUTests();

    __host__
    bool useGPU(int totalElements, int totalThreads, long long* returnTime = nullptr);

    __host__
    bool sortOnGPU(int *array, int totalElements, int totalThreads);
#endif

#if USE_CPU
	__host__
	void doCPUTests();

    __host__
    void useCPU(int totalElements,long long* returnTime = nullptr);

    __host__
    void sortOnCPU(int *array, int totalElements);
#endif

// ***********************************
//          MAIN LOOP
// ***********************************


int main()
{
    srand((unsigned int)time(NULL));
    
    #if USE_CPU
		doCPUTests();
		std::cout << std::endl << std::endl;
    #endif

    #if USE_GPU
		if (!doGPUTests())
			return 1;
    #endif

    return 0;
}

// ***********************************
//          HELPER FUNCTIONS
// ***********************************


__host__
bool isCorrect(int *array, int totalElements)
{
    for(int i = 0; i < totalElements-1; i++)
    {
        if(array[i] > array[i+1])
            return false;
    }
    return true;
}

__host__
void printArray(int *array, int totalElements)
{
    for(int i = 0; i < totalElements; i++)
    {
        std::cout << array[i] << std::endl;
    }
}

__host__
void generateArray(int *array, int totalElements)
{
    for(int i = 0; i < totalElements; i++)
    {
        array[i] = rand();
    }

    #if PRINTARRAY
        printArray(array, totalElements);
        std::cout << "\n\n";
    #endif
}

__host__
void presentResult(int *array, int totalElements, long long elapsedTime)
{
    #if PRINTARRAY
        printArray(array, totalElements);
        std::cout << "\n\n";
    #endif

    if (isCorrect(array, totalElements))
        std::cout << "The array was correctly sorted!\n";
    else
        std::cout << "Not correct!\n";

    std::cout << "Time elapsed during sorting: "
        << elapsedTime
        << "ms\n";
}

#if USE_GPU

//#define GPU_TEST_PRINT_AVERAGE(totalElements, totalThreads) \
//	elapsedTime = 0; \
//	for (int i = 0; i < TOTAL_RUNS; i++) \
//	{ \
//		if (!useGPU(totalElements, totalThreads, &returnTime)) { \
//			std::cerr << "Total Elements: " << totalElements << ", Total Threads: " << totalThreads; \
//			return false; \
//		} \
//		elapsedTime += returnTime; \
//	} \
//	std::cout << elapsedTime / TOTAL_RUNS << std::endl; 

__host__
bool doGPUTests()
{
	long long returnTime;
	long long elapsedTime;

	for (int i = 100; i <= MAX_TOTAL_ELEMENTS; i *= 10)
	{
		std::cout << i << " elements:\n";
		for (int j = 2; j <= 16; j *= 2)
		{
			elapsedTime = 0; 
			for (int k = 0; k < TOTAL_RUNS; k++) 
			{ 
				if (!useGPU(i, (i + j - 1) / j, &returnTime))
				{
						std::cerr << "Total Elements: " << i << ", Total Threads: " << (i + j - 1) / j << " (" << j << ")\n";
						return false; 
				} 
				elapsedTime += returnTime; 
			} 
			std::cout << elapsedTime / TOTAL_RUNS << std::endl;

			//GPU_TEST_PRINT_AVERAGE(i, (i+j-1)/j);	// Round up to whole threads
		}
	}


	return true;
}


    __host__
    bool useGPU(int totalElements, int totalThreads, long long* returnTime)
    {
		if (totalElements > MAX_TOTAL_ELEMENTS)
		{
			std::cerr << "totalElements are exceeding the maximum!";
			return false;
		}

        cudaError_t cudaStatus;
        int array[MAX_TOTAL_ELEMENTS];

        generateArray(array, totalElements);

		if(!returnTime)
			std::cout << "Starting GPU sorting method now.\n";
        auto timeStart = std::chrono::high_resolution_clock::now();

        if(!sortOnGPU(array, totalElements, totalThreads))
        {
            std::cerr << "sortOnGPU unsuccessful, terminating.\n";
            if (cudaDeviceReset() != cudaSuccess)
                std::cerr << "cudaDeviceReset failed!\n";
            return false;
        }

        auto timeEnd = std::chrono::high_resolution_clock::now();
		if (!returnTime)
			std::cout << "GPU sorting method finished!\n";

		if (!returnTime)
			presentResult(
				array,
				totalElements,
				std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count()
			);
		else
		{
			if (!isCorrect(array, totalElements))
			{
				std::cerr << "Not Correct!\n";
				return false;
			}
		}

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
			std::cerr << "cudaDeviceReset failed!\n";
            return false;
        }

		if (returnTime)
			*returnTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();

		return true;
    }
#endif

#if USE_CPU
	__host__
	void doCPUTests()
	{
		long long returnTime;
		long long elapsedTime;

		for (int i = 100; i <= MAX_TOTAL_ELEMENTS; i *= 10)
		{
			std::cout << i << " elements:\n";
			elapsedTime = 0;
			for (int k = 0; k < TOTAL_RUNS; k++)
			{
				useCPU(i, &returnTime);
				elapsedTime += returnTime;
			}
			std::cout << elapsedTime / TOTAL_RUNS << std::endl;
		}
	}

    __host__
    void useCPU(int totalElements, long long* returnTime)
    {
		if (totalElements > MAX_TOTAL_ELEMENTS)
		{
			std::cerr << "totalElements are exceeding the maximum!";
			return;
		}
        int array[MAX_TOTAL_ELEMENTS];

        generateArray(array, totalElements);

		if (!returnTime)
			std::cout << "Starting CPU sorting method now.\n";
        auto timeStart = std::chrono::high_resolution_clock::now();

        sortOnCPU(array, totalElements);

        auto timeEnd = std::chrono::high_resolution_clock::now();
		if (!returnTime)
			std::cout << "CPU sorting method finished!\n";

		if (!returnTime)
			presentResult(
				array,
				totalElements,
				std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count()
			);
		else
		{
			*returnTime = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
			if (!isCorrect(array, totalElements))
				std::cerr << "Not Correct!\n";
		}
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
    void cpuSortEven(int *array, int totalElements)
    {
        for(int i = 0; i < totalElements - 1; i+=2)
        {
            if(array[i] > array[i+1])
                hSwap(array+i, array+i+1);
        }
    }

    __host__ inline
    void cpuSortOdd(int *array, int totalElements)
    {
        for(int i = 1; i < totalElements - 1; i+=2)
        {
            if(array[i] > array[i+1])
                hSwap(array+i, array+i+1);
        }
    }

    __host__
    void sortOnCPU(int *array, int totalElements)
    {
        for(int i = 0; i < totalElements / 2; i++)
        {
            cpuSortEven(array, totalElements);
            cpuSortOdd(array, totalElements);
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
    void gpuSortEven(int *array, int totalElements, int totalThreads)
    {
		int id = (blockIdx.x * blockDim.x + threadIdx.x);
		
		if (id < totalThreads)
		{
			for (
				int i = id * 2;
				i < totalElements - 1;
				i += totalThreads * 2)
			{
				if (array[i] > array[i + 1])
				{
					dSwap(array+i, array+i+1);
				}
			}
		}
    }

    __global__
    void gpuSortOdd(int *array, int totalElements, int totalThreads)
    {
		int id = (blockIdx.x * blockDim.x + threadIdx.x);
        
		if (id < totalThreads)
		{
			for (
				int i = id * 2 + 1;
				i < totalElements - 1;
				i += totalThreads * 2)
			{
				if (array[i] > array[i + 1])
				{
					dSwap(array + i, array + i + 1);
				}
			}
		}
    }

	#define CUDA_CHECK_ERROR(result) \
		cudaStatus = result; \
		if(cudaStatus != cudaSuccess) { \
			std::cerr << __FILE__ << ": Error at line " << __LINE__ << ":\n" << cudaGetErrorString(cudaStatus) << std::endl; \
			goto Error; \
		}

    __host__
    bool sortOnGPU(int *array, int totalElements, int totalThreads)
    {
        cudaError_t cudaStatus;
        int *d_array = nullptr;

		CUDA_CHECK_ERROR(
			cudaMalloc((void**)&d_array, totalElements * sizeof(int)));

		CUDA_CHECK_ERROR(
			cudaMemcpy(d_array, array, totalElements * sizeof(int), cudaMemcpyHostToDevice));

		int nrOfBlocks = totalThreads / BLOCK_SIZE + 1;

        for(int i = 0; i < totalElements / 2; i++)
        {
            gpuSortEven<<<nrOfBlocks, BLOCK_SIZE>>>(d_array, totalElements, totalThreads);
			CUDA_CHECK_ERROR(
				cudaGetLastError());

            gpuSortOdd<<<nrOfBlocks, BLOCK_SIZE>>>(d_array, totalElements, totalThreads);
			CUDA_CHECK_ERROR(
				cudaGetLastError());
        }
		CUDA_CHECK_ERROR(
			cudaDeviceSynchronize());

		CUDA_CHECK_ERROR(
			cudaMemcpy(array, d_array, totalElements * sizeof(int), cudaMemcpyDeviceToHost));

        return true;

        Error:
        cudaFree(d_array);
        return false;
    }
#endif