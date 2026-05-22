#ifndef SPMM_SUM_TILING_H
#define SPMM_SUM_TILING_H
#include <cstdint> 

// Tiling data structure for SpMM sum kernel
struct SpmmSumTilingData {
    uint32_t numSparseRows;
    uint32_t numSparseCols;
    uint32_t numDenseCols;
    uint32_t nnz;
};
#endif

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ACCUM_BUFFER_NUM = 4;
constexpr uint32_t ALIGN_BYTES = 32; 
constexpr uint32_t FLOATS_PER_ALIGN = ALIGN_BYTES / sizeof(float);  // 8 floats per 32 bytes
constexpr uint32_t UINT32_PER_ALIGN = ALIGN_BYTES / sizeof(uint32_t);
// constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024; 
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024; 
constexpr uint32_t UB_RESERVED = 2048; 
constexpr uint32_t UB_AVAILABLE = UB_TOTAL_SIZE - UB_RESERVED;

// AscendC kernel for SpMM with sum reduction
class KernelSpmmSum {
public:
    __aicore__ inline KernelSpmmSum() {}
    
    // Initialize kernel with CSR matrix and dense matrix pointers
    __aicore__ inline void Init(GM_ADDR row_ptr, GM_ADDR col_ind,
                                GM_ADDR dense_matrix, GM_ADDR output,
                                uint32_t numSparseRows, uint32_t numSparseCols, uint32_t numDenseCols, 
                                uint32_t nnz)
    {
        this->numSparseRows = numSparseRows;
        this->numSparseCols = numSparseCols;
        this->numDenseCols = numDenseCols;
        this->nnz = nnz;
        // 划分每个AIV执行的行范围
        uint32_t totalBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerBlock = (numSparseRows + totalBlocks - 1) / totalBlocks;
        this->startRow = blockIdx * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock > numSparseRows ? numSparseRows : this->startRow + rowsPerBlock;
        
        uint32_t localRows = this->endRow > this->startRow ? (this->endRow - this->startRow) : 0;
        if (localRows > 0) {
            this->alignedStartRow = (this->startRow / UINT32_PER_ALIGN) * UINT32_PER_ALIGN;
            uint32_t alignedEndRow = ((this->endRow + 1 + UINT32_PER_ALIGN - 1) / UINT32_PER_ALIGN) * UINT32_PER_ALIGN;
            this->rowPtrAlignedElements = alignedEndRow - this->alignedStartRow;
            this->rowPtrOffset = this->startRow - this->alignedStartRow;
        } else {
            this->rowPtrAlignedElements = UINT32_PER_ALIGN;
            this->rowPtrOffset = 0;
            this->alignedStartRow = 0;
        }
        uint32_t rowPtrBytes = this->rowPtrAlignedElements * sizeof(uint32_t);
        rowPtrGm.SetGlobalBuffer((__gm__ uint32_t *)row_ptr, numSparseRows + 1);
        denseGm.SetGlobalBuffer((__gm__ float *)dense_matrix, numSparseCols * numDenseCols);
        outputGm.SetGlobalBuffer((__gm__ float *)output, numSparseRows * numDenseCols);

        colIndGm.SetGlobalBuffer((__gm__ uint32_t *)col_ind, nnz);
        // 固定UB开销，计算剩余UB空间
        uint32_t rowBytes = numDenseCols * sizeof(float);
        uint32_t accumBytes = ACCUM_BUFFER_NUM * rowBytes;
        uint32_t fixedCost = rowBytes + accumBytes + rowPtrBytes;
        uint32_t remainingUb = UB_AVAILABLE > fixedCost ? (UB_AVAILABLE - fixedCost) : 0;

        uint32_t itemCost = BUFFER_NUM * rowBytes + 2 * sizeof(uint32_t);
        uint32_t paddingCost = 4 * FLOATS_PER_ALIGN * sizeof(uint32_t);
        uint32_t usableUb = remainingUb > paddingCost ? (remainingUb - paddingCost) : 0;
        this->batchSize = usableUb / itemCost;
        if (this->batchSize == 0) this->batchSize = 1;

        uint32_t batchBufferSize = this->batchSize * rowBytes;
        uint32_t sparseBatchBytes = (this->batchSize + 2 * FLOATS_PER_ALIGN) * sizeof(float);
        // if(blockIdx == 0){
        //     AscendC::printf("batchSize:%d\n",this->batchSize);
        // }
        pipe.InitBuffer(accumQueue, ACCUM_BUFFER_NUM, rowBytes);
        pipe.InitBuffer(tempQueue, BUFFER_NUM, batchBufferSize);
        pipe.InitBuffer(colIndQue, BUFFER_NUM, sparseBatchBytes);
        pipe.InitBuffer(rowPtrBuf, rowPtrBytes);
    }
    
    // Process SpMM computation for assigned rows
    __aicore__ inline void Process()
    {
        if (this->endRow <= this->startRow) return;
        AscendC::LocalTensor<uint32_t> localRowPtr = rowPtrBuf.Get<uint32_t>();
        AscendC::DataCopy(localRowPtr, rowPtrGm[this->alignedStartRow], this->rowPtrAlignedElements);
        for(uint32_t rowIdx = this->startRow; rowIdx < this->endRow; ++rowIdx)
        {
            AscendC::LocalTensor<float> accumBlock = accumQueue.AllocTensor<float>();
            AscendC::Duplicate<float>(accumBlock, float(0.0f), numDenseCols);

            uint32_t localIdx = rowIdx - this->startRow + this->rowPtrOffset;
            uint32_t rowStart = localRowPtr.GetValue(localIdx);
            uint32_t rowEnd = localRowPtr.GetValue(localIdx + 1);
            uint32_t rowNnz = rowEnd - rowStart;
            if (rowNnz > 0) 
            {
                uint32_t batchCount = (rowNnz + batchSize - 1) / batchSize;
                for (uint32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
                    uint32_t batchStart = rowStart + batchIdx * batchSize;
                    uint32_t curBatchNnz = (batchStart + batchSize > rowStart + rowNnz) ? (rowStart + rowNnz - batchStart) : batchSize;
                    
                    uint32_t alignedBatchStart = (batchStart / FLOATS_PER_ALIGN) * FLOATS_PER_ALIGN;
                    uint32_t alignedBatchEnd = ((batchStart + curBatchNnz + FLOATS_PER_ALIGN - 1) / FLOATS_PER_ALIGN) * FLOATS_PER_ALIGN;
                    uint32_t alignedBatchTotal = alignedBatchEnd - alignedBatchStart;
                    uint32_t localOffset = batchStart - alignedBatchStart;

                    AscendC::LocalTensor<uint32_t> localColInd = colIndQue.AllocTensor<uint32_t>();
                    AscendC::DataCopy(localColInd, colIndGm[alignedBatchStart], alignedBatchTotal);
                    colIndQue.EnQue(localColInd);
                    AscendC::LocalTensor<uint32_t> readyColInd = colIndQue.DeQue<uint32_t>();
                    
                    CopyIn(curBatchNnz, localOffset, readyColInd);
                    Compute(accumBlock, curBatchNnz);
                    colIndQue.FreeTensor(readyColInd);
                }
            }
            accumQueue.EnQue<float>(accumBlock);
            CopyOut(rowIdx); 
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchNnz, uint32_t localOffset, AscendC::LocalTensor<uint32_t>& readyColInd) {
        AscendC::LocalTensor<float> tempFeaturesBatch = tempQueue.AllocTensor<float>();
        
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t col = readyColInd.GetValue(localOffset + i); 
            uint32_t offset = i * numDenseCols;
            AscendC::DataCopy(tempFeaturesBatch[offset], denseGm[col * numDenseCols], numDenseCols);
        }
        
        tempQueue.EnQue<float>(tempFeaturesBatch);
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<float>& accumBlock, uint32_t batchNnz)
    {
        AscendC::LocalTensor<float> tempFeaturesBatch = tempQueue.DeQue<float>();

        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t offset = i * numDenseCols;
            AscendC::Add(accumBlock, accumBlock, tempFeaturesBatch[offset], numDenseCols);
        }

        tempQueue.FreeTensor(tempFeaturesBatch);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx){
        AscendC::LocalTensor<float> accumBlock = accumQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[rowIdx * numDenseCols], accumBlock, numDenseCols);
        accumQueue.FreeTensor(accumBlock);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, ACCUM_BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tempQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> colIndQue;
    AscendC::TBuf<AscendC::TPosition::VECIN> rowPtrBuf;
    
    AscendC::GlobalTensor<uint32_t> rowPtrGm;
    AscendC::GlobalTensor<uint32_t> colIndGm;
    AscendC::GlobalTensor<float> denseGm; 
    AscendC::GlobalTensor<float> outputGm;
    uint32_t numSparseRows;
    uint32_t numSparseCols; 
    uint32_t numDenseCols;
    uint32_t nnz;
    uint32_t batchSize;
    uint32_t startRow, endRow;
    uint32_t alignedStartRow;
    uint32_t rowPtrOffset;
    uint32_t rowPtrAlignedElements;
};

// Kernel entry point for SpMM sum operation
extern "C" __global__ __aicore__ void spmm_sum(GM_ADDR row_ptr, GM_ADDR col_ind,
                                                    GM_ADDR dense_matrix,
                                                    GM_ADDR output, GM_ADDR tiling_ptr)
{

    AscendC::GlobalTensor<uint32_t> tilingGm;
    tilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling_ptr, 4);
    
    uint32_t numSparseRows = tilingGm.GetValue(0);
    uint32_t numSparseCols = tilingGm.GetValue(1);
    uint32_t numDenseCols = tilingGm.GetValue(2);
    uint32_t nnz = tilingGm.GetValue(3);
    
    KernelSpmmSum op;
    op.Init(row_ptr, col_ind, dense_matrix, output,
            numSparseRows, numSparseCols, numDenseCols,
            nnz);
    op.Process();
}