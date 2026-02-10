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
constexpr uint32_t ALIGN_BYTES = 32; 
constexpr uint32_t FLOATS_PER_ALIGN = ALIGN_BYTES / sizeof(float);  // 8 floats per 32 bytes
constexpr uint32_t UPMEMORY_SIZE = 23000;  // Default batch size for feature vectors

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
        this->numDenseCols =numDenseCols;
        this->nnz = nnz;
        
        this->alignedDenseCols = ((numDenseCols + FLOATS_PER_ALIGN - 1) / FLOATS_PER_ALIGN) * FLOATS_PER_ALIGN;
        this->batchSize = UPMEMORY_SIZE / numDenseCols;
        
        this->alignedPart = (numDenseCols / FLOATS_PER_ALIGN) * FLOATS_PER_ALIGN;
        this->remainder = numDenseCols - this->alignedPart;
        this->useAlignedCopy = (this->remainder == 0);
        
        uint32_t totalBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t rowsPerBlock = (numSparseRows + totalBlocks - 1) / totalBlocks;
        this->startRow = blockIdx * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock > numSparseRows ? numSparseRows : this->startRow + rowsPerBlock;
        
        rowPtrGm.SetGlobalBuffer((__gm__ uint32_t *)row_ptr, numSparseRows + 1);
        colIndGm.SetGlobalBuffer((__gm__ uint32_t *)col_ind, nnz);
        denseGm.SetGlobalBuffer((__gm__ float *)dense_matrix, numSparseCols * numDenseCols);
        outputGm.SetGlobalBuffer((__gm__ float *)output, numSparseRows * numDenseCols);
        
        uint32_t alignedBufferSize = alignedDenseCols * sizeof(float);
        uint32_t batchBufferSize = this->batchSize * alignedDenseCols * sizeof(float);
        pipe.InitBuffer(accumQueue, BUFFER_NUM, alignedBufferSize);
        pipe.InitBuffer(tempQueue, BUFFER_NUM, batchBufferSize);
    }
    
    // Process SpMM computation for assigned rows
    __aicore__ inline void Process()
    {
        for(uint32_t rowIdx = this->startRow; rowIdx < this->endRow; rowIdx++)
        {
            uint32_t rowStart = rowPtrGm.GetValue(rowIdx);
            uint32_t rowEnd = rowPtrGm.GetValue(rowIdx + 1);
            uint32_t rowNnz = rowEnd - rowStart;
            
            if (rowNnz == 0) {
                // Handle empty row - output zeros
                AscendC::LocalTensor<float> accum = accumQueue.AllocTensor<float>();
                AscendC::Duplicate<float>(accum, float(0.0f), alignedDenseCols);
                accumQueue.EnQue<float>(accum);
                CopyOut(rowIdx);
                continue;
            }
            
            AscendC::LocalTensor<float> accum = accumQueue.AllocTensor<float>();
            AscendC::Duplicate<float>(accum, float(0.0f), alignedDenseCols);
            
            uint32_t batchCount = (rowNnz + batchSize - 1) / batchSize;
            if (batchCount == 1) {
                CopyIn(rowStart, rowStart + rowNnz);
                Compute(accum, rowNnz);
            } else {
                uint32_t firstBatchEnd = rowStart + batchSize;
                CopyIn(rowStart, firstBatchEnd);
                for (uint32_t batchIdx = 1; batchIdx < batchCount; ++batchIdx) {
                    uint32_t batchStart = rowStart + batchIdx * batchSize;
                    uint32_t batchEnd = batchStart + batchSize > rowStart + rowNnz ? rowStart + rowNnz : batchStart + batchSize;
                    CopyIn(batchStart, batchEnd);
                    Compute(accum, batchSize);
                }
                
                uint32_t lastBatchNnz = (rowNnz % batchSize == 0) ? batchSize : (rowNnz % batchSize);
                Compute(accum, lastBatchNnz);
            }
            
            accumQueue.EnQue<float>(accum);
            CopyOut(rowIdx);
        }
    }

private:
    // Copy dense matrix rows to local buffer
    __aicore__ inline void CopyIn(uint32_t batchStart, uint32_t batchEnd){
        uint32_t batchNnz = batchEnd - batchStart;
        
        AscendC::LocalTensor<float> tempFeaturesBatch = tempQueue.AllocTensor<float>();
        AscendC::Duplicate<float>(tempFeaturesBatch, float(0.0f), batchNnz * alignedDenseCols);
        
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t idx = batchStart + i;
            uint32_t col = colIndGm.GetValue(idx);
            uint32_t offset = i * alignedDenseCols;
            AscendC::DataCopy(tempFeaturesBatch[offset], denseGm[col * numDenseCols], alignedPart);
        }
        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t idx = batchStart + i;
            uint32_t col = colIndGm.GetValue(idx);
            uint32_t offset = i * alignedDenseCols;
            for (uint32_t j = 0; j < remainder; ++j) {
                uint32_t colIdx = alignedPart + j;
                float val = denseGm.GetValue(col * numDenseCols + colIdx);
                tempFeaturesBatch.SetValue(offset + colIdx, val);
            }
        }
        
        tempQueue.EnQue<float>(tempFeaturesBatch);
    }

    // Accumulate feature vectors using sum reduction
    __aicore__ inline void Compute(AscendC::LocalTensor<float>& accum, uint32_t batchNnz){
        AscendC::LocalTensor<float> tempFeaturesBatch = tempQueue.DeQue<float>();

        for (uint32_t i = 0; i < batchNnz; ++i) {
            uint32_t offset = i * alignedDenseCols;
            AscendC::Add(accum, accum, tempFeaturesBatch[offset], alignedDenseCols);
        }
        
        tempQueue.FreeTensor(tempFeaturesBatch);
    }

    // Copy accumulated result to global memory
    __aicore__ inline void CopyOut(int32_t rowIdx){
        AscendC::LocalTensor<float> accum = accumQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[rowIdx * numDenseCols], accum, alignedPart);
        
        for (uint32_t j = 0; j < remainder; ++j) {
            uint32_t idx = alignedPart + j;
            float val = accum.GetValue(idx);
            outputGm.SetValue(rowIdx * numDenseCols + idx, val);
        }
        accumQueue.FreeTensor(accum);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> accumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tempQueue; 
    
    AscendC::GlobalTensor<uint32_t> rowPtrGm;
    AscendC::GlobalTensor<uint32_t> colIndGm;
    AscendC::GlobalTensor<float> denseGm; 
    AscendC::GlobalTensor<float> outputGm;
    uint32_t numSparseRows;
    uint32_t numSparseCols; 
    uint32_t numDenseCols;
    uint32_t alignedDenseCols; 
    uint32_t nnz;
    uint32_t batchSize;
    uint32_t alignedPart;      // Pre-computed aligned part
    uint32_t remainder;        // Pre-computed remainder
    bool useAlignedCopy;       // Whether to use aligned copy optimization
    uint32_t startRow, endRow; // Row range assigned to this block
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
