#include "evaluator.h"
#include "calc_indexes.h"
#include "calc_index_and_fetch_inplace.h"
#include "calc_index_and_fetch_inplace_fp16.h"

#include <library/cpp/float16/float16.h>
#include <library/cpp/sse/sse.h>

#include <util/generic/algorithm.h>
#include <util/stream/format.h>
#include <util/system/compiler.h>

#include <cstring>

namespace NCB::NModelEvaluation::AVX {

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValues(const size_t docCountInBlock, const double* __restrict treeLeafPtr, const TIndexType* __restrict indexesPtr, double* __restrict writePtr) {
        Y_PREFETCH_READ(treeLeafPtr, 3);
        Y_PREFETCH_READ(treeLeafPtr + 128, 3);
        const auto docCountInBlock4 = (docCountInBlock | 0x3) ^ 0x3;
        for (size_t docId = 0; docId < docCountInBlock4; docId += 4) {
            writePtr[0] += treeLeafPtr[indexesPtr[0]];
            writePtr[1] += treeLeafPtr[indexesPtr[1]];
            writePtr[2] += treeLeafPtr[indexesPtr[2]];
            writePtr[3] += treeLeafPtr[indexesPtr[3]];
            writePtr += 4;
            indexesPtr += 4;
        }
        for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
            *writePtr += treeLeafPtr[*indexesPtr];
            ++writePtr;
            ++indexesPtr;
        }
    }

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValuesFP16(const size_t docCountInBlock, const double* __restrict treeLeafPtr, const TIndexType* __restrict indexesPtr, float* __restrict writePtr) {
        Y_PREFETCH_READ(treeLeafPtr, 3);
        Y_PREFETCH_READ(treeLeafPtr + 128, 3);
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            *writePtr += TFloat16(treeLeafPtr[*indexesPtr]).AsFloat();
            ++writePtr;
            ++indexesPtr;
        }
    }

    template <int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE_SSE3(const double* __restrict treeLeafPtr, const ui8* __restrict indexesPtr, __m128d* __restrict writePtr) {
        _mm_prefetch((const char*)(treeLeafPtr + 64), _MM_HINT_T2);

        NFORCE_UNROLL
        for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
    #define GATHER_LEAFS(subBlock) const __m128d additions##subBlock = _mm_set_pd(treeLeafPtr[indexesPtr[subBlock * 2 + 1]], treeLeafPtr[indexesPtr[subBlock * 2 + 0]]);
    #define ADD_LEAFS(subBlock) writePtr[subBlock] = _mm_add_pd(writePtr[subBlock], additions##subBlock);

            GATHER_LEAFS(0);
            GATHER_LEAFS(1);
            GATHER_LEAFS(2);
            GATHER_LEAFS(3);
            ADD_LEAFS(0);
            ADD_LEAFS(1);
            ADD_LEAFS(2);
            ADD_LEAFS(3);

            GATHER_LEAFS(4);
            GATHER_LEAFS(5);
            GATHER_LEAFS(6);
            GATHER_LEAFS(7);
            ADD_LEAFS(4);
            ADD_LEAFS(5);
            ADD_LEAFS(6);
            ADD_LEAFS(7);
            writePtr += 8;
            indexesPtr += 16;
        }
    #undef GATHER_LEAFS
    #undef ADD_LEAFS
    }

    template <int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE_AVX2(const double* __restrict treeLeafPtr, const ui32* __restrict indexesPtr, __m256d* __restrict writePtr) {
        _mm_prefetch((const char*)(treeLeafPtr + 64), _MM_HINT_T2);

        NFORCE_UNROLL
        for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
    #define GATHER_LEAFS(subBlock)\
        ui32 indexes##subBlock = indexesPtr[subBlock];\
        const __m256d additions##subBlock = _mm256_set_pd(treeLeafPtr[(indexes##subBlock >> 3 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 2 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 1 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock) & 0xff]);
    #define ADD_LEAFS(subBlock) writePtr[subBlock] = _mm256_add_pd(writePtr[subBlock], additions##subBlock);

            GATHER_LEAFS(0);
            GATHER_LEAFS(1);
            GATHER_LEAFS(2);
            GATHER_LEAFS(3);
            ADD_LEAFS(0);
            ADD_LEAFS(1);
            ADD_LEAFS(2);
            ADD_LEAFS(3);

            GATHER_LEAFS(4);
            GATHER_LEAFS(5);
            GATHER_LEAFS(6);
            GATHER_LEAFS(7);
            ADD_LEAFS(4);
            ADD_LEAFS(5);
            ADD_LEAFS(6);
            ADD_LEAFS(7);
            writePtr += 8;
            indexesPtr += 8;
        }
    #undef GATHER_LEAFS
    #undef ADD_LEAFS
    }

    template <int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE_AVX512(const double* __restrict treeLeafPtr, const ui64* __restrict indexesPtr, __m512d* __restrict writePtr) {
        _mm_prefetch((const char*)(treeLeafPtr + 64), _MM_HINT_T2);

        NFORCE_UNROLL
        for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
    #define GATHER_LEAFS(subBlock)\
        ui64 indexes##subBlock = indexesPtr[subBlock];\
        const __m512d additions##subBlock = _mm512_set_pd(treeLeafPtr[(indexes##subBlock >> 7 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 6 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 5 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 4 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 3 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 2 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock >> 1 * 8) & 0xff],\
                                                          treeLeafPtr[(indexes##subBlock)          & 0xff]);
    #define ADD_LEAFS(subBlock) writePtr[subBlock] = _mm512_add_pd(writePtr[subBlock], additions##subBlock);

            GATHER_LEAFS(0);
            GATHER_LEAFS(1);
            GATHER_LEAFS(2);
            GATHER_LEAFS(3);
            ADD_LEAFS(0);
            ADD_LEAFS(1);
            ADD_LEAFS(2);
            ADD_LEAFS(3);

            GATHER_LEAFS(4);
            GATHER_LEAFS(5);
            GATHER_LEAFS(6);
            GATHER_LEAFS(7);
            ADD_LEAFS(4);
            ADD_LEAFS(5);
            ADD_LEAFS(6);
            ADD_LEAFS(7);
            writePtr += 8;
            indexesPtr += 8;
        }
    #undef GATHER_LEAFS
    #undef ADD_LEAFS
    }

    template <EEvaluationMethod EvaluationMethod, int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE(const double* __restrict treeLeafPtr, const ui8* __restrict indexesPtr, double* __restrict writePtr) {
        switch (EvaluationMethod) {
        case EEvaluationMethod::SSE3_BS128:
            GatherAddLeafSSE_SSE3<SSEBlockCount>(treeLeafPtr, indexesPtr, (__m128d*)writePtr);
            return;
        case EEvaluationMethod::AVX2_BS256:
        case EEvaluationMethod::AVX2_BS128:
        case EEvaluationMethod::AVX2_BS64:
            GatherAddLeafSSE_AVX2<SSEBlockCount>(treeLeafPtr, (ui32*)indexesPtr, (__m256d*)writePtr);
            return;
        case EEvaluationMethod::AVX512_BS512:
        case EEvaluationMethod::AVX512_BS256:
        case EEvaluationMethod::AVX512_BS128:
        case EEvaluationMethod::AVX512_BS64:
            GatherAddLeafSSE_AVX512<SSEBlockCount>(treeLeafPtr, (ui64*)indexesPtr, (__m512d*)writePtr);
            return;
        case EEvaluationMethod::NON_SSE:
        case EEvaluationMethod::AVX512_BS512_SHUFFLE:
        case EEvaluationMethod::AVX512_BS256_SHUFFLE:
        case EEvaluationMethod::AVX512_BS128_SHUFFLE:
        case EEvaluationMethod::AVX2_BS256_GATHER:
        case EEvaluationMethod::AVX2_BS128_GATHER:
        case EEvaluationMethod::AVX512_BS512_GATHER:
        case EEvaluationMethod::AVX512_BS256_GATHER:
        case EEvaluationMethod::AVX512_BS128_GATHER:
        case EEvaluationMethod::NON_SSE_FP16:
        case EEvaluationMethod::AVX512_BS512_FP16:
        case EEvaluationMethod::AVX512_BS256_FP16:
        case EEvaluationMethod::AVX512_BS128_FP16:
        case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
        case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
        case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
            CB_ENSURE(false);
        }
    }

    template <EEvaluationMethod EvaluationMethod, int SSEBlockCount>
    Y_FORCE_INLINE void CalculateLeafValues4(
        const size_t docCountInBlock,
        const double* __restrict treeLeafPtr0,
        const double* __restrict treeLeafPtr1,
        const double* __restrict treeLeafPtr2,
        const double* __restrict treeLeafPtr3,
        const ui8* __restrict indexesPtr0,
        const ui8* __restrict indexesPtr1,
        const ui8* __restrict indexesPtr2,
        const ui8* __restrict indexesPtr3,
        double* __restrict writePtr)
    {
        if (SSEBlockCount > 0) {
            _mm_prefetch((const char*)(writePtr), _MM_HINT_T2);
            GatherAddLeafSSE<EvaluationMethod, SSEBlockCount>(treeLeafPtr0, indexesPtr0, writePtr);
            GatherAddLeafSSE<EvaluationMethod, SSEBlockCount>(treeLeafPtr1, indexesPtr1, writePtr);
            GatherAddLeafSSE<EvaluationMethod, SSEBlockCount>(treeLeafPtr2, indexesPtr2, writePtr);
            GatherAddLeafSSE<EvaluationMethod, SSEBlockCount>(treeLeafPtr3, indexesPtr3, writePtr);
        }
        if (SSEBlockCount != 8) {
            indexesPtr0 += IConsts<EvaluationMethod>::SUBBLOCK_SIZE * SSEBlockCount;
            indexesPtr1 += IConsts<EvaluationMethod>::SUBBLOCK_SIZE * SSEBlockCount;
            indexesPtr2 += IConsts<EvaluationMethod>::SUBBLOCK_SIZE * SSEBlockCount;
            indexesPtr3 += IConsts<EvaluationMethod>::SUBBLOCK_SIZE * SSEBlockCount;
            writePtr += IConsts<EvaluationMethod>::SUBBLOCK_SIZE * SSEBlockCount;
            for (size_t docId = SSEBlockCount * IConsts<EvaluationMethod>::SUBBLOCK_SIZE; docId < docCountInBlock; ++docId) {
                *writePtr = *writePtr + treeLeafPtr0[*indexesPtr0] + treeLeafPtr1[*indexesPtr1] + treeLeafPtr2[*indexesPtr2] + treeLeafPtr3[*indexesPtr3];
                ++writePtr;
                ++indexesPtr0;
                ++indexesPtr1;
                ++indexesPtr2;
                ++indexesPtr3;
            }
        }
    }




    template <int SSEBlockCount>
    Y_FORCE_INLINE static void GatherAddLeafSSE_AVX512_FP16(const ui16* __restrict treeLeafPtr, const ui64* __restrict indexesPtr, __m512* __restrict writePtr) {
        _mm_prefetch((const char*)(treeLeafPtr + 64), _MM_HINT_T2);

        NFORCE_UNROLL
        for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
    #define GATHER_LEAFS(subBlock)\
        ui64 indexes0##subBlock = indexesPtr[subBlock];\
        ui64 indexes1##subBlock = indexesPtr[subBlock + 1];\
        const __m256i additions##subBlock = _mm256_set_epi16(treeLeafPtr[(indexes1##subBlock >> 7 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 6 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 5 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 4 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 3 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 2 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock >> 1 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes1##subBlock)          & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 7 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 6 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 5 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 4 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 3 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 2 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock >> 1 * 8) & 0xff],\
                                                             treeLeafPtr[(indexes0##subBlock)          & 0xff]);
    #define ADD_LEAFS(subBlock) writePtr[subBlock / 2] = _mm512_add_ps(writePtr[subBlock / 2], _mm512_cvtph_ps(additions##subBlock));

            GATHER_LEAFS(0);
            GATHER_LEAFS(2);
            ADD_LEAFS(0);
            ADD_LEAFS(2);

            GATHER_LEAFS(4);
            GATHER_LEAFS(6);
            ADD_LEAFS(4);
            ADD_LEAFS(6);
            writePtr += 4;
            indexesPtr += 8;
        }
    #undef GATHER_LEAFS
    #undef ADD_LEAFS
    }


    template <int SSEBlockCount>
    Y_FORCE_INLINE void CalculateLeafValues_AVX512_FP16(
        const size_t docCountInBlock,
        const ui16* __restrict treeLeafPtr0,
        const ui8* __restrict indexesPtr0,
        float* __restrict writePtr)
    {
        if (SSEBlockCount > 0) {
            _mm_prefetch((const char*)(writePtr), _MM_HINT_T2);
            GatherAddLeafSSE_AVX512_FP16<SSEBlockCount>(treeLeafPtr0, (const ui64*)indexesPtr0, (__m512*)writePtr);
        }
        if (SSEBlockCount != 8) {
            indexesPtr0 += IConsts<EEvaluationMethod::AVX512_BS512_FP16>::SUBBLOCK_SIZE * SSEBlockCount;
            writePtr += IConsts<EEvaluationMethod::AVX512_BS512_FP16>::SUBBLOCK_SIZE * SSEBlockCount;
            for (size_t docId = SSEBlockCount * IConsts<EEvaluationMethod::AVX512_BS512_FP16>::SUBBLOCK_SIZE; docId < docCountInBlock; ++docId) {
                *writePtr = *writePtr + TFloat16::Load(treeLeafPtr0[*indexesPtr0]).AsFloat();
                ++writePtr;
                ++indexesPtr0;
            }
        }
    }


    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValuesMulti(const size_t docCountInBlock, const double* __restrict leafPtr, const TIndexType* __restrict indexesVec, const int approxDimension, double* __restrict writePtr) {
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            auto leafValuePtr = leafPtr + indexesVec[docId] * approxDimension;
            for (int classId = 0; classId < approxDimension; ++classId) {
                writePtr[classId] += leafValuePtr[classId];
            }
            writePtr += approxDimension;
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool IsSingleClassModel, bool NeedXorMask, int SSEBlockCount, bool CalcLeafIndexesOnly = false>
    Y_FORCE_INLINE void CalcTreesBlockedImpl(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const ui8* __restrict binFeatures,
        const size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVecUI32,
        size_t treeStart,
        const size_t treeEnd,
        double* __restrict resultsPtr) {
        const TRepackedBin* treeSplitsCurPtr =
            trees.GetRepackedBins().data() + trees.GetModelTreeData()->GetTreeStartOffsets()[treeStart];

        ui8* __restrict indexesVec = (ui8*)indexesVecUI32;
        const auto treeLeafPtr = trees.GetModelTreeData()->GetLeafValues().data();
        auto firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
        bool allTreesAreShallow = AllOf(
            trees.GetModelTreeData()->GetTreeSizes().begin() + treeStart,
            trees.GetModelTreeData()->GetTreeSizes().begin() + treeEnd,
            [](int depth) { return depth <= 8; }
        );
        CB_ENSURE(IsSingleClassModel && !CalcLeafIndexesOnly && allTreesAreShallow, "SSE may not work in this cases");
        if constexpr (EvaluationMethod == EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16 ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16 ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16 ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS512_FP16 ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS256_FP16 ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS128_FP16 ||
                      EvaluationMethod == EEvaluationMethod::NON_SSE_FP16) {
            alignas(64) float res[IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE] = {};
            if (EvaluationMethod == EEvaluationMethod::NON_SSE_FP16) {
                for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
                    auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
                    memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
                    CalcIndexesBasic<EvaluationMethod, NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr,
                                                    curTreeSize);
                    CalculateLeafValuesFP16(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                            indexesVecUI32, res);
                    treeSplitsCurPtr += curTreeSize;
                }
            } else if (EvaluationMethod == EEvaluationMethod::AVX512_BS512_FP16 ||
                       EvaluationMethod == EEvaluationMethod::AVX512_BS256_FP16 ||
                       EvaluationMethod == EEvaluationMethod::AVX512_BS128_FP16) {
                const auto treeLeafPtr16 = applyData.TreeLeafValuesFP16.data();
                const auto firstLeafOffsets16 = applyData.TreeFirstLeafOffsetsFP16;
                for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
                    auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
                    memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
                    CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr,
                                                           curTreeSize);
                    CalculateLeafValues_AVX512_FP16<SSEBlockCount>(docCountInBlock, treeLeafPtr16 + firstLeafOffsets16[treeId],
                                                    indexesVec, res);
                    treeSplitsCurPtr += curTreeSize;
                }
            } else {
                const auto treeLeafPtr16 = applyData.TreeLeafValuesFP16.data();
                const auto firstLeafOffsets16 = applyData.TreeFirstLeafOffsetsFP16;
                for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
                    CalcIndexesAndFetchFP16<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock,
                                        treeSplitsCurPtr, treeLeafPtr16 + firstLeafOffsets16[treeId], res, trees.GetModelTreeData()->GetTreeSizes()[treeId]);
                    treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId];
                }
            }
            if (EvaluationMethod == EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16 ||
                EvaluationMethod == EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16 ||
                EvaluationMethod == EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16) {
                size_t docId = 0;
                for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
                    for (size_t i = 0; i < 32; ++i, ++docId) {
                        if (i % 2 == 0) {
                            resultsPtr[docId] = res[docId - i / 2];
                        } else {
                            resultsPtr[docId] = res[docId - i / 2 - 1 + 32];
                        }
                    }
                    for (size_t i = 0; i < 32; ++i, ++docId) {
                        if (i % 2 == 0) {
                            resultsPtr[docId] = res[docId - i / 2 - 16];
                        } else {
                            resultsPtr[docId] = res[docId - i / 2 - 1 + 16];
                        }
                    }
                }
                for (; docId < docCountInBlock; ++docId) {
                    resultsPtr[docId] = res[docId];
                }
            } else {
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    resultsPtr[docId] = res[docId];
                }
            }
            return;
        }
        if constexpr (EvaluationMethod == EEvaluationMethod::AVX512_BS512_SHUFFLE ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS256_SHUFFLE ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS128_SHUFFLE ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS512_GATHER ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS256_GATHER ||
                      EvaluationMethod == EEvaluationMethod::AVX512_BS128_GATHER ||
                      EvaluationMethod == EEvaluationMethod::AVX2_BS256_GATHER ||
                      EvaluationMethod == EEvaluationMethod::AVX2_BS128_GATHER)
        {
            const auto treeLeafPtrAligned = applyData.TreeLeafValuesAligned.data();
            const auto firstLeafOffsetsAligned = applyData.TreeFirstLeafOffsetsAligned;
            alignas(64) double res[IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE] = {};
            for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
                CalcIndexesAndFetch<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock,
                            treeSplitsCurPtr, treeLeafPtrAligned + firstLeafOffsetsAligned[treeId], res, trees.GetModelTreeData()->GetTreeSizes()[treeId]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId];
            }
            size_t docId = 0;
            if constexpr (EvaluationMethod == EEvaluationMethod::AVX512_BS512_SHUFFLE ||
                          EvaluationMethod == EEvaluationMethod::AVX512_BS256_SHUFFLE ||
                          EvaluationMethod == EEvaluationMethod::AVX512_BS128_SHUFFLE ||
                          EvaluationMethod == EEvaluationMethod::AVX512_BS512_GATHER ||
                          EvaluationMethod == EEvaluationMethod::AVX512_BS256_GATHER ||
                          EvaluationMethod == EEvaluationMethod::AVX512_BS128_GATHER)
            {
                for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
                    for (size_t i = 0; i < 64; ++i, ++docId) {
                        resultsPtr[docId] = res[docId - i + i % 8 * 8 + i / 8];
                    }
                }
            }
            if constexpr (EvaluationMethod == EEvaluationMethod::AVX2_BS256_GATHER ||
                          EvaluationMethod == EEvaluationMethod::AVX2_BS128_GATHER) {
                for (size_t blockId = 0; blockId < SSEBlockCount; ++blockId) {
                    for (size_t i = 0; i < 32; ++i, ++docId) {
                        resultsPtr[docId] = res[docId - i + i % 8 * 4 + i / 8];
                    }
                }
            }
            for (; docId < docCountInBlock; ++docId) {
                resultsPtr[docId] = res[docId];
            }
            return;
        }

        if (EvaluationMethod != EEvaluationMethod::NON_SSE && IsSingleClassModel &&
                !CalcLeafIndexesOnly && allTreesAreShallow) {
            auto alignedResultsPtr = resultsPtr;
            TVector<char> resultsTmpArray;
            const size_t neededMemory = docCountInBlock * trees.GetDimensionsCount() * sizeof(double);
            if ((uintptr_t)alignedResultsPtr % sizeof(__m512d) != 0) {
                if (neededMemory < 2048) {
                    alignedResultsPtr = GetAligned((double*)alloca(neededMemory + 0x80));
                } else {
                    resultsTmpArray.yresize(neededMemory + 0x80);
                    alignedResultsPtr = (double*)GetAligned(resultsTmpArray.data());
                }
                CB_ENSURE((uintptr_t)alignedResultsPtr % sizeof(__m512d) == 0, "Failed to align");
                memset(alignedResultsPtr, 0, neededMemory);
            }
            auto treeEnd4 = treeStart + (((treeEnd - treeStart) | 0x3) ^ 0x3);
            for (size_t treeId = treeStart; treeId < treeEnd4; treeId += 4) {
                memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
                CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 0,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId];
                CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 1,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 1]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 1];
                CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 2,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 2]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 2];
                CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec + docCountInBlock * 3,
                                                           treeSplitsCurPtr, trees.GetModelTreeData()->GetTreeSizes()[treeId + 3]);
                treeSplitsCurPtr += trees.GetModelTreeData()->GetTreeSizes()[treeId + 3];

                CalculateLeafValues4<EvaluationMethod, SSEBlockCount>(
                    docCountInBlock,
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 0],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 1],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 2],
                    treeLeafPtr + firstLeafOffsetsPtr[treeId + 3],
                    indexesVec + docCountInBlock * 0,
                    indexesVec + docCountInBlock * 1,
                    indexesVec + docCountInBlock * 2,
                    indexesVec + docCountInBlock * 3,
                    alignedResultsPtr
                );
            }
            if (alignedResultsPtr != resultsPtr) {
                memcpy(resultsPtr, alignedResultsPtr, neededMemory);
            }
            treeStart = treeEnd4;
        }

        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
            memset(indexesVec, 0, sizeof(ui32) * docCountInBlock);
            if (!CalcLeafIndexesOnly && curTreeSize <= 8 && EvaluationMethod != EEvaluationMethod::NON_SSE) {
                CalcIndexesSse<EvaluationMethod, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr,
                                                           curTreeSize);
                if (IsSingleClassModel) { // single class model
                    CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec, resultsPtr);
                } else { // multiclass model
                    CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec,
                                             trees.GetDimensionsCount(), resultsPtr);
                }
            } else {
                CalcIndexesBasic<EvaluationMethod, NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr,
                                                 curTreeSize);
                if constexpr (CalcLeafIndexesOnly) {
                    indexesVecUI32 += docCountInBlock;
                    indexesVec += sizeof(ui32) * docCountInBlock;
                } else {
                    if (IsSingleClassModel) { // single class model
                        CalculateLeafValues(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                            indexesVecUI32, resultsPtr);
                    } else { // multiclass model
                        CalculateLeafValuesMulti(docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId],
                                                 indexesVecUI32, trees.GetDimensionsCount(), resultsPtr);
                    }
                }
            }
            treeSplitsCurPtr += curTreeSize;
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    Y_FORCE_INLINE void CalcTreesBlocked(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData<EvaluationMethod>* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        switch (docCountInBlock / IConsts<EvaluationMethod>::SUBBLOCK_SIZE) {
            case 0:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 0, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 1:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 1, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 2:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 2, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 3:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 3, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 4:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 4, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 5:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 5, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 6:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 6, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 7:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 7, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            case 8:
                CalcTreesBlockedImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, 8, CalcLeafIndexesOnly>(
                    trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr);
                break;
            default:
                Y_UNREACHABLE();
        }
    }

/*  template <EEvaluationMethod EvaluationMethod, bool IsSingleClassModel, bool NeedXorMask, bool calcIndexesOnly = false>
    inline void CalcTreesSingleDocImpl(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& ,
        const TCPUEvaluatorQuantizedData<EvaluationMethod>* quantizedData,
        size_t,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict results) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        Y_ASSERT(calcIndexesOnly || (results && AllOf(results, results + trees.GetDimensionsCount(),
                                                      [](double value) { return value == 0.0; })));
        const TRepackedBin* treeSplitsCurPtr =
            trees.GetRepackedBins().data() + trees.GetModelTreeData()->GetTreeStartOffsets()[treeStart];
        const double* treeLeafPtr = trees.GetFirstLeafPtrForTree(treeStart);
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            const auto curTreeSize = trees.GetModelTreeData()->GetTreeSizes()[treeId];
            TCalcerIndexType index = 0;
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);
                const ui32 featureIndex = (treeSplitsCurPtr[depth].FeatureIndex);
                if constexpr (NeedXorMask) {
                    const ui8 xorMask = (ui8)(treeSplitsCurPtr[depth].XorMask);
                    index |= ((binFeatures[featureIndex] ^ xorMask) >= borderVal) << depth;
                } else {
                    index |= (binFeatures[featureIndex] >= borderVal) << depth;
                }
            }
            if constexpr (calcIndexesOnly) {
                *indexesVec++ = index;
            } else {
                if constexpr (IsSingleClassModel) { // single class model
                    results[0] += treeLeafPtr[index];
                } else { // multiclass model
                    auto leafValuePtr = treeLeafPtr + index * trees.GetDimensionsCount();
                    for (int classId = 0; classId < (int)trees.GetDimensionsCount(); ++classId) {
                        results[classId] += leafValuePtr[classId];
                    }
                }
                treeLeafPtr += (1ull << curTreeSize) * trees.GetDimensionsCount();
            }
            treeSplitsCurPtr += curTreeSize;
        }
    }
*/
/*
    template <bool NeedXorMask>
    Y_FORCE_INLINE void CalcIndexesNonSymmetric(
        const TModelTrees& trees,
        const ui8* __restrict binFeatures,
        const size_t firstDocId,
        const size_t docCountInBlock,
        const size_t treeId,
        TCalcerIndexType* __restrict indexesVec
    ) {
        const TRepackedBin* treeSplitsPtr = trees.GetRepackedBins().data();
        const TNonSymmetricTreeStepNode* treeStepNodes = trees.GetModelTreeData()->GetNonSymmetricStepNodes().data();
        std::fill(indexesVec + firstDocId, indexesVec + docCountInBlock, trees.GetModelTreeData()->GetTreeStartOffsets()[treeId]);
        if (binFeatures == nullptr) {
            return;
        }
        size_t countStopped = 0;
        while (countStopped != docCountInBlock - firstDocId) {
            countStopped = 0;
            for (size_t docId = firstDocId; docId < docCountInBlock; ++docId) {
                const auto* stepNode = treeStepNodes + indexesVec[docId];
                const TRepackedBin split = treeSplitsPtr[indexesVec[docId]];
                ui8 featureValue = binFeatures[split.FeatureIndex * docCountInBlock + docId];
                if constexpr (NeedXorMask) {
                    featureValue ^= split.XorMask;
                }
                const auto diff = (featureValue >= split.SplitIdx) ? stepNode->RightSubtreeDiff
                                                                   : stepNode->LeftSubtreeDiff;
                countStopped += (diff == 0);
                indexesVec[docId] += diff;
            }
        }
    }
#if defined(_sse4_1_)
    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    inline void CalcNonSymmetricTrees(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexes,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeaturesI = quantizedData->QuantizedData.data();
        const TRepackedBin* treeSplitsPtr = trees.GetRepackedBins().data();
        const i32* treeStepNodes = reinterpret_cast<const i32*>(trees.GetModelTreeData()->GetNonSymmetricStepNodes().data());
        const ui32* __restrict nonSymmetricNodeIdToLeafIdPtr = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().data();
        const double* __restrict leafValuesPtr = trees.GetModelTreeData()->GetLeafValues().data();
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            const ui32 treeStartIndex = trees.GetModelTreeData()->GetTreeStartOffsets()[treeId];
            __m128i* indexesVec = reinterpret_cast<__m128i*>(indexes);
            size_t docId = 0;
            // handle special case of model containing only empty splits
            for (; binFeaturesI != nullptr && docId + 8 <= docCountInBlock; docId += 8, indexesVec+=2) {
                const ui8* __restrict binFeatures = binFeaturesI + docId;
                __m128i index0 = _mm_set1_epi32(treeStartIndex);
                __m128i index1 = _mm_set1_epi32(treeStartIndex);
                __m128i diffs0, diffs1;
                do {
                    const TRepackedBin splits[8] = {
                        treeSplitsPtr[_mm_extract_epi32(index0, 0)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 1)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 2)],
                        treeSplitsPtr[_mm_extract_epi32(index0, 3)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 0)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 1)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 2)],
                        treeSplitsPtr[_mm_extract_epi32(index1, 3)]
                    };
                    const __m128i zeroes = _mm_setzero_si128();
                    diffs0 = _mm_unpacklo_epi16(
                        _mm_hadd_epi16(
                            _mm_and_si128(
                                _mm_xor_si128(
                                    _mm_cmplt_epi32(
                                        _mm_xor_si128(
                                            _mm_setr_epi32(
                                                binFeatures[splits[0].FeatureIndex * docCountInBlock + 0],
                                                binFeatures[splits[1].FeatureIndex * docCountInBlock + 1],
                                                binFeatures[splits[2].FeatureIndex * docCountInBlock + 2],
                                                binFeatures[splits[3].FeatureIndex * docCountInBlock + 3]
                                            ),
                                            _mm_setr_epi32(
                                                splits[0].XorMask,
                                                splits[1].XorMask,
                                                splits[2].XorMask,
                                                splits[3].XorMask
                                            )
                                        ),
                                        _mm_setr_epi32(
                                            splits[0].SplitIdx,
                                            splits[1].SplitIdx,
                                            splits[2].SplitIdx,
                                            splits[3].SplitIdx
                                        )
                                   ),
                                   _mm_set1_epi32(0xffff0000)
                               ),
                                _mm_setr_epi32(
                                    treeStepNodes[_mm_extract_epi32(index0, 0)],
                                    treeStepNodes[_mm_extract_epi32(index0, 1)],
                                    treeStepNodes[_mm_extract_epi32(index0, 2)],
                                    treeStepNodes[_mm_extract_epi32(index0, 3)]
                                )
                            ),
                            zeroes
                        ),
                        zeroes
                    );
                    diffs1 = _mm_unpacklo_epi16(
                        _mm_hadd_epi16(
                            _mm_and_si128(
                                _mm_xor_si128(
                                    _mm_cmplt_epi32(
                                        _mm_xor_si128(
                                            _mm_setr_epi32(
                                                binFeatures[splits[4].FeatureIndex * docCountInBlock + 4],
                                                binFeatures[splits[5].FeatureIndex * docCountInBlock + 5],
                                                binFeatures[splits[6].FeatureIndex * docCountInBlock + 6],
                                                binFeatures[splits[7].FeatureIndex * docCountInBlock + 7]
                                            ),
                                            _mm_setr_epi32(
                                                splits[4].XorMask,
                                                splits[5].XorMask,
                                                splits[6].XorMask,
                                                splits[7].XorMask
                                            )
                                        ),
                                        _mm_setr_epi32(
                                            splits[4].SplitIdx,
                                            splits[5].SplitIdx,
                                            splits[6].SplitIdx,
                                            splits[7].SplitIdx
                                        )
                                    ),
                                    _mm_set1_epi32(0xffff0000)
                                ),
                                _mm_setr_epi32(
                                    treeStepNodes[_mm_extract_epi32(index1, 0)],
                                    treeStepNodes[_mm_extract_epi32(index1, 1)],
                                    treeStepNodes[_mm_extract_epi32(index1, 2)],
                                    treeStepNodes[_mm_extract_epi32(index1, 3)]
                                )
                            ),
                            zeroes
                        ),
                        zeroes
                    );
                    index0 = _mm_add_epi32(
                        index0,
                        diffs0
                    );
                    index1 = _mm_add_epi32(
                        index1,
                        diffs1
                    );
                } while (!_mm_testz_si128(diffs0, _mm_cmpeq_epi32(diffs0, diffs0)) || !_mm_testz_si128(diffs1, _mm_cmpeq_epi32(diffs1, diffs1)));
                _mm_storeu_si128(indexesVec, index0);
                _mm_storeu_si128(indexesVec + 1, index1);
            }
            if (docId < docCountInBlock) {
                CalcIndexesNonSymmetric<NeedXorMask>(trees, binFeaturesI, docId, docCountInBlock, treeId, indexes);
            }
            if constexpr (CalcLeafIndexesOnly) {
                const auto firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
                const auto approxDimension = trees.GetDimensionsCount();
                for (docId = 0; docId < docCountInBlock; ++docId) {
                    Y_ASSERT((nonSymmetricNodeIdToLeafIdPtr[indexes[docId]] - firstLeafOffsetsPtr[treeId]) % approxDimension == 0);
                    indexes[docId] = ((nonSymmetricNodeIdToLeafIdPtr[indexes[docId]] - firstLeafOffsetsPtr[treeId]) / approxDimension);
                }
                indexes += docCountInBlock;
            } else if constexpr (IsSingleClassModel) {
                for (docId = 0; docId + 8 <= docCountInBlock; docId+=8) {
                    resultsPtr[docId + 0] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 0]]];
                    resultsPtr[docId + 1] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 1]]];
                    resultsPtr[docId + 2] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 2]]];
                    resultsPtr[docId + 3] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 3]]];
                    resultsPtr[docId + 4] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 4]]];
                    resultsPtr[docId + 5] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 5]]];
                    resultsPtr[docId + 6] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 6]]];
                    resultsPtr[docId + 7] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId + 7]]];
                }
                for (; docId < docCountInBlock; ++docId) {
                    resultsPtr[docId] += leafValuesPtr[nonSymmetricNodeIdToLeafIdPtr[indexes[docId]]];
                }
            } else {
                const auto approxDim = trees.GetDimensionsCount();
                auto resultWritePtr = resultsPtr;
                for (docId = 0; docId < docCountInBlock; ++docId) {
                    const ui32 firstValueIdx = nonSymmetricNodeIdToLeafIdPtr[indexes[docId]];
                    for (int classId = 0; classId < (int)approxDim; ++classId, ++resultWritePtr) {
                        *resultWritePtr += leafValuesPtr[firstValueIdx + classId];
                    }
                }
            }
        }
    }

#else
    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly = false>
    inline void CalcNonSymmetricTrees(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            CalcIndexesNonSymmetric<NeedXorMask>(trees, binFeatures, 0, docCountInBlock, treeId, indexesVec);
            for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                indexesVec[docId] = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[indexesVec[docId]];
            }
            if constexpr (CalcLeafIndexesOnly) {
                const auto firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
                const auto approxDimension = trees.GetDimensionsCount();
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    Y_ASSERT((indexesVec[docId] - firstLeafOffsetsPtr[treeId]) % approxDimension == 0);
                    indexesVec[docId] = ((indexesVec[docId] - firstLeafOffsetsPtr[treeId]) / approxDimension);
                }
                indexesVec += docCountInBlock;
            } else {
                if constexpr (IsSingleClassModel) {
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        resultsPtr[docId] += trees.GetModelTreeData()->GetLeafValues()[indexesVec[docId]];
                    }
                } else {
                    auto resultWritePtr = resultsPtr;
                    for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                        const ui32 firstValueIdx = indexesVec[docId];
                        for (int classId = 0;
                             classId < (int)trees.GetDimensionsCount(); ++classId, ++resultWritePtr) {
                            *resultWritePtr += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx + classId];
                        }
                    }
                }
            }
        }
    }
#endif


    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcIndexesOnly>
    inline void CalcNonSymmetricTreesSingle(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr
    ) {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();
        TCalcerIndexType index;
        const TRepackedBin* treeSplitsPtr = trees.GetRepackedBins().data();
        const TNonSymmetricTreeStepNode* treeStepNodes = trees.GetModelTreeData()->GetNonSymmetricStepNodes().data();
        const auto firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
        // handle special empty-model case when there is no any splits at all
        const bool skipWork = quantizedData->QuantizedData.GetSize() == 0;
        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            index = trees.GetModelTreeData()->GetTreeStartOffsets()[treeId];
            while (!skipWork) {
                const auto* stepNode = treeStepNodes + index;
                const TRepackedBin split = treeSplitsPtr[index];
                ui8 featureValue = binFeatures[split.FeatureIndex];
                if constexpr (NeedXorMask) {
                    featureValue ^= split.XorMask;
                }
                const auto diff = (featureValue >= split.SplitIdx) ? stepNode->RightSubtreeDiff
                                                                   : stepNode->LeftSubtreeDiff;
                index += diff;
                if (diff == 0) {
                    break;
                }
            }
            const ui32 firstValueIdx = trees.GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[index];
            if constexpr (CalcIndexesOnly) {
                Y_ASSERT((firstValueIdx - firstLeafOffsetsPtr[treeId]) % trees.GetDimensionsCount() == 0);
                *indexesVec++ = ((firstValueIdx - firstLeafOffsetsPtr[treeId]) / trees.GetDimensionsCount());
            } else {
                if constexpr (IsSingleClassModel) {
                    *resultsPtr += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx];
                } else {
                    for (int classId = 0; classId < (int)trees.GetDimensionsCount(); ++classId) {
                        resultsPtr[classId] += trees.GetModelTreeData()->GetLeafValues()[firstValueIdx + classId];
                    }
                }
            }
        }
    }*/


    template <EEvaluationMethod EvaluationMethod, bool AreTreesOblivious, bool IsSingleDoc, bool IsSingleClassModel, bool NeedXorMask,
        bool CalcLeafIndexesOnly>
    struct CalcTreeFunctionInstantiationGetter {
        TTreeCalcFunction<EvaluationMethod> operator()() const {
            if constexpr (AreTreesOblivious) {
                // if constexpr (IsSingleDoc) {
                //    return CalcTreesSingleDocImpl<EvaluationMethod, IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                //} else {
                    return CalcTreesBlocked<EvaluationMethod, IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                //}
            } else {
                CB_ENSURE(false, "This method is not implemented with AVX yet");
                /*if constexpr (IsSingleDoc) {
                    return CalcNonSymmetricTreesSingle<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                } else {
                    return CalcNonSymmetricTrees<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
                }*/
            }
        }
    };

    template <EEvaluationMethod EvaluationMethod, template <EEvaluationMethod, bool...> class TFunctor, bool... params>
    struct FunctorTemplateParamsSubstitutor {
        static auto Call() {
            return TFunctor<EvaluationMethod, params...>()();
        }

        template <typename... Bools>
        static auto Call(bool nextParam, Bools... lastParams) {
            if (nextParam) {
                return FunctorTemplateParamsSubstitutor<EvaluationMethod, TFunctor, params..., true>::Call(lastParams...);
            } else {
                return FunctorTemplateParamsSubstitutor<EvaluationMethod, TFunctor, params..., false>::Call(lastParams...);
            }
        }
    };

    template<EEvaluationMethod EvaluationMethod>
    TTreeCalcFunction<EvaluationMethod> GetCalcTreesFunction(
        const TModelTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly
    ) {
        const bool areTreesOblivious = trees.IsOblivious();
        const bool isSingleDoc = (docCountInBlock == 1);
        const bool isSingleClassModel = (trees.GetDimensionsCount() == 1);
        const bool needXorMask = !trees.GetOneHotFeatures().empty();
        return FunctorTemplateParamsSubstitutor<EvaluationMethod, CalcTreeFunctionInstantiationGetter>::Call(
            areTreesOblivious, isSingleDoc, isSingleClassModel, needXorMask, calcIndexesOnly);
    }

    template TTreeCalcFunction<EEvaluationMethod::NON_SSE> GetCalcTreesFunction<EEvaluationMethod::NON_SSE>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::SSE3_BS128> GetCalcTreesFunction<EEvaluationMethod::SSE3_BS128>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX2_BS256> GetCalcTreesFunction<EEvaluationMethod::AVX2_BS256>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX2_BS128> GetCalcTreesFunction<EEvaluationMethod::AVX2_BS128>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX2_BS64> GetCalcTreesFunction<EEvaluationMethod::AVX2_BS64>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS512> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS512>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS256> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS256>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS128> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS128>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS64> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS64>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS512_SHUFFLE> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS512_SHUFFLE>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS256_SHUFFLE> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS256_SHUFFLE>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS128_SHUFFLE> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS128_SHUFFLE>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX2_BS256_GATHER> GetCalcTreesFunction<EEvaluationMethod::AVX2_BS256_GATHER>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX2_BS128_GATHER> GetCalcTreesFunction<EEvaluationMethod::AVX2_BS128_GATHER>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS512_GATHER> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS512_GATHER>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS256_GATHER> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS256_GATHER>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS128_GATHER> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS128_GATHER>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::NON_SSE_FP16> GetCalcTreesFunction<EEvaluationMethod::NON_SSE_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS512_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS512_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS256_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS256_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS128_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS128_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16>(const TModelTrees&, size_t, bool);
    template TTreeCalcFunction<EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16> GetCalcTreesFunction<EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16>(const TModelTrees&, size_t, bool);
}
