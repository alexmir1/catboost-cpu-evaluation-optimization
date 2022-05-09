#pragma once 

#include "evaluator.h"

#include <library/cpp/float16/float16.h>

namespace NCB::NModelEvaluation::AVX {

    template <bool NeedXorMask, size_t SSEBlockCount, size_t curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetch_AVX512_BS512_Shuffle(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const double* __restrict treeLeafPtr,
            double* __restrict res) {
        constexpr size_t SSE_BLOCK_SIZE = IConsts<EEvaluationMethod::AVX512_BS512_SHUFFLE>::SUBBLOCK_SIZE;
        constexpr size_t RegNum = (1 << curTreeSize) * 64 / 512;
        __m512d leaf_vals[RegNum];
        NFORCE_UNROLL
        for (size_t i = 0; i < RegNum; ++i) {
            leaf_vals[i] = _mm512_load_pd((__m512i *)(treeLeafPtr + i * (512 / 64)));
        }
        NFORCE_UNROLL
        for (size_t i = 0; i < SSEBlockCount; ++i) {
            __m512i v = _mm512_setzero_si512();
            NFORCE_UNROLL
            for (size_t depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * i;
                const __m512i borderValVec = _mm512_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
                if (!NeedXorMask) {
                    const __m512i val = _mm512_loadu_si512((const __m512i *)binFeaturePtr);
                    v = _mm512_or_si512(v, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask(val, borderValVec), (1 << depth)));
                } else {
                    const __m512i xorMaskVec = _mm512_set1_epi8(treeSplitsCurPtr[depth].XorMask);
                    const __m512i val = _mm512_loadu_si512((const __m512i *)binFeaturePtr);
                    v = _mm512_or_si512(v, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask(_mm512_xor_si512(val, xorMaskVec), borderValVec), (1 << depth)));
                }
            }

            NFORCE_UNROLL
            for (size_t j = 0; j < 8; ++j) {
                __m512i vp;
                switch (j) {
                    case 0:
                        vp = v;
                        break;
                    case 1:
                        vp = _mm512_srli_epi64(v, 8 * 1);
                        break;
                    case 2:
                        vp = _mm512_srli_epi64(v, 8 * 2);
                        break;
                    case 3:
                        vp = _mm512_srli_epi64(v, 8 * 3);
                        break;
                    case 4:
                        vp = _mm512_srli_epi64(v, 8 * 4);
                        break;
                    case 5:
                        vp = _mm512_srli_epi64(v, 8 * 5);
                        break;
                    case 6:
                        vp = _mm512_srli_epi64(v, 8 * 6);
                        break;
                    case 7:
                        vp = _mm512_srli_epi64(v, 8 * 7);
                        break;
                }
                __m512i shuffle = _mm512_and_si512(vp, _mm512_set1_epi64(0x07));
                __m512i reg_index = _mm512_srli_epi64(_mm512_and_si512(vp, _mm512_set1_epi64(0xf8)), 3);
                __m512d data = _mm512_setzero_si512();
                NFORCE_UNROLL
                for (size_t k = 0; k < RegNum; ++k) {
                    data = _mm512_mask_permutexvar_pd(data,
                            _mm512_cmpeq_epu64_mask(reg_index, _mm512_set1_epi64(k)),
                            shuffle,
                            leaf_vals[k]
                    );
                }
                _mm512_store_pd(   res + 8 * (j + 8 * i), _mm512_add_pd(
                    _mm512_load_pd(res + 8 * (j + 8 * i)), data));
            }
        }
        if (SSEBlockCount != 8) {
            const size_t diff = docCountInBlock - SSEBlockCount * 64;
            Y_ASSERT(diff < 64);
            if (diff > 0) {
                alignas(64) ui8 indexesVec[64] = {};
                alignas(64) double vals[64];
                CalcIndexesBasic<EEvaluationMethod::AVX512_BS512, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec - SSEBlockCount * 64, treeSplitsCurPtr, curTreeSize);
                for (size_t j = 0; j < (diff + 7) / 8; ++j) {
                    __m512i vp = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128(((ui64*)indexesVec)[j]));
                    __m512i shuffle = _mm512_and_si512(vp, _mm512_set1_epi64(0x07));
                    __m512i reg_index = _mm512_srli_epi64(_mm512_and_si512(vp, _mm512_set1_epi64(0xf8)), 3);
                    __m512d data = _mm512_setzero_si512();
                    NFORCE_UNROLL
                    for (size_t k = 0; k < RegNum; ++k) {
                        data = _mm512_or_pd(data, _mm512_maskz_permutexvar_pd(
                                _mm512_cmpeq_epu64_mask(reg_index, _mm512_set1_epi64(k)),
                                shuffle,
                                leaf_vals[k]
                        ));
                    }
                    _mm512_store_pd(vals + 8 * j, data);
                }
                for (size_t i = 0; i < diff; ++i) {
                    res[SSEBlockCount * 64 + i] += vals[i];
                }
            }
        }
    }

    template <bool NeedXorMask, size_t SSEBlockCount, size_t curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetch_AVX512_BS512_Gather(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const double* __restrict treeLeafPtr,
            double* __restrict res) {
        constexpr size_t SSE_BLOCK_SIZE = 64;
        NFORCE_UNROLL
        for (size_t i = 0; i < SSEBlockCount; ++i) {
            __m512i v = _mm512_setzero_si512();
            NFORCE_UNROLL
            for (size_t depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * i;
                const __m512i borderValVec = _mm512_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
                if (!NeedXorMask) {
                    const __m512i val = _mm512_loadu_si512((const __m512i *)binFeaturePtr);
                    v = _mm512_or_si512(v, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask(val, borderValVec), (1 << depth)));
                } else {
                    const __m512i xorMaskVec = _mm512_set1_epi8(treeSplitsCurPtr[depth].XorMask);
                    const __m512i val = _mm512_loadu_si512((const __m512i *)binFeaturePtr);
                    v = _mm512_or_si512(v, _mm512_maskz_set1_epi8(_mm512_cmpge_epu8_mask(_mm512_xor_si512(val, xorMaskVec), borderValVec), (1 << depth)));
                }
            }

            /*NFORCE_UNROLL
            for (size_t j = 0; j < 4; ++j) {
                __m512i vp;
                if (j == 0) {
                    vp = _mm512_unpacklo_epi16(_mm512_unpacklo_epi8(v, _mm512_setzero_si512()), _mm512_setzero_si512());
                } else if (j == 1) {
                    vp = _mm512_unpackhi_epi16(_mm512_unpacklo_epi8(v, _mm512_setzero_si512()), _mm512_setzero_si512());
                } else if (j == 2) {
                    vp = _mm512_unpacklo_epi16(_mm512_unpackhi_epi8(v, _mm512_setzero_si512()), _mm512_setzero_si512());
                } else {
                    vp = _mm512_unpackhi_epi16(_mm512_unpackhi_epi8(v, _mm512_setzero_si512()), _mm512_setzero_si512());
                }
                __m512d data1 = _mm512_i32gather_pd(_mm512_castsi512_si256(vp),       treeLeafPtr, sizeof(double));
                __m512d data2 = _mm512_i32gather_pd(_mm512_extracti32x8_epi32(vp, 1), treeLeafPtr, sizeof(double));
                _mm512_store_pd(   res + 8 * (2 * j + 8 * i), _mm512_add_pd(
                    _mm512_load_pd(res + 8 * (2 * j + 8 * i)), data1));
                _mm512_store_pd(   res + 8 * (2 * j + 8 * i + 1), _mm512_add_pd(
                    _mm512_load_pd(res + 8 * (2 * j + 8 * i + 1)), data2));
            }*/
            NFORCE_UNROLL
            for (size_t j = 0; j < 8; ++j) {
                __m512i vp;
                switch (j) {
                    case 0:
                        vp = v;
                        break;
                    case 1:
                        vp = _mm512_srli_epi64(v, 8 * 1);
                        break;
                    case 2:
                        vp = _mm512_srli_epi64(v, 8 * 2);
                        break;
                    case 3:
                        vp = _mm512_srli_epi64(v, 8 * 3);
                        break;
                    case 4:
                        vp = _mm512_srli_epi64(v, 8 * 4);
                        break;
                    case 5:
                        vp = _mm512_srli_epi64(v, 8 * 5);
                        break;
                    case 6:
                        vp = _mm512_srli_epi64(v, 8 * 6);
                        break;
                    case 7:
                        vp = _mm512_srli_epi64(v, 8 * 7);
                        break;
                }
                __m512i ind = _mm512_and_si512(vp, _mm512_set1_epi64(0xff));
                __m512d data = _mm512_i64gather_pd(ind, treeLeafPtr, sizeof(double));
                _mm512_store_pd(   res + 8 * (j + 8 * i), _mm512_add_pd(
                    _mm512_load_pd(res + 8 * (j + 8 * i)), data));
            }
        }
        if (SSEBlockCount != 8) {
            const size_t diff = docCountInBlock - SSEBlockCount * 64;
            Y_ASSERT(diff < 64);
            if (diff > 0) {
                alignas(64) ui8 indexesVec[64] = {};
                alignas(64) double vals[64];
                CalcIndexesBasic<EEvaluationMethod::AVX512_BS512, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec - SSEBlockCount * 64, treeSplitsCurPtr, curTreeSize);
                for (size_t j = 0; j < (diff + 7) / 8; ++j) {
                    __m256i vp = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(((ui64*)indexesVec)[j]));
                    __m512d data = _mm512_i32gather_pd(vp, treeLeafPtr, sizeof(double));
                    _mm512_store_pd(vals + 8 * j, data);
                }
                for (size_t i = 0; i < diff; ++i) {
                    res[SSEBlockCount * 64 + i] += vals[i];
                }
            }
        }
    }

    template <bool NeedXorMask, size_t SSEBlockCount, size_t curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetch_AVX2_BS256_Gather(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const double* __restrict treeLeafPtr,
            double* __restrict res) {
        constexpr size_t SSE_BLOCK_SIZE = 32;
    #define _mm256_cmpge_epu8(a, b) _mm256_cmpeq_epi8(_mm256_max_epu8((a), (b)), (a))
        NFORCE_UNROLL
        for (size_t i = 0; i < SSEBlockCount; ++i) {
            __m256i v = _mm256_setzero_si256();
            __m256i mask = _mm256_set1_epi8(0x01);
            NFORCE_UNROLL
            for (size_t depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * i;
                const __m256i borderValVec = _mm256_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
                if (!NeedXorMask) {
                    const __m256i val = _mm256_lddqu_si256((const __m256i *)(binFeaturePtr));
                    v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpge_epu8(val, borderValVec), mask));
                } else {
                    const __m256i xorMaskVec = _mm256_set1_epi8(treeSplitsCurPtr[depth].XorMask);
                    const __m256i val = _mm256_lddqu_si256((const __m256i *)(binFeaturePtr));
                    v = _mm256_or_si256(v, _mm256_and_si256(_mm256_cmpge_epu8(_mm256_xor_si256(val, xorMaskVec), borderValVec), mask));
                }
                mask = _mm256_slli_epi16(mask, 1);
            }
    #undef _mm256_cmpge_epu8

            NFORCE_UNROLL
            for (size_t j = 0; j < 8; ++j) {
                __m256i vp;
                switch (j) {
                    case 0:
                        vp = v;
                        break;
                    case 1:
                        vp = _mm256_srli_epi64(v, 8 * 1);
                        break;
                    case 2:
                        vp = _mm256_srli_epi64(v, 8 * 2);
                        break;
                    case 3:
                        vp = _mm256_srli_epi64(v, 8 * 3);
                        break;
                    case 4:
                        vp = _mm256_srli_epi64(v, 8 * 4);
                        break;
                    case 5:
                        vp = _mm256_srli_epi64(v, 8 * 5);
                        break;
                    case 6:
                        vp = _mm256_srli_epi64(v, 8 * 6);
                        break;
                    case 7:
                        vp = _mm256_srli_epi64(v, 8 * 7);
                        break;
                }
                __m256i ind = _mm256_and_si256(vp, _mm256_set1_epi64x(0xff));
                __m256d data = _mm256_i64gather_pd(treeLeafPtr, ind, sizeof(double));
                _mm256_store_pd(   res + 4 * (j + 8 * i), _mm256_add_pd(
                    _mm256_load_pd(res + 4 * (j + 8 * i)), data));
            }
        }
        if (SSEBlockCount != 8) {
            const size_t diff = docCountInBlock - SSEBlockCount * 32;
            Y_ASSERT(diff < 32);
            if (diff > 0) {
                alignas(32) ui8 indexesVec[32] = {};
                alignas(32) double vals[32];
                CalcIndexesBasic<EEvaluationMethod::AVX2_BS256, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec - SSEBlockCount * 32, treeSplitsCurPtr, curTreeSize);
                for (size_t j = 0; j < (diff + 3) / 4; ++j) {
                    __m128i vp = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(((ui32*)indexesVec)[j]));
                    __m256d data = _mm256_i32gather_pd(treeLeafPtr, vp, sizeof(double));
                    _mm256_store_pd(vals + 4 * j, data);
                }
                for (size_t i = 0; i < diff; ++i) {
                    res[SSEBlockCount * 32 + i] += vals[i];
                }
            }
        }
    }


    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetchDepthed(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const double* __restrict treeLeafPtr,
            double* __restrict res) {
        switch (EvaluationMethod) {
            case EEvaluationMethod::NON_SSE:
            case EEvaluationMethod::NON_SSE_FP16:
            case EEvaluationMethod::SSE3_BS128:
            case EEvaluationMethod::AVX2_BS256:
            case EEvaluationMethod::AVX2_BS128:
            case EEvaluationMethod::AVX2_BS64:
            case EEvaluationMethod::AVX512_BS512:
            case EEvaluationMethod::AVX512_BS256:
            case EEvaluationMethod::AVX512_BS128:
            case EEvaluationMethod::AVX512_BS64:
            case EEvaluationMethod::AVX512_BS512_FP16:
            case EEvaluationMethod::AVX512_BS256_FP16:
            case EEvaluationMethod::AVX512_BS128_FP16:
            case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
                CB_ENSURE(false);
            case EEvaluationMethod::AVX512_BS512_SHUFFLE:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE:
                CalcIndexesAndFetch_AVX512_BS512_Shuffle<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    treeSplitsCurPtr,
                    treeLeafPtr,
                    res
                );
                return;
            case EEvaluationMethod::AVX2_BS256_GATHER:
            case EEvaluationMethod::AVX2_BS128_GATHER:
                CalcIndexesAndFetch_AVX2_BS256_Gather<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    treeSplitsCurPtr,
                    treeLeafPtr,
                    res
                );
                return;
            case EEvaluationMethod::AVX512_BS512_GATHER:
            case EEvaluationMethod::AVX512_BS256_GATHER:
            case EEvaluationMethod::AVX512_BS128_GATHER:
                CalcIndexesAndFetch_AVX512_BS512_Gather<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    treeSplitsCurPtr,
                    treeLeafPtr,
                    res
                );
                return;
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t SSEBlockCount>
    static void CalcIndexesAndFetch(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        const TRepackedBin* __restrict treeSplitsCurPtr,
        const double* __restrict treeLeafPtr,
        double* __restrict res,
        const int curTreeSize) {
        switch (curTreeSize)
        {
        case 1:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 1>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 2:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 2>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 3:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 3>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 4:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 4>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 5:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 5>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 6:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 6>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 7:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 7>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 8:
            CalcIndexesAndFetchDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 8>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        }
    }
}
