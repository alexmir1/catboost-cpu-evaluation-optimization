#pragma once 

#include "evaluator.h"

#include <library/cpp/float16/float16.h>

namespace NCB::NModelEvaluation::AVX {

    template <bool NeedXorMask, size_t SSEBlockCount, size_t curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetch_AVX512_BS512_Shuffle_FP16(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const ui16* __restrict treeLeafPtr,
            float* __restrict res) {
        constexpr size_t SSE_BLOCK_SIZE = IConsts<EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16>::SUBBLOCK_SIZE;
        constexpr size_t RegNum = curTreeSize >= 5 ? (1 << curTreeSize) * 16 / 512 : 1;
        __m512i leaf_vals[RegNum];
        NFORCE_UNROLL
        for (size_t i = 0; i < RegNum; ++i) {
            leaf_vals[i] = _mm512_load_si512((__m512i *)(treeLeafPtr + i * (512 / 16)));
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

#define SHUFFLE32\
                __m512i shuffle = _mm512_cvtepu8_epi16(_mm256_and_si256(vp, _mm256_set1_epi8(0x1f)));\
                __m256i reg_index = _mm256_srli_epi64(_mm256_and_si256(vp, _mm256_set1_epi8(0xe0)), 5);\
                __m512i data = _mm512_setzero_si512();\
                NFORCE_UNROLL\
                for (size_t k = 0; k < RegNum; ++k) {\
                    data = _mm512_or_si512(data, _mm512_maskz_permutexvar_epi16(\
                            _mm256_cmpeq_epu8_mask(reg_index, _mm256_set1_epi8(k)),\
                            shuffle,\
                            leaf_vals[k]\
                    ));\
                }

            /*
            for (size_t j = 0; j < 2; ++j) {
                __m256i vp;
                if (j == 0) {
                    vp = _mm512_extracti32x8_epi32(v, 0);
                } else {
                    vp = _mm512_extracti32x8_epi32(v, 1);
                }
                SHUFFLE32
                _mm512_store_ps(   res + 32 * (j + 2 * i), _mm512_add_ps(
                    _mm512_load_ps(res + 32 * (j + 2 * i)), _mm512_cvtph_ps(_mm512_castsi512_si256(data))));
                _mm512_store_ps(   res + 32 * (j + 2 * i) + 16, _mm512_add_ps(
                    _mm512_load_ps(res + 32 * (j + 2 * i) + 16), _mm512_cvtph_ps(_mm512_extracti32x8_epi32(data, 1))));
            }*/
            NFORCE_UNROLL
            for (size_t j = 0; j < 2; ++j) {
                __m512i vp;
                if (j == 0) {
                    vp = v;
                } else {
                    vp = _mm512_srli_epi16(v, 8);
                }
                __m512i shuffle = _mm512_and_si512(vp, _mm512_set1_epi16(0x1f));
                __m512i reg_index = _mm512_srli_epi64(_mm512_and_si512(vp, _mm512_set1_epi16(0xe0)), 5);
                __m512i data = _mm512_setzero_si512();
                NFORCE_UNROLL
                for (size_t k = 0; k < RegNum; ++k) {
                    data = _mm512_or_si512(data, _mm512_maskz_permutexvar_epi16(
                            _mm512_cmpeq_epu16_mask(reg_index, _mm512_set1_epi16(k)),
                            shuffle,
                            leaf_vals[k]
                    ));
                }
                _mm512_store_ps(   res + 32 * (j + 2 * i), _mm512_add_ps(
                    _mm512_load_ps(res + 32 * (j + 2 * i)), _mm512_cvtph_ps(_mm512_castsi512_si256(data))));
                _mm512_store_ps(   res + 32 * (j + 2 * i) + 16, _mm512_add_ps(
                    _mm512_load_ps(res + 32 * (j + 2 * i) + 16), _mm512_cvtph_ps(_mm512_extracti32x8_epi32(data, 1))));
            }
        }
        if (SSEBlockCount != 8) {
            const size_t diff = docCountInBlock - SSEBlockCount * 64;
            Y_ASSERT(diff < 64);
            if (diff > 0) {
                alignas(64) ui8 indexesVec[64] = {};
                alignas(64) ui16 vals[64];
                CalcIndexesBasic<EEvaluationMethod::AVX512_BS512, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec - SSEBlockCount * 64, treeSplitsCurPtr, curTreeSize);
                __m256i vp = _mm256_load_si256((__m256i*)indexesVec);
                SHUFFLE32
                _mm512_store_ps(vals, data);
                if (diff > 32) {
                    __m256i vp = _mm256_load_si256((__m256i*)(indexesVec + 32));
                    SHUFFLE32
                    _mm512_store_ps(vals + 32, data);
                }
                for (size_t i = 0; i < diff; ++i) {
                    res[SSEBlockCount * 64 + i] += TFloat16::Load(vals[i]).AsFloat();
                }
            }
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesAndFetchDepthedFP16(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            const ui16* __restrict treeLeafPtr,
            float* __restrict res) {
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
            case EEvaluationMethod::AVX512_BS512_GATHER:
            case EEvaluationMethod::AVX512_BS256_GATHER:
            case EEvaluationMethod::AVX512_BS128_GATHER:
            case EEvaluationMethod::AVX2_BS256_GATHER:
            case EEvaluationMethod::AVX2_BS128_GATHER:
            case EEvaluationMethod::AVX512_BS512_FP16:
            case EEvaluationMethod::AVX512_BS256_FP16:
            case EEvaluationMethod::AVX512_BS128_FP16:
                CB_ENSURE(false);
            case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
                CalcIndexesAndFetch_AVX512_BS512_Shuffle_FP16<NeedXorMask, SSEBlockCount, curTreeSize>(
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
    static void CalcIndexesAndFetchFP16(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        const TRepackedBin* __restrict treeSplitsCurPtr,
        const ui16* __restrict treeLeafPtr,
        float* __restrict res,
        const int curTreeSize) {
        switch (curTreeSize)
        {
        case 1:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 1>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 2:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 2>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 3:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 3>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 4:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 4>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 5:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 5>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 6:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 6>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 7:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 7>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        case 8:
            CalcIndexesAndFetchDepthedFP16<EvaluationMethod, NeedXorMask, SSEBlockCount, 8>(binFeatures, docCountInBlock, treeSplitsCurPtr, treeLeafPtr, res);
            break;
        default:
            break;
        }
    }
}
