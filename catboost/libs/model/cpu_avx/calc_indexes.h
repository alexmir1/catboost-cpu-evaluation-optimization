#pragma once

#include "evaluator.h"

namespace NCB::NModelEvaluation::AVX {

    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t START_BLOCK, typename TIndexType>
    Y_FORCE_INLINE void CalcIndexesBasic(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            TIndexType* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            int curTreeSize) {
        if (START_BLOCK * IConsts<EvaluationMethod>::SUBBLOCK_SIZE >= docCountInBlock) {
            return;
        }
        for (int depth = 0; depth < curTreeSize; ++depth) {
            const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);

            const auto featureId = treeSplitsCurPtr[depth].FeatureIndex;
            const ui8* __restrict binFeaturePtr = &binFeatures[featureId * docCountInBlock];
            const ui8 xorMask = treeSplitsCurPtr[depth].XorMask;
            if (NeedXorMask) {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #ifndef _ubsan_enabled_
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = START_BLOCK * IConsts<EvaluationMethod>::SUBBLOCK_SIZE; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal) << depth;
                }
            } else {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #ifndef _ubsan_enabled_
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = START_BLOCK * IConsts<EvaluationMethod>::SUBBLOCK_SIZE; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId]) >= borderVal) << depth;
                }
            }
        }
    }

    template <bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesSseDepthed_SSE3_BS128(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            ui8* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr) {
        constexpr size_t SSE_BLOCK_SIZE = IConsts<EEvaluationMethod::SSE3_BS128>::SUBBLOCK_SIZE;
        if (SSEBlockCount == 0) {
            CalcIndexesBasic<EEvaluationMethod::SSE3_BS128, NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            return;
        }
    #define _mm_cmpge_epu8(a, b) _mm_cmpeq_epi8(_mm_max_epu8((a), (b)), (a))
    #define LOAD_16_DOC_HISTS(reg, binFeaturesPtr16) \
            const __m128i val##reg = _mm_lddqu_si128((const __m128i *)(binFeaturesPtr16));
    #define UPDATE_16_DOC_BINS(reg) \
            reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(val##reg, borderValVec), mask));

    #define LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(reg, binFeaturesPtr16) \
            LOAD_16_DOC_HISTS(reg, binFeaturesPtr16);\
            reg = _mm_or_si128(reg, _mm_and_si128(_mm_cmpge_epu8(_mm_xor_si128(val##reg, xorMaskVec), borderValVec), mask));
        NFORCE_UNROLL
        for (size_t regId = 0; regId < SSEBlockCount; regId += 2) {
            __m128i v0 = _mm_setzero_si128();
            __m128i v1 = _mm_setzero_si128();
            __m128i mask = _mm_set1_epi8(0x01);
            NFORCE_UNROLL
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * regId;
                const __m128i borderValVec = _mm_set1_epi8(treeSplitsCurPtr[depth].SplitIdx);
                if (!NeedXorMask) {
                    LOAD_16_DOC_HISTS(v0, binFeaturePtr);
                    if (regId + 1 < SSEBlockCount) {
                        LOAD_16_DOC_HISTS(v1, binFeaturePtr + SSE_BLOCK_SIZE);
                        UPDATE_16_DOC_BINS(v1);
                    }
                    UPDATE_16_DOC_BINS(v0);
                } else {
                    const __m128i xorMaskVec = _mm_set1_epi8(treeSplitsCurPtr[depth].XorMask);
                    LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(v0, binFeaturePtr);
                    if (regId + 1 < SSEBlockCount) {
                        LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED(v1, binFeaturePtr + SSE_BLOCK_SIZE);
                    }
                }
                mask = _mm_slli_epi16(mask, 1);
            }
            _mm_storeu_si128((__m128i *)(indexesVec + SSE_BLOCK_SIZE * regId), v0);
            if (regId + 1 < SSEBlockCount) {
                _mm_storeu_si128((__m128i *)(indexesVec + SSE_BLOCK_SIZE * regId + SSE_BLOCK_SIZE), v1);
            }
        }
        if (SSEBlockCount != 8) {
            CalcIndexesBasic<EEvaluationMethod::SSE3_BS128, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
        }
    #undef _mm_cmpge_epu8
    #undef LOAD_16_DOC_HISTS
    #undef UPDATE_16_DOC_BINS
    #undef LOAD_AND_UPDATE_16_DOCUMENT_BITS_XORED
    }

    template <bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesSseDepthed_AVX2_BS256(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            ui8* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr) {
        constexpr size_t SSE_BLOCK_SIZE = IConsts<EEvaluationMethod::AVX2_BS256>::SUBBLOCK_SIZE;
        if (SSEBlockCount == 0) {
            CalcIndexesBasic<EEvaluationMethod::AVX2_BS256, NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
            return;
        }
    #define _mm256_cmpge_epu8(a, b) _mm256_cmpeq_epi8(_mm256_max_epu8((a), (b)), (a))

        NFORCE_UNROLL
        for (size_t regId = 0; regId < SSEBlockCount; ++regId) {
            __m256i v = _mm256_setzero_si256();
            __m256i mask = _mm256_set1_epi8(0x01);
            NFORCE_UNROLL
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * regId;
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
            _mm256_storeu_si256((__m256i *)(indexesVec + SSE_BLOCK_SIZE * regId), v);
        }
        if (SSEBlockCount != 8) {
            CalcIndexesBasic<EEvaluationMethod::AVX2_BS256, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
        }
    #undef _mm256_cmpge_epu8
    }

    template <bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesSseDepthed_AVX512_BS512(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            ui8* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr) {
        constexpr size_t SSE_BLOCK_SIZE = IConsts<EEvaluationMethod::AVX512_BS512>::SUBBLOCK_SIZE;
        NFORCE_UNROLL
        for (size_t regId = 0; regId < SSEBlockCount; ++regId) {
            __m512i v = _mm512_setzero_si512();
            NFORCE_UNROLL
            for (int depth = 0; depth < curTreeSize; ++depth) {
                const ui8 *__restrict binFeaturePtr = binFeatures + treeSplitsCurPtr[depth].FeatureIndex * docCountInBlock + SSE_BLOCK_SIZE * regId;
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
            _mm512_storeu_si512((__m512i *)(indexesVec + SSE_BLOCK_SIZE * regId), v);
        }
        if (SSEBlockCount != 8) {
            CalcIndexesBasic<EEvaluationMethod::AVX512_BS512, NeedXorMask, SSEBlockCount>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t SSEBlockCount, int curTreeSize>
    Y_FORCE_INLINE void CalcIndexesSseDepthed(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            ui8* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr) {
        switch (EvaluationMethod) {
            case EEvaluationMethod::SSE3_BS128:
                CalcIndexesSseDepthed_SSE3_BS128<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    indexesVec,
                    treeSplitsCurPtr
                );
                return;
            case EEvaluationMethod::AVX2_BS256:
            case EEvaluationMethod::AVX2_BS128:
            case EEvaluationMethod::AVX2_BS64:
                CalcIndexesSseDepthed_AVX2_BS256<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    indexesVec,
                    treeSplitsCurPtr
                );
                return;
            case EEvaluationMethod::AVX512_BS512:
            case EEvaluationMethod::AVX512_BS256:
            case EEvaluationMethod::AVX512_BS128:
            case EEvaluationMethod::AVX512_BS64:
            case EEvaluationMethod::AVX512_BS512_FP16:
            case EEvaluationMethod::AVX512_BS256_FP16:
            case EEvaluationMethod::AVX512_BS128_FP16:
                CalcIndexesSseDepthed_AVX512_BS512<NeedXorMask, SSEBlockCount, curTreeSize>(
                    binFeatures,
                    docCountInBlock,
                    indexesVec,
                    treeSplitsCurPtr
                );
                return;
            case EEvaluationMethod::NON_SSE:
            case EEvaluationMethod::AVX512_BS512_SHUFFLE:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE:
            case EEvaluationMethod::AVX512_BS512_GATHER:
            case EEvaluationMethod::AVX512_BS256_GATHER:
            case EEvaluationMethod::AVX512_BS128_GATHER:
            case EEvaluationMethod::AVX2_BS256_GATHER:
            case EEvaluationMethod::AVX2_BS128_GATHER:
            case EEvaluationMethod::NON_SSE_FP16:
            case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
                CB_ENSURE(false);
        }
    }

    template <EEvaluationMethod EvaluationMethod, bool NeedXorMask, size_t SSEBlockCount>
    static void CalcIndexesSse(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TRepackedBin* __restrict treeSplitsCurPtr,
        const int curTreeSize) {
        switch (curTreeSize)
        {
        case 1:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 1>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 2:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 2>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 3:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 3>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 4:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 4>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 5:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 5>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 6:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 6>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 7:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 7>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        case 8:
            CalcIndexesSseDepthed<EvaluationMethod, NeedXorMask, SSEBlockCount, 8>(binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
            break;
        default:
            break;
        }
    }
}
