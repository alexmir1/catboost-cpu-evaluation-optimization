#pragma once

#include <immintrin.h>

#include <catboost/libs/model/model.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/cpp/sse/sse.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/ymath.h>

#include "evaluator_interface.h"
#include "util.h"

namespace NCB::NModelEvaluation::AVX {

    template<EEvaluationMethod EvaluationMethod>
    class TCPUEvaluatorQuantizedData final : public IQuantizedData {
    public:
        TCPUEvaluatorQuantizedData() = default;

        TCPUEvaluatorQuantizedData(TCPUEvaluatorQuantizedData&& other) = default;

        TCPUEvaluatorQuantizedData(size_t size) {
            Holder.resize(size + 64);
            QuantizedData = { GetAligned(Holder.data()), size };
        }

        size_t ObjectsCount = 0;
        size_t BlocksCount = 0;
        size_t BlockStride = 0;
        TArrayRef<ui8> QuantizedData;
        TVector<ui8> Holder;


        TCPUEvaluatorQuantizedData ExtractBlock(size_t blockId) const {
            TCPUEvaluatorQuantizedData result;
            result.BlocksCount = 1;
            size_t width = QuantizedData.size() / ObjectsCount;

            result.ObjectsCount = Min(
                IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE,
                ObjectsCount - IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE * (blockId));
            result.BlockStride = width * result.ObjectsCount;

            result.QuantizedData = QuantizedData.Slice(
                BlockStride * blockId,
                result.BlockStride
            );
            return result;
        }

    public:
        size_t GetObjectsCount() const override {
            return ObjectsCount;
        }
    };

    inline void OneHotBinsFromTransposedCatFeatures(
        const TConstArrayRef<TOneHotFeature> OneHotFeatures,
        const THashMap<int, int> catFeaturePackedIndex,
        const size_t docCount,
        TArrayRef<ui32> transposedHash,
        ui8*& result
    ) {
        for (const auto& oheFeature : OneHotFeatures) {
            const auto catIdx = catFeaturePackedIndex.at(oheFeature.CatFeatureIndex);
            for (size_t docId = 0; docId < docCount; ++docId) {
                static_assert(sizeof(int) >= sizeof(i32));
                const int val = *reinterpret_cast<i32*>(&(transposedHash[catIdx * docCount + docId]));
                ui8* writePosition = &result[docId];
                for (size_t blockStart = 0;
                     blockStart < oheFeature.Values.size();
                     blockStart += MAX_VALUES_PER_BIN) {
                    const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, oheFeature.Values.size());
                    for (size_t borderIdx = blockStart; borderIdx < blockEnd; ++borderIdx) {
                        *writePosition |= (ui8)(val == oheFeature.Values[borderIdx]) * (borderIdx - blockStart + 1);
                    }
                    writePosition += docCount;
                }
            }
            result += docCount * ((oheFeature.Values.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
        }
    }

    template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloatsNonSseSingle(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        size_t docId,
        float nanSubstitutionValue = 0.0f
    ) {
        float val = floatAccessor(position, start + docId);
        if (UseNanSubstitution) {
            if (std::isnan(val)) {
                val = nanSubstitutionValue;
            }
        }

        ui8* writePtr = result + docId;
        for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
            for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                *writePtr += (ui8)(val > borders[borderId]);
            }
            writePtr += docCount;
        }
    }

    template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloatsNonSse(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        float nanSubstitutionValue = 0.0f
    ) {
        // result can be not aligned
        /*const auto docCount8 = (docCount | 0x7) ^0x7;
        for (size_t docId = 0; docId < docCount8; docId += 8) {
            float val[8] = {
                floatAccessor(position, start + docId + 0),
                floatAccessor(position, start + docId + 1),
                floatAccessor(position, start + docId + 2),
                floatAccessor(position, start + docId + 3),
                floatAccessor(position, start + docId + 4),
                floatAccessor(position, start + docId + 5),
                floatAccessor(position, start + docId + 6),
                floatAccessor(position, start + docId + 7)
            };
            if (UseNanSubstitution) {
                for (size_t i = 0; i < 8; ++i) {
                    if (std::isnan(val[i])) {
                        val[i] = nanSubstitutionValue;
                    }
                }
            }
            ui32* writePtr = (ui32*)(result + docId);
            for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    const auto border = borders[borderId];
                    writePtr[0] += (val[0] > border) + ((val[1] > border) << 8) + ((val[2] > border) << 16)
                                   + ((val[3] > border) << 24);
                    writePtr[1] += (val[4] > border) + ((val[5] > border) << 8) + ((val[6] > border) << 16)
                                   + ((val[7] > border) << 24);
                }
                writePtr = (ui32*)((ui8*)writePtr + docCount);
            }
        }*/
        for (size_t docId = 0; docId < docCount; ++docId) {
            BinarizeFloatsNonSseSingle<UseNanSubstitution, TFloatFeatureAccessor>(
                position,
                docCount,
                floatAccessor,
                borders,
                start,
                result,
                docId,
                nanSubstitutionValue
            );
        }
        result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }

#define FLOAT_ACCESSOR_BLOCK_4(BLOCK_NUM)\
    floatAccessor(position, start + docId + 4 * BLOCK_NUM + 0),\
    floatAccessor(position, start + docId + 4 * BLOCK_NUM + 1),\
    floatAccessor(position, start + docId + 4 * BLOCK_NUM + 2),\
    floatAccessor(position, start + docId + 4 * BLOCK_NUM + 3),


    template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloats_SSE3_BS128(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        const float nanSubstitutionValue = 0.0f
    ) {
        const __m128 substitutionValVec = _mm_set1_ps(nanSubstitutionValue);
        const auto docCount16 = (docCount | 0xf) ^ 0xf;
        for (size_t docId = 0; docId < docCount16; docId += 16) {
            alignas(0x10) const float val[16] = {
                FLOAT_ACCESSOR_BLOCK_4(0)
                FLOAT_ACCESSOR_BLOCK_4(1)
                FLOAT_ACCESSOR_BLOCK_4(2)
                FLOAT_ACCESSOR_BLOCK_4(3)
            };
            const __m128i mask = _mm_set1_epi8(1);
            __m128 floats0 = _mm_load_ps(val);
            __m128 floats1 = _mm_load_ps(val + 4);
            __m128 floats2 = _mm_load_ps(val + 8);
            __m128 floats3 = _mm_load_ps(val + 12);

            if (UseNanSubstitution) {
                {
                    const __m128 masks = _mm_cmpunord_ps(floats0, floats0);
                    floats0 = _mm_or_ps(_mm_andnot_ps(masks, floats0), _mm_and_ps(masks, substitutionValVec));
                }
                {
                    const __m128 masks = _mm_cmpunord_ps(floats1, floats1);
                    floats1 = _mm_or_ps(_mm_andnot_ps(masks, floats1), _mm_and_ps(masks, substitutionValVec));
                }
                {
                    const __m128 masks = _mm_cmpunord_ps(floats2, floats2);
                    floats2 = _mm_or_ps(_mm_andnot_ps(masks, floats2), _mm_and_ps(masks, substitutionValVec));
                }
                {
                    const __m128 masks = _mm_cmpunord_ps(floats3, floats3);
                    floats3 = _mm_or_ps(_mm_andnot_ps(masks, floats3), _mm_and_ps(masks, substitutionValVec));
                }
            }
            ui8* writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                __m128i resultVec = _mm_setzero_si128();
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    const __m128 borderVec = _mm_set1_ps(borders[borderId]);
                    const __m128i r0 = _mm_castps_si128(_mm_cmpgt_ps(floats0, borderVec));
                    const __m128i r1 = _mm_castps_si128(_mm_cmpgt_ps(floats1, borderVec));
                    const __m128i r2 = _mm_castps_si128(_mm_cmpgt_ps(floats2, borderVec));
                    const __m128i r3 = _mm_castps_si128(_mm_cmpgt_ps(floats3, borderVec));
                    const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
                    resultVec = _mm_add_epi8(resultVec, _mm_and_si128(packed, mask));
                }
                _mm_storeu_si128((__m128i *)writePtr, resultVec);
                writePtr += docCount;
            }
        }
        for (size_t docId = docCount16; docId < docCount; ++docId) {
            float val = floatAccessor(position, start + docId);
            if (UseNanSubstitution) {
                if (std::isnan(val)) {
                    val = nanSubstitutionValue;
                }
            }
            ui8* writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    *writePtr += (ui8)(val > borders[borderId]);
                }
                writePtr += docCount;
            }
        }
        result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }

    template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloats_AVX2_BS256(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        const float nanSubstitutionValue = 0.0f
    ) {
        const __m256 substitutionValVec = _mm256_set1_ps(nanSubstitutionValue);
        const auto docCount32 = docCount / 32 * 32;
        for (size_t docId = 0; docId < docCount32; docId += 32) {
            alignas(0x20) const float val[32] = {
                FLOAT_ACCESSOR_BLOCK_4(0)
                FLOAT_ACCESSOR_BLOCK_4(4)
                FLOAT_ACCESSOR_BLOCK_4(1)
                FLOAT_ACCESSOR_BLOCK_4(5)
                FLOAT_ACCESSOR_BLOCK_4(2)
                FLOAT_ACCESSOR_BLOCK_4(6)
                FLOAT_ACCESSOR_BLOCK_4(3)
                FLOAT_ACCESSOR_BLOCK_4(7)
            };
            const __m256i mask = _mm256_set1_epi8(1);
            __m256 floats0 = _mm256_load_ps(val);
            __m256 floats1 = _mm256_load_ps(val + 8);
            __m256 floats2 = _mm256_load_ps(val + 16);
            __m256 floats3 = _mm256_load_ps(val + 24);

            if (UseNanSubstitution) {
                {
                    __m128 part0 = _mm256_extractf128_ps(floats0, 0);
                    __m128 part1 = _mm256_extractf128_ps(floats0, 1);
                    const __m256 masks = _mm256_setr_m128(
                        _mm_cmpunord_ps(part0, part0),
                        _mm_cmpunord_ps(part1, part1)
                    );
                    floats0 = _mm256_or_ps(_mm256_andnot_ps(masks, floats0), _mm256_and_ps(masks, substitutionValVec));
                }
                {
                    __m128 part0 = _mm256_extractf128_ps(floats1, 0);
                    __m128 part1 = _mm256_extractf128_ps(floats1, 1);
                    const __m256 masks = _mm256_setr_m128(
                        _mm_cmpunord_ps(part0, part0),
                        _mm_cmpunord_ps(part1, part1)
                    );
                    floats1 = _mm256_or_ps(_mm256_andnot_ps(masks, floats1), _mm256_and_ps(masks, substitutionValVec));
                }
                {
                    __m128 part0 = _mm256_extractf128_ps(floats2, 0);
                    __m128 part1 = _mm256_extractf128_ps(floats2, 1);
                    const __m256 masks = _mm256_setr_m128(
                        _mm_cmpunord_ps(part0, part0),
                        _mm_cmpunord_ps(part1, part1)
                    );
                    floats2 = _mm256_or_ps(_mm256_andnot_ps(masks, floats2), _mm256_and_ps(masks, substitutionValVec));
                }
                {
                    __m128 part0 = _mm256_extractf128_ps(floats3, 0);
                    __m128 part1 = _mm256_extractf128_ps(floats3, 1);
                    const __m256 masks = _mm256_setr_m128(
                        _mm_cmpunord_ps(part0, part0),
                        _mm_cmpunord_ps(part1, part1)
                    );
                    floats3 = _mm256_or_ps(_mm256_andnot_ps(masks, floats3), _mm256_and_ps(masks, substitutionValVec));
                }
            }
            ui8* writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                __m256i resultVec = _mm256_setzero_si256();
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    const __m256 borderVec = _mm256_set1_ps(borders[borderId]);
                    const __m256i r0 = _mm256_castps_si256(_mm256_cmp_ps(floats0, borderVec, _CMP_GT_OS));
                    const __m256i r1 = _mm256_castps_si256(_mm256_cmp_ps(floats1, borderVec, _CMP_GT_OS));
                    const __m256i r2 = _mm256_castps_si256(_mm256_cmp_ps(floats2, borderVec, _CMP_GT_OS));
                    const __m256i r3 = _mm256_castps_si256(_mm256_cmp_ps(floats3, borderVec, _CMP_GT_OS));
                    const __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(r0, r1), _mm256_packs_epi32(r2, r3));
                    resultVec = _mm256_add_epi8(resultVec, _mm256_and_si256(packed, mask));
                }
                _mm256_storeu_si256((__m256i *)writePtr, resultVec);
                writePtr += docCount;
            }
        }
        for (size_t docId = docCount32; docId < docCount; ++docId) {
            BinarizeFloatsNonSseSingle<UseNanSubstitution, TFloatFeatureAccessor>(
                position,
                docCount,
                floatAccessor,
                borders,
                start,
                result,
                docId,
                nanSubstitutionValue
            );
        }
        result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }

    template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloats_AVX512_BS512(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        const float nanSubstitutionValue = 0.0f
    ) {
        const __m512 substitutionValVec = _mm512_set1_ps(nanSubstitutionValue);
        const auto docCount64 = docCount / 64 * 64;
        for (size_t docId = 0; docId < docCount64; docId += 64) {
            alignas(0x40) const float val[64] = {
                FLOAT_ACCESSOR_BLOCK_4(0)
                FLOAT_ACCESSOR_BLOCK_4(4)
                FLOAT_ACCESSOR_BLOCK_4(8)
                FLOAT_ACCESSOR_BLOCK_4(12)
                FLOAT_ACCESSOR_BLOCK_4(1)
                FLOAT_ACCESSOR_BLOCK_4(5)
                FLOAT_ACCESSOR_BLOCK_4(9)
                FLOAT_ACCESSOR_BLOCK_4(13)
                FLOAT_ACCESSOR_BLOCK_4(2)
                FLOAT_ACCESSOR_BLOCK_4(6)
                FLOAT_ACCESSOR_BLOCK_4(10)
                FLOAT_ACCESSOR_BLOCK_4(14)
                FLOAT_ACCESSOR_BLOCK_4(3)
                FLOAT_ACCESSOR_BLOCK_4(7)
                FLOAT_ACCESSOR_BLOCK_4(11)
                FLOAT_ACCESSOR_BLOCK_4(15)
            };
            const __m512i mask = _mm512_set1_epi8(1);
            __m512 floats0 = _mm512_load_ps(val);
            __m512 floats1 = _mm512_load_ps(val + 16 * 1);
            __m512 floats2 = _mm512_load_ps(val + 16 * 2);
            __m512 floats3 = _mm512_load_ps(val + 16 * 3);

            if (UseNanSubstitution) {
                {
                    const __mmask16 masks = _mm512_cmpunord_ps_mask(floats0, floats0);
                    floats0 = _mm512_mask_blend_ps(masks, substitutionValVec, floats0);
                }
                {
                    const __mmask16 masks = _mm512_cmpunord_ps_mask(floats1, floats1);
                    floats1 = _mm512_mask_blend_ps(masks, substitutionValVec, floats1);
                }
                {
                    const __mmask16 masks = _mm512_cmpunord_ps_mask(floats2, floats2);
                    floats2 = _mm512_mask_blend_ps(masks, substitutionValVec, floats2);
                }
                {
                    const __mmask16 masks = _mm512_cmpunord_ps_mask(floats3, floats3);
                    floats3 = _mm512_mask_blend_ps(masks, substitutionValVec, floats3);
                }
            }
            ui8* writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
                __m512i resultVec = _mm512_setzero_si512();
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    const __m512 borderVec = _mm512_set1_ps(borders[borderId]);
                    const __m512i r0 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats0, borderVec, _CMP_GT_OS), 1);
                    const __m512i r1 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats1, borderVec, _CMP_GT_OS), 1);
                    const __m512i r2 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats2, borderVec, _CMP_GT_OS), 1);
                    const __m512i r3 = _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats3, borderVec, _CMP_GT_OS), 1);
                    const __m512i packed = _mm512_packs_epi16(_mm512_packs_epi32(r0, r1), _mm512_packs_epi32(r2, r3));
                    resultVec = _mm512_add_epi8(resultVec, _mm512_and_si512(packed, mask));
                }
                _mm512_storeu_si512((__m512i *)writePtr, resultVec);
                writePtr += docCount;
            }
        }
        for (size_t docId = docCount64; docId < docCount; ++docId) {
            BinarizeFloatsNonSseSingle<UseNanSubstitution, TFloatFeatureAccessor>(
                position,
                docCount,
                floatAccessor,
                borders,
                start,
                result,
                docId,
                nanSubstitutionValue
            );
        }
        result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }

#undef FLOAT_ACCESSOR_BLOCK_4

    template <EEvaluationMethod EvaluationMethod, bool UseNanSubstitution, typename TFloatFeatureAccessor>
    Y_FORCE_INLINE void BinarizeFloats(
        TFeaturePosition position,
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        const float nanSubstitutionValue = 0.0f
    ) {
        switch (EvaluationMethod) {
            case EEvaluationMethod::NON_SSE:
            case EEvaluationMethod::NON_SSE_FP16:
                BinarizeFloatsNonSse<UseNanSubstitution, TFloatFeatureAccessor>(
                    position,
                    docCount,
                    floatAccessor,
                    borders,
                    start,
                    result,
                    nanSubstitutionValue
                );
                return;
            case EEvaluationMethod::SSE3_BS128:
                BinarizeFloats_SSE3_BS128<UseNanSubstitution, TFloatFeatureAccessor>(
                    position,
                    docCount,
                    floatAccessor,
                    borders,
                    start,
                    result,
                    nanSubstitutionValue
                );
                return;
            case EEvaluationMethod::AVX2_BS256:
            case EEvaluationMethod::AVX2_BS128:
            case EEvaluationMethod::AVX2_BS64:
                BinarizeFloats_AVX2_BS256<UseNanSubstitution, TFloatFeatureAccessor>(
                    position,
                    docCount,
                    floatAccessor,
                    borders,
                    start,
                    result,
                    nanSubstitutionValue
                );
                return;
            case EEvaluationMethod::AVX512_BS512:
            case EEvaluationMethod::AVX512_BS256:
            case EEvaluationMethod::AVX512_BS128:
            case EEvaluationMethod::AVX512_BS64:
            case EEvaluationMethod::AVX512_BS512_SHUFFLE:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE:
            case EEvaluationMethod::AVX2_BS256_GATHER:
            case EEvaluationMethod::AVX2_BS128_GATHER:
            case EEvaluationMethod::AVX512_BS512_GATHER:
            case EEvaluationMethod::AVX512_BS256_GATHER:
            case EEvaluationMethod::AVX512_BS128_GATHER:
            case EEvaluationMethod::AVX512_BS512_FP16:
            case EEvaluationMethod::AVX512_BS256_FP16:
            case EEvaluationMethod::AVX512_BS128_FP16:
            case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
            case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
                BinarizeFloats_AVX512_BS512<UseNanSubstitution, TFloatFeatureAccessor>(
                    position,
                    docCount,
                    floatAccessor,
                    borders,
                    start,
                    result,
                    nanSubstitutionValue
                );
                return;
        }
    };


    // TCatFeatureAccessor must return hashed cat feature values
    template <EEvaluationMethod EvaluationMethod, typename TCatFeatureAccessor>
    inline void ComputeOneHotAndCtrFeaturesForBlock(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        TCatFeatureAccessor catFeatureAccessor,
        size_t start,
        size_t docCount,
        ui8* resultPtrForBlockStart,
        TArrayRef<ui32> transposedHash,
        TArrayRef<float> ctrs,
        ui8** resultPtr,
        const TFeatureLayout* featureInfo = nullptr
    ) {
        if (applyData.UsedCatFeaturesCount != 0) {
            THashMap<int, int> catFeaturePackedIndexes;
            int usedFeatureIdx = 0;
            for (const auto& catFeature : trees.GetCatFeatures()) {
                if (!catFeature.UsedInModel()) {
                    continue;
                }
                catFeaturePackedIndexes[catFeature.Position.Index] = usedFeatureIdx;
                TFeaturePosition position = catFeature.Position;
                if (featureInfo) {
                    position = featureInfo->GetRemappedPosition(catFeature);
                }
                for (size_t docId = 0, writeIdx = usedFeatureIdx * docCount;
                     docId < docCount;
                     ++docId, ++writeIdx) {
                    transposedHash[writeIdx] = catFeatureAccessor(position, start + docId);
                }
                ++usedFeatureIdx;
            }
            Y_ASSERT(applyData.UsedCatFeaturesCount == (size_t)usedFeatureIdx);
            OneHotBinsFromTransposedCatFeatures(
                trees.GetOneHotFeatures(),
                catFeaturePackedIndexes,
                docCount,
                transposedHash,
                *resultPtr
            );
            if (!applyData.UsedModelCtrs.empty()) {
                ctrProvider->CalcCtrs(
                    applyData.UsedModelCtrs,
                    TConstArrayRef<ui8>(resultPtrForBlockStart, docCount * trees.GetEffectiveBinaryFeaturesBucketsCount()),
                    transposedHash,
                    docCount,
                    ctrs
                );
            }
            size_t ctrFloatsPosition = 0;
            for (const auto& ctr : trees.GetCtrFeatures()) {
                auto ctrFloatsPtr = &ctrs[ctrFloatsPosition];
                ctrFloatsPosition += docCount;
                BinarizeFloats<EvaluationMethod, false>(
                    TFeaturePosition(),
                    docCount,
                    [ctrFloatsPtr](TFeaturePosition, size_t index) { return ctrFloatsPtr[index]; },
                    ctr.Borders,
                    0,
                    *resultPtr
                );
            }
        }
    }


/**
* This function binarizes
*/
    template <EEvaluationMethod EvaluationMethod,
              typename TFloatFeatureAccessor, typename TCatFeatureAccessor,
              typename TTextFeatureAccessor, typename TEmbeddingFeatureAccessor>
    inline void BinarizeFeatures(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
        const TIntrusivePtr<TEmbeddingProcessingCollection>& embeddingProcessingCollection,
        TFloatFeatureAccessor floatAccessor,
        TCatFeatureAccessor catFeatureAccessor,
        TTextFeatureAccessor textFeatureAccessor,
        TEmbeddingFeatureAccessor embeddingFeatureAccessor,
        size_t start,
        size_t end,
        TCPUEvaluatorQuantizedData<EvaluationMethod>* cpuEvaluatorQuantizedData,
        TArrayRef<ui32> transposedHash,
        TArrayRef<float> ctrs,
        TArrayRef<float> estimatedFeatures,
        const TFeatureLayout* featureInfo = nullptr
    ) {
        const auto fullDocCount = end - start;
        auto &result = cpuEvaluatorQuantizedData->QuantizedData;
        auto expectedQuantizedFeaturesLen = trees.GetEffectiveBinaryFeaturesBucketsCount() * fullDocCount;
        CB_ENSURE(result.size() >= expectedQuantizedFeaturesLen, "Not enough space to store quantized features");
        cpuEvaluatorQuantizedData->BlocksCount = 0;
        cpuEvaluatorQuantizedData->BlockStride =
            trees.GetEffectiveBinaryFeaturesBucketsCount() * IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE;
        cpuEvaluatorQuantizedData->ObjectsCount = fullDocCount;
        ui8* resultPtr = result.data();
        std::fill(result.begin(), result.begin() + expectedQuantizedFeaturesLen, 0);
        for (; start < end; start += IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE) {
            ui8* resultPtrForBlockStart = resultPtr;
            ++cpuEvaluatorQuantizedData->BlocksCount;
            auto docCount = Min(end - start, IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE);
            for (const auto& floatFeature : trees.GetFloatFeatures()) {
                if (!floatFeature.UsedInModel()) {
                    continue;
                }
                TFeaturePosition position = floatFeature.Position;
                if (featureInfo) {
                    position = featureInfo->GetRemappedPosition(floatFeature);
                }
                if (!floatFeature.HasNans ||
                    floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsIs) {
                    BinarizeFloats<EvaluationMethod, false>(
                        position,
                        docCount,
                        floatAccessor,
                        floatFeature.Borders,
                        start,
                        resultPtr
                    );
                } else {
                    const float infinity = std::numeric_limits<float>::infinity();
                    if (floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsFalse) {
                        BinarizeFloats<EvaluationMethod, true>(
                            position,
                            docCount,
                            floatAccessor,
                            floatFeature.Borders,
                            start,
                            resultPtr,
                            -infinity
                        );
                    } else {
                        Y_ASSERT(floatFeature.NanValueTreatment == TFloatFeature::ENanValueTreatment::AsTrue);
                        BinarizeFloats<EvaluationMethod, true>(
                            position,
                            docCount,
                            floatAccessor,
                            floatFeature.Borders,
                            start,
                            resultPtr,
                            infinity
                        );
                    }
                }
            }
            if (applyData.UsedEstimatedFeaturesCount > 0) {
                CB_ENSURE(
                    textProcessingCollection || applyData.UsedTextFeaturesCount == 0,
                    "Fail to apply with text features: TextProcessingCollection must present in FullModel"
                );
                CB_ENSURE(
                    embeddingProcessingCollection || applyData.UsedEmbeddingFeaturesCount == 0,
                    "Fail to apply with embedding features: EmbeddingProcessingCollection must present in FullModel"
                );

                ui32 textFeaturesTotalNum = 0;

                if (textProcessingCollection) {
                    textFeaturesTotalNum = textProcessingCollection->TotalNumberOfOutputFeatures();

                    TVector<ui32> textFeatureIds;
                    THashMap<ui32, ui32> textFeatureIdToFlatIndex;
                    // TODO(d-kruchinin) Check out how index recalculation affects the speed
                    for (const auto& textFeature : trees.GetTextFeatures()) {
                        if (!textFeature.UsedInModel()) {
                            continue;
                        }
                        TFeaturePosition position = textFeature.Position;
                        if (featureInfo) {
                            position = featureInfo->GetRemappedPosition(textFeature);
                        }
                        textFeatureIds.push_back(position.Index);
                        textFeatureIdToFlatIndex[position.Index] = position.FlatIndex;
                    }

                    textProcessingCollection->CalcFeatures(
                        [start, &textFeatureAccessor, &textFeatureIdToFlatIndex](ui32 textFeatureId, ui32 docId) {
                            return textFeatureAccessor(
                                TFeaturePosition{
                                    SafeIntegerCast<int>(textFeatureId),
                                    SafeIntegerCast<int>(textFeatureIdToFlatIndex[textFeatureId])
                                },
                                start + docId
                            );
                        },
                        MakeConstArrayRef(textFeatureIds),
                        docCount,
                        estimatedFeatures
                    );
                }

                if (embeddingProcessingCollection) {
                    TVector<ui32> embeddingFeatureIds;
                    THashMap<ui32, ui32> embeddingFeatureIdToFlatIndex;
                    for (const auto& embeddingFeature : trees.GetEmbeddingFeatures()) {
                        if (!embeddingFeature.UsedInModel()) {
                            continue;
                        }
                        TFeaturePosition position = embeddingFeature.Position;
                        if (featureInfo) {
                            position = featureInfo->GetRemappedPosition(embeddingFeature);
                        }
                        embeddingFeatureIds.push_back(position.Index);
                        embeddingFeatureIdToFlatIndex[position.Index] = position.FlatIndex;
                    }
                    embeddingProcessingCollection->CalcFeatures(
                        [start, &embeddingFeatureAccessor, &embeddingFeatureIdToFlatIndex](ui32 embeddingFeatureId,
                                                                                           ui32 docId) {
                            return embeddingFeatureAccessor(
                                TFeaturePosition{
                                    SafeIntegerCast<int>(embeddingFeatureId),
                                    SafeIntegerCast<int>(embeddingFeatureIdToFlatIndex[embeddingFeatureId])
                                },
                                start + docId
                            );
                        },
                        MakeConstArrayRef(embeddingFeatureIds),
                        docCount,
                        MakeArrayRef(estimatedFeatures.begin() + textFeaturesTotalNum * docCount,
                                     estimatedFeatures.end())
                    );
                }

                for (auto estimatedFeature : trees.GetEstimatedFeatures()) {
                    ui32 reorderedOffset;
                    if (estimatedFeature.ModelEstimatedFeature.SourceFeatureType == EEstimatedSourceFeatureType::Embedding) {
                        reorderedOffset = textFeaturesTotalNum
                            + embeddingProcessingCollection->GetAbsoluteCalcerOffset(estimatedFeature.ModelEstimatedFeature.CalcerId)
                            + estimatedFeature.ModelEstimatedFeature.LocalId;
                    } else {
                        reorderedOffset =
                            textProcessingCollection->GetAbsoluteCalcerOffset(estimatedFeature.ModelEstimatedFeature.CalcerId)
                            + estimatedFeature.ModelEstimatedFeature.LocalId;
                    }

                    auto estimatedFeaturePtr = &estimatedFeatures[reorderedOffset * docCount];

                    BinarizeFloats<EvaluationMethod, false>(
                        TFeaturePosition(),
                        docCount,
                        [estimatedFeaturePtr](TFeaturePosition, size_t index) {
                            return estimatedFeaturePtr[index];
                        },
                        MakeConstArrayRef(estimatedFeature.Borders),
                        0,
                        resultPtr
                    );
                }
            }
            ComputeOneHotAndCtrFeaturesForBlock<EvaluationMethod>(
                trees,
                applyData,
                ctrProvider,
                catFeatureAccessor,
                start,
                docCount,
                resultPtrForBlockStart,
                transposedHash,
                ctrs,
                &resultPtr,
                featureInfo
            );
        }
    }

}
