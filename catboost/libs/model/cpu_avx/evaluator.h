#pragma once

#include "quantization.h"

#include <util/generic/utility.h>
#include <util/generic/vector.h>

#include <util/stream/labeled.h>
#include <util/system/platform.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <type_traits>

#include <library/cpp/sse/sse.h>

namespace NCB::NModelEvaluation::AVX {

    template<EEvaluationMethod EvaluationMethod>
    using TTreeCalcFunction = std::function<void(
        const TModelTrees& modelTrees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData<EvaluationMethod>*,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict results)>;


    template<EEvaluationMethod EvaluationMethod>
    TTreeCalcFunction<EvaluationMethod> GetCalcTreesFunction(
        const TModelTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly = false);

    template <
        typename TFloatFeatureAccessor,
        typename TCatFeatureAccessor,
        typename TFunctor
    >
    inline void ProcessDocsInBlocks(
        const TModelTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        size_t docCount,
        size_t blockSize,
        TFunctor callback,
        const NCB::NModelEvaluation::TFeatureLayout* featureInfo
    ) {
        ProcessDocsInBlocks(
            trees,
            ctrProvider,
            TIntrusivePtr<TTextProcessingCollection>(),
            TIntrusivePtr<TEmbeddingProcessingCollection>(),
            floatFeatureAccessor,
            catFeaturesAccessor,
            [](TFeaturePosition, size_t) -> TStringBuf {
                CB_ENSURE_INTERNAL(
                    false,
                    "Trying to access text data from model.Calc() interface which has no text features"
                );
                return "Undefined";
            },
            [](TFeaturePosition, size_t) -> TConstArrayRef<float> {
                CB_ENSURE_INTERNAL(
                    false,
                    "Trying to access embedding data from model.Calc() interface which has no embedding features"
                );
                return {};
            },
            docCount,
            blockSize,
            callback,
            featureInfo
        );
    }

    template <
        EEvaluationMethod EvaluationMethod,
        typename TFloatFeatureAccessor,
        typename TCatFeatureAccessor,
        typename TTextFeatureAccessor,
        typename TEmbeddingFeatureAccessor,
        typename TFunctor
    >
    inline void ProcessDocsInBlocks(
        const TModelTrees& trees,
        const TIntrusivePtr<ICtrProvider>& ctrProvider,
        const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
        const TIntrusivePtr<TEmbeddingProcessingCollection>& embeddingProcessingCollection,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        TTextFeatureAccessor textFeatureAccessor,
        TEmbeddingFeatureAccessor embeddingFeatureAccessor,
        size_t docCount,
        size_t blockSize,
        TFunctor callback,
        const NCB::NModelEvaluation::TFeatureLayout* featureInfo
    ) {
        const size_t binSlots = blockSize * trees.GetEffectiveBinaryFeaturesBucketsCount();

        TCPUEvaluatorQuantizedData<EvaluationMethod> quantizedData(binSlots);
        /*if (binSlots < 65 * 1024 - 0x80) { // 65KB of stack maximum
            quantizedData.QuantizedData = NCB::TMaybeOwningArrayHolder<ui8>::CreateNonOwning(
                MakeArrayRef(GetAligned((ui8*)(alloca(binSlots + 0x80))), binSlots));
        } else {
            TVector<ui8> binFeaturesHolder;
            binFeaturesHolder.yresize(binSlots);
            quantizedData.QuantizedData = NCB::TMaybeOwningArrayHolder<ui8>::CreateOwning(std::move(binFeaturesHolder));
        }*/

        auto applyData = trees.GetApplyData();
        TVector<ui32> transposedHash(blockSize * applyData->UsedCatFeaturesCount);
        TVector<float> ctrs(applyData->UsedModelCtrs.size() * blockSize);
        ui32 estimatedFeaturesNum = 0;
        if (textProcessingCollection) {
            estimatedFeaturesNum += textProcessingCollection->TotalNumberOfOutputFeatures();
        }
        if (embeddingProcessingCollection) {
            estimatedFeaturesNum += embeddingProcessingCollection->TotalNumberOfOutputFeatures();
        }
        TVector<float> estimatedFeatures(estimatedFeaturesNum * blockSize);

        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            BinarizeFeatures(
                trees,
                *applyData,
                ctrProvider,
                textProcessingCollection,
                embeddingProcessingCollection,
                floatFeatureAccessor,
                catFeaturesAccessor,
                textFeatureAccessor,
                embeddingFeatureAccessor,
                blockStart,
                blockStart + docCountInBlock,
                &quantizedData,
                transposedHash,
                ctrs,
                estimatedFeatures,
                featureInfo
            );
            callback(docCountInBlock, &quantizedData);
        }
    }

    template <typename T>
    void Transpose2DArray(
        TConstArrayRef<T> srcArray, // assume values are laid row by row
        size_t srcRowCount,
        size_t srcColumnCount,
        TArrayRef<T> dstArray
    ) {
        Y_ASSERT(srcArray.size() == srcRowCount * srcColumnCount);
        Y_ASSERT(srcArray.size() == dstArray.size());
        for (size_t srcRowIndex = 0; srcRowIndex < srcRowCount; ++srcRowIndex) {
            for (size_t srcColumnIndex = 0; srcColumnIndex < srcColumnCount; ++srcColumnIndex) {
                dstArray[srcColumnIndex * srcRowCount + srcRowIndex] =
                    srcArray[srcRowIndex * srcColumnCount + srcColumnIndex];
            }
        }
    }
}
