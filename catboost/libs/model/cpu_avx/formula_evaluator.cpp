#include <catboost/libs/model/eval_processing.h>
#include <catboost/libs/model/model.h>

#include "evaluator_interface.h"
#include "evaluator.h"

namespace NCB::NModelEvaluation::AVX {
    namespace NDetail {
        template <EEvaluationMethod EvaluationMethod,
                  typename TFloatFeatureAccessor, typename TCatFeatureAccessor,
                  typename TTextFeatureAccessor, typename TEmbeddingFeatureAccessor>
        inline void CalcGenericAVX(
            const TModelTrees& trees,
            const TModelTrees::TForApplyData& applyData,
            const TIntrusivePtr<ICtrProvider>& ctrProvider,
            const TIntrusivePtr<TTextProcessingCollection>& textProcessingCollection,
            const TIntrusivePtr<TEmbeddingProcessingCollection>& embeddingProcessingCollection,
            TFloatFeatureAccessor floatFeatureAccessor,
            TCatFeatureAccessor catFeaturesAccessor,
            TTextFeatureAccessor textFeatureAccessor,
            TEmbeddingFeatureAccessor embeddingFeatureAccessor,
            size_t docCount,
            size_t treeStart,
            size_t treeEnd,
            EPredictionType predictionType,
            TArrayRef<double> results,
            const NCB::NModelEvaluation::TFeatureLayout* featureInfo = nullptr
        ) {
            const size_t blockSize = Min(IConsts<EvaluationMethod>::FORMULA_EVALUATION_BLOCK_SIZE, docCount);
            auto calcTrees = GetCalcTreesFunction<EvaluationMethod>(trees, blockSize);
            if (trees.GetTreeCount() == 0) {
                auto biasRef = trees.GetScaleAndBias().GetBiasRef();
                if (biasRef.size() == 1) {
                    Fill(results.begin(), results.end(), biasRef[0]);
                } else {
                    for (size_t idx = 0; idx < results.size();) {
                        for (size_t dim = 0; dim < biasRef.size(); ++dim, ++idx) {
                            results[idx] = biasRef[dim];
                        }
                    }
                }
                return;
            }
            Fill(results.begin(), results.end(), 0.0);
            TVector<TCalcerIndexType> indexesVec(blockSize);
            TEvalResultProcessor resultProcessor(
                docCount,
                results,
                predictionType,
                trees.GetScaleAndBias(),
                trees.GetDimensionsCount(),
                blockSize
            );
            ui32 blockId = 0;
            ProcessDocsInBlocks<EvaluationMethod>(
                trees,
                ctrProvider,
                textProcessingCollection,
                embeddingProcessingCollection,
                floatFeatureAccessor,
                catFeaturesAccessor,
                textFeatureAccessor,
                embeddingFeatureAccessor,
                docCount,
                blockSize,
                [&] (size_t docCountInBlock, const TCPUEvaluatorQuantizedData<EvaluationMethod>* quantizedData) {
                    auto blockResultsView = resultProcessor.GetViewForRawEvaluation(blockId);
                    calcTrees(
                        trees,
                        applyData,
                        quantizedData,
                        docCountInBlock,
                        // docCount == 1 ? nullptr : indexesVec.data(),
                        indexesVec.data(),
                        treeStart,
                        treeEnd,
                        blockResultsView.data()
                    );
                    resultProcessor.PostprocessBlock(blockId, treeStart);
                    ++blockId;
                },
                featureInfo
            );
        }

        template<EEvaluationMethod EvaluationMethod>
        class TCpuAvxEvaluator final : public IModelEvaluator {
        public:
            explicit TCpuAvxEvaluator(const TFullModel& fullModel)
                : ModelTrees(fullModel.ModelTrees)
                , ApplyData(ModelTrees->GetApplyData())
                , CtrProvider(fullModel.CtrProvider)
                , TextProcessingCollection(fullModel.TextProcessingCollection)
                , EmbeddingProcessingCollection(fullModel.EmbeddingProcessingCollection)
            {}

            void SetPredictionType(EPredictionType type) override {
                PredictionType = type;
            }

            EPredictionType GetPredictionType() const override {
                return PredictionType;
            }

            void SetFeatureLayout(const TFeatureLayout& featureLayout) override {
                ExtFeatureLayout = featureLayout;
            }

            size_t GetTreeCount() const override {
                return ModelTrees->GetTreeCount();
            }

            TModelEvaluatorPtr Clone() const override {
                return new TCpuAvxEvaluator(*this);
            }

            i32 GetApproxDimension() const override {
                return ModelTrees->GetDimensionsCount();
            }

            void SetProperty(const TStringBuf propName, const TStringBuf propValue) override {
                CB_ENSURE(false, "CPU evaluator don't have any properties. Got: " << propName);
                Y_UNUSED(propValue);
            }

            void CalcFlatTransposed(
                TConstArrayRef<TConstArrayRef<float>> transposedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                CB_ENSURE(
                    ModelTrees->GetFlatFeatureVectorExpectedSize() <= transposedFeatures.size(),
                    "Not enough features provided" << LabeledOutput(ModelTrees->GetFlatFeatureVectorExpectedSize(), transposedFeatures.size())
                );
                TMaybe<size_t> docCount;
                CB_ENSURE(!ModelTrees->GetFloatFeatures().empty() || !ModelTrees->GetCatFeatures().empty(),
                          "Both float features and categorical features information are empty");
                auto getPosition = [featureInfo] (const auto& feature) -> TFeaturePosition {
                    if (!featureInfo) {
                        return feature.Position;
                    } else {
                        return featureInfo->GetRemappedPosition(feature);
                    }
                };
                if (!ModelTrees->GetFloatFeatures().empty()) {
                    for (const auto& floatFeature : ModelTrees->GetFloatFeatures()) {
                        if (floatFeature.UsedInModel()) {
                            docCount = transposedFeatures[getPosition(floatFeature).FlatIndex].size();
                            break;
                        }
                    }
                }
                if (!docCount.Defined() && !ModelTrees->GetCatFeatures().empty()) {
                    for (const auto& catFeature : ModelTrees->GetCatFeatures()) {
                        if (catFeature.UsedInModel()) {
                            docCount = transposedFeatures[getPosition(catFeature).FlatIndex].size();
                            break;
                        }
                    }
                }

                CB_ENSURE(docCount.Defined(), "couldn't determine document count, something went wrong");
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&transposedFeatures](TFeaturePosition floatFeature, size_t index) -> float {
                        return transposedFeatures[floatFeature.FlatIndex][index];
                    },
                    [&transposedFeatures](TFeaturePosition catFeature, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(transposedFeatures[catFeature.FlatIndex][index]);
                    },
                    TCpuAvxEvaluator::TextFeatureAccessorStub,
                    TCpuAvxEvaluator::EmbeddingFeatureAccessorStub,
                    *docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcFlat(
                TConstArrayRef<TConstArrayRef<float>> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                auto expectedFlatVecSize = ModelTrees->GetFlatFeatureVectorExpectedSize();
                if (featureInfo && featureInfo->FlatIndexes) {
                    CB_ENSURE(
                        featureInfo->FlatIndexes->size() >= expectedFlatVecSize,
                        "Feature layout FlatIndexes expected to be at least " << expectedFlatVecSize << " long"
                    );
                    expectedFlatVecSize = *MaxElement(featureInfo->FlatIndexes->begin(), featureInfo->FlatIndexes->end());
                }
                for (const auto& flatFeaturesVec : features) {
                    CB_ENSURE(
                        flatFeaturesVec.size() >= expectedFlatVecSize,
                        "insufficient flat features vector size: " << flatFeaturesVec.size() << " expected: " << expectedFlatVecSize
                    );
                }
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&features](TFeaturePosition position, size_t index) -> float {
                        return features[index][position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t index) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[index][position.FlatIndex]);
                    },
                    TCpuAvxEvaluator::TextFeatureAccessorStub,
                    TCpuAvxEvaluator::EmbeddingFeatureAccessorStub,
                    features.size(),
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcFlatSingle(
                TConstArrayRef<float> features,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                CB_ENSURE(
                    ModelTrees->GetFlatFeatureVectorExpectedSize() <= features.size(),
                    "Not enough features provided"
                );
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&features](TFeaturePosition position, size_t ) -> float {
                        return features[position.FlatIndex];
                    },
                    [&features](TFeaturePosition position, size_t ) -> int {
                        return ConvertFloatCatFeatureToIntHash(features[position.FlatIndex]);
                    },
                    TCpuAvxEvaluator::TextFeatureAccessorStub,
                    TCpuAvxEvaluator::EmbeddingFeatureAccessorStub,
                    1,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                CB_ENSURE(
                    ModelTrees->GetTextFeatures().empty(),
                    "Model contains text features but they aren't provided"
                );
                Calc(
                    floatFeatures,
                    catFeatures,
                    {},
                    treeStart,
                    treeEnd,
                    results,
                    featureInfo
                );
            }

            void CalcWithHashedCatAndText(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                Calc(floatFeatures, catFeatures, textFeatures, treeStart, treeEnd, results, featureInfo);
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<int>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size());
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return catFeatures[index][position.Index];
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    TCpuAvxEvaluator::EmbeddingFeatureAccessorStub,
                    docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                CB_ENSURE(
                    ModelTrees->GetTextFeatures().empty(),
                    "Model contains text features but they aren't provided"
                );
                Calc(
                    floatFeatures,
                    catFeatures,
                    {},
                    treeStart,
                    treeEnd,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size());
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    TCpuAvxEvaluator::EmbeddingFeatureAccessorStub,
                    docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void Calc(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                TConstArrayRef<TConstArrayRef<TConstArrayRef<float>>> embeddingFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results,
                const TFeatureLayout* featureInfo
            ) const {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                ValidateInputFeatures(floatFeatures, catFeatures, textFeatures, featureInfo);
                const size_t docCount = Max(catFeatures.size(), floatFeatures.size(), textFeatures.size());
                CalcGenericAVX<EvaluationMethod>(
                    *ModelTrees,
                    *ApplyData,
                    CtrProvider,
                    TextProcessingCollection,
                    EmbeddingProcessingCollection,
                    [&floatFeatures](TFeaturePosition position, size_t index) -> float {
                        return floatFeatures[index][position.Index];
                    },
                    [&catFeatures](TFeaturePosition position, size_t index) -> int {
                        return CalcCatFeatureHash(catFeatures[index][position.Index]);
                    },
                    [&textFeatures](TFeaturePosition position, size_t index) -> TStringBuf {
                        return textFeatures[index][position.Index];
                    },
                    [&embeddingFeatures](TFeaturePosition position, size_t index) -> TConstArrayRef<float> {
                        return embeddingFeatures[index][position.Index];
                    },
                    docCount,
                    treeStart,
                    treeEnd,
                    PredictionType,
                    results,
                    featureInfo
                );
            }

            void CalcLeafIndexesSingle(
                TConstArrayRef<float> floatFeatures,
                TConstArrayRef<TStringBuf> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout* featureInfo
            ) const override {
                if (!featureInfo) {
                    featureInfo = ExtFeatureLayout.Get();
                }
                CB_ENSURE(false, "This method is not implemented with AVX yet");
                Y_UNUSED(floatFeatures, catFeatures, treeStart, treeEnd, indexes, featureInfo);
            }

            void CalcLeafIndexes(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> catFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes,
                const TFeatureLayout* featureInfo
            ) const override {
                CB_ENSURE(false, "This method is not implemented with AVX yet");
                Y_UNUSED(floatFeatures, catFeatures, treeStart, treeEnd, indexes, featureInfo);
            }

            void Calc(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<double> results
            ) const override {
                CB_ENSURE(false, "This method is not implemented with AVX yet");
                Y_UNUSED(quantizedFeatures, treeStart, treeEnd, results);
            }

            void CalcLeafIndexes(
                const IQuantizedData* quantizedFeatures,
                size_t treeStart,
                size_t treeEnd,
                TArrayRef<ui32> indexes
            ) const override {
                Y_UNUSED(quantizedFeatures, treeStart, treeEnd, indexes);
                CB_ENSURE(false, "This method is not implemented with AVX yet");
            }

        void Quantize(
            TConstArrayRef<TConstArrayRef<float>> features,
            IQuantizedData* quantizedData
        ) const override {
            Y_UNUSED(features);
            Y_UNUSED(quantizedData);
            CB_ENSURE(false, "Unimplemented method called, please contact catboost developers via GitHub issue or in support chat");
        }

        private:
            template <typename TCatFeatureContainer = TConstArrayRef<int>>
            void ValidateInputFeatures(
                TConstArrayRef<TConstArrayRef<float>> floatFeatures,
                TConstArrayRef<TCatFeatureContainer> catFeatures,
                TConstArrayRef<TConstArrayRef<TStringBuf>> textFeatures,
                const TFeatureLayout* featureInfo
            ) const {
                Y_UNUSED(floatFeatures, catFeatures, textFeatures, featureInfo);
                CB_ENSURE(false, "This method is not implemented with AVX yet");
            }

            static TStringBuf TextFeatureAccessorStub(TFeaturePosition position, size_t index) {
                Y_UNUSED(position, index);
                CB_ENSURE(false, "This type of apply interface is not implemented with text features yet");
            }

            static TConstArrayRef<float> EmbeddingFeatureAccessorStub(TFeaturePosition position, size_t index) {
                Y_UNUSED(position, index);
                CB_ENSURE(false, "This type of apply interface is not implemented with embedding features yet");
            }
        private:
            TCOWTreeWrapper ModelTrees;
            TAtomicSharedPtr<TModelTrees::TForApplyData> ApplyData;
            const TIntrusivePtr<ICtrProvider> CtrProvider;
            const TIntrusivePtr<TTextProcessingCollection> TextProcessingCollection;
            const TIntrusivePtr<TEmbeddingProcessingCollection> EmbeddingProcessingCollection;
            EPredictionType PredictionType = EPredictionType::RawFormulaVal;
            TMaybe<TFeatureLayout> ExtFeatureLayout;
        };
    }

    TModelEvaluatorPtr CreateAvxEvaluator(EEvaluationMethod method, const TFullModel& fullModel) {
        switch (method) {
        case EEvaluationMethod::NON_SSE:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::NON_SSE>(fullModel);
        case EEvaluationMethod::SSE3_BS128:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::SSE3_BS128>(fullModel);
        case EEvaluationMethod::AVX2_BS256:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX2_BS256>(fullModel);
        case EEvaluationMethod::AVX2_BS128:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX2_BS128>(fullModel);
        case EEvaluationMethod::AVX2_BS64:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX2_BS64>(fullModel);
        case EEvaluationMethod::AVX512_BS512:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS512>(fullModel);
        case EEvaluationMethod::AVX512_BS256:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS256>(fullModel);
        case EEvaluationMethod::AVX512_BS128:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS128>(fullModel);
        case EEvaluationMethod::AVX512_BS64:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS64>(fullModel);
        case EEvaluationMethod::AVX512_BS512_SHUFFLE:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS512_SHUFFLE>(fullModel);
        case EEvaluationMethod::AVX512_BS256_SHUFFLE:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS256_SHUFFLE>(fullModel);
        case EEvaluationMethod::AVX512_BS128_SHUFFLE:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS128_SHUFFLE>(fullModel);
        case EEvaluationMethod::AVX2_BS256_GATHER:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX2_BS256_GATHER>(fullModel);
        case EEvaluationMethod::AVX2_BS128_GATHER:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX2_BS128_GATHER>(fullModel);
        case EEvaluationMethod::AVX512_BS512_GATHER:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS512_GATHER>(fullModel);
        case EEvaluationMethod::AVX512_BS256_GATHER:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS256_GATHER>(fullModel);
        case EEvaluationMethod::AVX512_BS128_GATHER:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS128_GATHER>(fullModel);
        case EEvaluationMethod::NON_SSE_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::NON_SSE_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS512_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS512_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS256_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS256_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS128_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS128_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS512_SHUFFLE_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS256_SHUFFLE_FP16>(fullModel);
        case EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16:
            return new NDetail::TCpuAvxEvaluator<EEvaluationMethod::AVX512_BS128_SHUFFLE_FP16>(fullModel);
        }
    }
}
