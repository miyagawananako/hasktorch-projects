module EvaluateScores (evaluateAccuracy, evaluatePrecision, evaluateRecall, confusionMatrix) where

import Torch.Tensor (Tensor)
import Torch.Layer.MLP (MLPParams)

evaluateAccuracy :: Float -> Float -> Float -> Float -> Float
evaluateAccuracy tp fp tn fn = if tp + fp + tn + fn == 0 then 0 else (tp + tn) / (tp + fp + tn + fn)

evaluatePrecision :: Float -> Float -> Float
evaluatePrecision tp fp = if tp + fp == 0 then 0 else tp / (tp + fp)

evaluateRecall :: Float -> Float -> Float
evaluateRecall tp fn = if tp + fn == 0 then 0 else tp / (tp + fn)

confusionMatrix :: MLPParams -> [(Tensor, Tensor)] -> [[Int]]
confusionMatrix model validData = []