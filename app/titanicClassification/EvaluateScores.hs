module EvaluateScores (evaluateAccuracy, evaluatePrecision, evaluateRecall, confusionMatrix, evaluateF1Score) where

import Torch.Tensor (Tensor, asValue)
import Torch.Layer.MLP (MLPParams, mlpLayer)

evaluateAccuracy :: Float -> Float -> Float -> Float -> Float
evaluateAccuracy tp fp tn fn = if tp + fp + tn + fn == 0 then 0 else (tp + tn) / (tp + fp + tn + fn)

evaluatePrecision :: Float -> Float -> Float
evaluatePrecision tp fp = if tp + fp == 0 then 1 else tp / (tp + fp)

evaluateRecall :: Float -> Float -> Float
evaluateRecall tp fn = if tp + fn == 0 then 0 else tp / (tp + fn)

confusionMatrix :: MLPParams -> [(Tensor, Tensor)] -> [[Int]]
confusionMatrix model dataset = matrix
  where
    actualAndPredictionPair = map (\(input, actual) -> (actual, mlpLayer model input)) dataset
    initMatrix = [[0, 0], [0, 0]]
    matrix = foldl updateMatrix initMatrix actualAndPredictionPair
    updateMatrix acc (actual, predicted) = 
      let actualClass = if (asValue actual :: Float) > 0.5 then 1 :: Int else 0 :: Int
          predictedClass = if (asValue predicted :: Float) > 0.5 then 1 :: Int else 0 :: Int
      in case (actualClass, predictedClass) of
        (1, 1) -> [[acc !! 0 !! 0 + 1, acc !! 0 !! 1], [acc !! 1 !! 0, acc !! 1 !! 1]]
        (1, 0) -> [[acc !! 0 !! 0, acc !! 0 !! 1 + 1], [acc !! 1 !! 0, acc !! 1 !! 1]]
        (0, 1) -> [[acc !! 0 !! 0, acc !! 0 !! 1], [acc !! 1 !! 0 + 1, acc !! 1 !! 1]]
        (0, 0) -> [[acc !! 0 !! 0, acc !! 0 !! 1], [acc !! 1 !! 0, acc !! 1 !! 1 + 1]]
        _      -> acc

evaluateF1Score :: Float -> Float -> Float
evaluateF1Score precision recall = if precision + recall == 0 then 0 else 2 * precision * recall / (precision + recall)