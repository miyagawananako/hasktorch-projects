evaluateAccuracy :: Float -> Float -> Float -> Float -> Float
evaluateAccuracy tp fp tn fn = if tp + fp + tn + fn == 0 then 0 else (tp + tn) / (tp + fp + tn + fn)

evaluatePrecision :: Float -> Float -> Float
evaluatePrecision tp fp = if tp + fp == 0 then 0 else tp / (tp + fp)

evaluateRecall :: Float -> Float -> Float
evaluateRecall tp fn = if tp + fn == 0 then 0 else tp / (tp + fn)

main :: IO ()
main = do
  print $ evaluateAccuracy 1.0 2.0 3.0 4.0
  print $ evaluatePrecision 1.0 2.0
  print $ evaluateRecall 1.0 2.0