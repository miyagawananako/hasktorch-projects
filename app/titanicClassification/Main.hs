{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import EvaluateScores (evaluateAccuracy, evaluatePrecision, evaluateRecall, confusionMatrix, evaluateF1Score)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.Vector as V hiding (catMaybes)
import Data.Maybe
import Control.Applicative ((<|>))
import System.Random.Shuffle (shuffleM)

import Prelude hiding (tanh) 
--hasktorch
import Torch.Tensor       (asValue)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (sample)
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools
import Torch.Train (saveParams, loadParams)

-- PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
data Passenger = Passenger {
  passengerId :: Maybe Float,
  survived :: Maybe Float,
  pclass :: Maybe Float,
  sex :: Maybe Float,  -- male, femaleを0.0, 1.0に変換する
  age :: Maybe Float,
  sibSp :: Maybe Float,
  parch :: Maybe Float,
  fare :: Maybe Float,
  embarked :: Maybe Float  -- Q, S, Cを0.0, 1.0, 2.0に変換する
} deriving Show

-- SexをFloatに変換する関数
sexToFloat :: BL.ByteString -> Maybe Float
sexToFloat "male" = Just 0.0
sexToFloat "female" = Just 1.0
sexToFloat _ = Nothing

-- EmbarkedをFloatに変換する関数
embarkedToFloat :: BL.ByteString -> Maybe Float
embarkedToFloat "Q" = Just 0.0
embarkedToFloat "S" = Just 1.0
embarkedToFloat "C" = Just 2.0
embarkedToFloat _ = Nothing

instance Csv.FromNamedRecord Passenger where
    parseNamedRecord r = Passenger <$> r Csv..: "PassengerId"
                                   <*> (r Csv..: "Survived" <|> pure Nothing)
                                   <*> r Csv..: "Pclass"
                                   <*> (sexToFloat <$> r Csv..: "Sex")
                                   <*> r Csv..: "Age"
                                   <*> r Csv..: "SibSp"
                                   <*> r Csv..: "Parch"
                                   <*> r Csv..: "Fare"
                                   <*> (embarkedToFloat <$> r Csv..: "Embarked")


-- Passengerのage, fare以外のデータが全て揃っているかどうかを判定する関数
isComplete :: Passenger -> Bool
isComplete Passenger{..} = all isJust [survived, pclass, sex, sibSp, parch, embarked]

readDataFromFile :: FilePath -> IO (V.Vector Passenger)
readDataFromFile path = do
  csvData <- BL.readFile path
  case Csv.decodeByName csvData of
    Left err -> do
      putStrLn err
      return V.empty -- 空のベクトルを返す
    Right (_, v) -> return $ V.filter isComplete v

calculateAverageAge :: V.Vector Passenger -> Float
calculateAverageAge passengers = 
  let ages = catMaybes $ V.toList $ fmap age passengers
      totalAge = sum ages
      count = fromIntegral $ length ages
  in if count > 0 then totalAge / count else 0.0

calculateAverageFare :: V.Vector Passenger -> Float
calculateAverageFare passengers = 
  let fares = catMaybes $ V.toList $ fmap fare passengers
      totalFare = sum fares
      count = fromIntegral $ length fares
  in if count > 0 then totalFare / count else 0.0

-- Passengerの入力に使う値をFloatのリストに変換する関数（ageが欠損していたらaverageAgeを代入）
passengerToFloatList :: Float -> Float -> Passenger -> [Float]
passengerToFloatList averageAge averageFare Passenger{..} =
  let age' = fromMaybe averageAge age
      fare' = fromMaybe averageFare fare
  in catMaybes [pclass, sex, Just age', sibSp, parch, Just fare', embarked]

-- Passengerを([Float], Float)のペアに変換する関数
passengerToPair :: Float -> Float -> Passenger -> ([Float], Float)
passengerToPair averageAge averageFare p = (passengerToFloatList averageAge averageFare p, fromMaybe 0.0 (survived p))

-- Passengerのベクトルを([Float], Float)のペアのリストに変換する関数（ageが欠損していたらaverageAgeを代入）
createPairList :: V.Vector Passenger -> Float -> Float -> [([Float], Float)]
createPairList passengers averageAge averageFare = map (passengerToPair averageAge averageFare) $ V.toList passengers

-- test
readDataFromTestFile :: FilePath -> IO (V.Vector Passenger)
readDataFromTestFile path = do
  csvData <- BL.readFile path
  case Csv.decodeByName csvData of
    Left err -> do
      putStrLn err
      return V.empty -- 空のベクトルを返す
    Right (_, v) -> return v

-- Passengerを([Float], Float)のペアに変換する関数(passengerId, [Float])
passengerToTestPair :: Float -> Float -> Passenger -> (Float, [Float])
passengerToTestPair averageAge averageFare p = (fromMaybe 0.0 (passengerId p), passengerToFloatList averageAge averageFare p)

-- Passengerのベクトルを([Float], Float)のペアのリストに変換する関数
createTestPairList :: V.Vector Passenger -> Float -> Float -> [(Float, [Float])]
createTestPairList passengers averageAge averageFare = map (passengerToTestPair averageAge averageFare) $ V.toList passengers

main :: IO ()
main = do
  inputVectorData <- readDataFromFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/data/train.csv"
  let averageAge = calculateAverageAge inputVectorData
      averageFare = calculateAverageFare inputVectorData
  pairData <- shuffleM (createPairList inputVectorData averageAge averageFare)
  let (trainingData, validData) = splitAt (length pairData * 8 `div` 10) pairData
  print (length trainingData)
  print (length validData)

  testVectorData <- readDataFromTestFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/data/test.csv"

  let iter = 1500::Int -- 訓練のイテレーター数
      device = Device CPU 0  -- 使用するデバイス
      hypParams = MLPHypParams device 7 [(8,Sigmoid),(1,Sigmoid)]  -- ニューラルネットワークのハイパーパラメータ。入力層のノード数は7。隠れ層の8ノードのSigmoid活性化関数。出力層は1ノードのSigmoid活性化関数を持つMLPを定義している。
  initModel <- sample hypParams  -- hyperParamsに従って、初期モデルをサンプリングする。
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do  -- 各エポックでモデルを更新し、損失を蓄積。
    let loss = sumTensors $ for trainingData $ \(input,groundTruth) ->
                  let y = asTensor'' device groundTruth
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'  -- 平均二乗誤差を計算
        lossValue = (asValue loss) / (fromIntegral (length trainingData) :: Float)
    showLoss 10 epoc lossValue  -- エポック数と損失数を表示。10は表示の間隔。
    u <- update model opt loss 1e-5  -- モデルを更新する
    let validLoss = sumTensors $ for validData $ \(input,groundTruth) ->
                  let y = asTensor'' device groundTruth
                      y' = mlpLayer (fst u) $ asTensor'' device input
                  in mseLoss y y'
        validLossValue = (asValue validLoss) / (fromIntegral (length validData) :: Float)
    return (u, (lossValue, validLossValue))  --更新されたモデルと損失値を返す

  saveParams trainedModel "app/titanicClassification/model.pt"
  model <- loadParams hypParams "app/titanicClassification/model.pt"

  let (trainLosses, validLosses) = unzip losses   -- lossesを分解する
  drawLearningCurve "/home/acf16408ip/hasktorch-projects/app/titanicClassification/curve/graph-titanic.png" "Learning Curve" [("Training", reverse trainLosses), ("Validation", reverse validLosses)]

  -- モデルの評価
  let validData'' = map (\(input, groundTruth) -> (asTensor'' device input, asTensor'' device groundTruth)) validData
  let matrix = confusionMatrix model validData''
  print matrix
  let tp = fromIntegral $ matrix !! 0 !! 0
      fn = fromIntegral $ matrix !! 0 !! 1
      fp = fromIntegral $ matrix !! 1 !! 0
      tn = fromIntegral $ matrix !! 1 !! 1
  let accuracy = evaluateAccuracy tp fp tn fn
  let precision = evaluatePrecision tp fp 
  let recall = evaluateRecall tp fn
  let f1Score = evaluateF1Score precision recall
  print accuracy
  print precision
  print recall
  print f1Score

  -- テストデータの予測
  let testPairData = createTestPairList testVectorData averageAge averageFare
  let testResult = for testPairData $ \(passengerId, input) ->
        let y' = mlpLayer trainedModel $ asTensor'' device input
            passengerId' = round passengerId
            survived' = if (asValue y'::Float) > 0.5 then 1 else 0
        in (passengerId' :: Int, survived' :: Int)

-- headerを追加して、CSVファイルに書き込む
  let csvData = Csv.encode $ for testResult $ \(passengerId, survived) ->
        [passengerId, survived]
  
  BL.writeFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/submission.csv" csvData

  where for = flip map  -- map関数の引数の順序を反転したもの