{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.Vector as V hiding (catMaybes)
import Data.Maybe

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
import Control.Applicative ((<|>))

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


-- Passengerのデータが全て揃っているかどうかを判定する関数
isComplete :: Passenger -> Bool
isComplete Passenger{..} = all isJust [survived, pclass, sex, age, sibSp, parch, fare, embarked]

readDataFromFile :: FilePath -> IO (V.Vector Passenger)
readDataFromFile path = do
  csvData <- BL.readFile path
  case Csv.decodeByName csvData of
    Left err -> do
      putStrLn err
      return V.empty -- 空のベクトルを返す
    Right (_, v) -> return $ V.filter isComplete v

-- training
-- Passengerの入力に使う値をFloatのリストに変換する関数
passengerToFloatList :: Passenger -> [Float]
passengerToFloatList Passenger{..} = catMaybes [pclass, sex, age, sibSp, parch, fare, embarked]

-- Passengerを([Float], Float)のペアに変換する関数
passengerToPair :: Passenger -> ([Float], Float)
passengerToPair p = (passengerToFloatList p, fromMaybe 0.0 (survived p))

-- Passengerのベクトルを([Float], Float)のペアのリストに変換する関数
createPairList :: V.Vector Passenger -> [([Float], Float)]
createPairList = map passengerToPair . V.toList

-- test
isCompleteTest :: Passenger -> Bool
isCompleteTest Passenger{..} = all isJust [passengerId, pclass, sex, age, sibSp, parch, fare, embarked]

readDataFromTestFile :: FilePath -> IO (V.Vector Passenger)
readDataFromTestFile path = do
  csvData <- BL.readFile path
  case Csv.decodeByName csvData of
    Left err -> do
      putStrLn err
      return V.empty -- 空のベクトルを返す
    Right (_, v) -> return $ V.filter isCompleteTest v

-- Passengerの入力に使う値をFloatのリストに変換する関数
passengerToFloatTestList :: Passenger -> [Float]
passengerToFloatTestList Passenger{..} = catMaybes [pclass, sex, age, sibSp, parch, fare, embarked]

-- Passengerを([Float], Float)のペアに変換する関数(passengerId, [Float])
passengerToTestPair :: Passenger -> (Float, [Float])
passengerToTestPair p = (fromMaybe 0.0 (passengerId p), passengerToFloatTestList p)

-- Passengerのベクトルを([Float], Float)のペアのリストに変換する関数
createTestPairList :: V.Vector Passenger -> [(Float, [Float])]
createTestPairList = map passengerToTestPair . V.toList

main :: IO ()
main = do
  inputVectorData <- readDataFromFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/data/train.csv"
  print inputVectorData
  testVectorData <- readDataFromTestFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/data/test.csv"
  print testVectorData
  let pairData = createPairList inputVectorData
      (trainingData, validData) = splitAt (length pairData * 8 `div` 10) pairData
  print (length trainingData)
  print (length validData)

  let iter = 1500::Int -- 訓練のイテレーター数
      device = Device CPU 0  -- 使用するデバイス
      hypParams = MLPHypParams device 7 [(8,Sigmoid),(1,Sigmoid)]  -- ニューラルネットワークのハイパーパラメータ。入力層のノード数は7。隠れ層の8ノードのSigmoid活性化関数。出力層は1ノードのSigmoid活性化関数を持つMLPを定義している。
  initModel <- sample hypParams  -- hyperParamsに従って、初期モデルをサンプリングする。
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do  -- 各エポックでモデルを更新し、損失を蓄積。
    let loss = sumTensors $ for trainingData $ \(input,groundTruth) ->
                  let y = asTensor'' device groundTruth
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'  -- 平均二乗誤差を計算
        lossValue = (asValue loss)::Float  -- 消失テンソルをFloat値に変換
    let validLoss = sumTensors $ for validData $ \(input,groundTruth) ->
                  let y = asTensor'' device groundTruth
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'  -- 平均二乗誤差を計算
        validLossValue = (asValue validLoss)::Float  -- 消失テンソルをFloat値に変換
    showLoss 10 epoc lossValue  -- エポック数と損失数を表示。10は表示の間隔。
    u <- update model opt loss 1e-5  -- モデルを更新する
    return (u, (lossValue, validLossValue))  --更新されたモデルと損失値を返す

  let (trainLosses, validLosses) = unzip losses   -- lossesを分解する
  drawLearningCurve "/home/acf16408ip/hasktorch-projects/app/titanicClassification/graph-titanic.png" "Learning Curve" [("Training", reverse trainLosses), ("Validation", reverse validLosses)]
  -- print trainedModel

  let testPairData = createTestPairList testVectorData
  -- print testPairData
  let testResult = for testPairData $ \(passengerId, input) ->
        let y' = mlpLayer trainedModel $ asTensor'' device input
        in (passengerId, asValue y'::Float)
  print testResult

  where for = flip map  -- map関数の引数の順序を反転したもの