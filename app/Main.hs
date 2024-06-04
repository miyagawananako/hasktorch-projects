{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (when)
import Torch as T hiding (take, div, index)
import qualified Data.ByteString.Lazy as BL
import GHC.Generics (Generic)
import Data.ByteString.Char8 as C hiding (map, putStrLn, take, tail, filter, length, drop, unzip, index)
import Data.Vector as V hiding ((++), map, take, tail, filter, length, drop, unzip)
import Data.Csv (FromNamedRecord, (.:), parseNamedRecord, decodeByName)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools
import System.Random.Shuffle (shuffleM)

data WeatherData = WeatherData
  { date :: !ByteString
  , daily_mean_temperature :: !Float
  } deriving (Generic, Show)

instance FromNamedRecord WeatherData where
    parseNamedRecord r = WeatherData <$> r .: "date" <*> r .: "daily_mean_temperature"

extractTemperatures :: V.Vector WeatherData -> [Float]
extractTemperatures vector_weatherdata =
  let weatherList = V.toList vector_weatherdata
  in map daily_mean_temperature weatherList

createPairedData :: [Float] -> [([Float], Float)]
createPairedData temperatureList = [(take 7 (drop i temperatureList), temperatureList !! (i+7)) | i <- [0..(length temperatureList - 8)]]

readTemperaturesFromFile :: FilePath -> IO [([Float], Float)]
readTemperaturesFromFile path = do
  csvData <- BL.readFile path
  case decodeByName csvData of
      Left err -> error err
      Right (_, v) -> return (createPairedData $ extractTemperatures v)

trainingTemperatures :: IO [([Float], Float)]
trainingTemperatures = readTemperaturesFromFile "data/train.csv"

validTemperatures :: IO [([Float], Float)]
validTemperatures = readTemperaturesFromFile "data/valid.csv"

evalTemperatures :: IO [([Float], Float)]
evalTemperatures = readTemperaturesFromFile "data/eval.csv"

model :: T.Linear -> T.Tensor -> T.Tensor
model state input = squeezeAll $ linear state input

printParams :: T.Linear -> IO ()
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)

-- pop :: [a] -> (a, [a])
-- pop (x:xs) = (x, xs)

main :: IO ()
main = do
  trainingData <- trainingTemperatures
  print $ take 5 trainingData 

  -- validData <- validTemperatures
  -- print $ take 5 validData

  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1}  -- 線形モデルの初期パラメータ。inとoutは入出力の特徴の数
  printParams init

  (trained, losses) <- foldLoop (init, []) numIters $ \(state, losses) i -> do  -- ループでは現在の状態(state, randGen)とイテレーションiが与えられる
    initRandamTrainData <- shuffleM trainingData
    (trained', lossValue, loss, _) <- foldLoop (state, 0, T.zeros' [1,1], initRandamTrainData) ((length trainingData) `div` batchsize) $ \(state', _, _, randamTrainData) j -> do  -- ループでは現在の状態(state')とイテレーションjが与えられる
        let index = (j - 1) * batchsize
            dataList = take batchsize $ drop index randamTrainData
            (inputData, targetData) = unzip dataList
            input = asTensor inputData :: T.Tensor
            target = asTensor targetData :: T.Tensor
            (y, y') = (target, model state' input)  -- 真の出力yとモデルの予想出力y'を計算する
            newLoss = mseLoss y y'  -- 平均二乗誤差を計算してlossに束縛   
        (newParam, _) <- runStep state' optimizer newLoss 1e-6 -- パラメータ更新タイミングはバッチごと！
        pure (newParam, asValue newLoss, newLoss, randamTrainData)  -- 新しいパラメータとlossを返す

    when (i `mod` 50 == 0) $ do
          putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss 
    pure (trained', losses ++ [lossValue]) -- epochごとにlossを足していけばいい

  printParams trained
  drawLearningCurve "data/graph-weather.png" "Learning Curve" [("Training", losses)]
  pure ()
  where
    optimizer = GD  -- 勾配降下法を使う
    numIters = 300  -- 何回ループさせて学習させるか
    batchsize = 64  -- バッチサイズ
    numFeatures = 7