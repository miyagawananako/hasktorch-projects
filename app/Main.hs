{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (when)
import Torch as T hiding (take)
import qualified Data.ByteString.Lazy as BL
import GHC.Generics (Generic)
import Data.ByteString.Char8 as C hiding (map, putStrLn, take, tail, filter, length, drop)
import Data.Vector as V hiding ((++), map, take, tail, filter, length, drop)
import Data.Csv (FromNamedRecord, (.:), parseNamedRecord, decodeByName)

data WeatherData = WeatherData
  { date :: !ByteString
  , daily_mean_temprature :: !Float
  } deriving (Generic, Show)

instance FromNamedRecord WeatherData where
    parseNamedRecord r = WeatherData <$> r .: "date" <*> r .: "daily_mean_temprature"

-- WeatherData型を受け取ったらdaily_mean_tempratureを返す 
return_daily_mean_temprature :: WeatherData -> Float
return_daily_mean_temprature = daily_mean_temprature

-- Vector WeatherDataを受け取ったらFloatのリストを返す
make_list :: (V.Vector WeatherData) -> [Float]
make_list vector_weatherdata =
  let tempature_list = V.toList vector_weatherdata
  in map return_daily_mean_temprature tempature_list

readFromFile :: FilePath -> IO [Float]
readFromFile path = do
  csvData <- BL.readFile path
  case decodeByName csvData of
      Left err -> do
          putStrLn err
          return []
      Right (_, v) -> return (make_list v)

trainingData :: IO [Float]
trainingData = readFromFile "data/train.csv"

validData :: IO [Float]
validData = readFromFile "data/valid.csv"

evalData :: IO [Float]
evalData = readFromFile "data/eval.csv"

model :: T.Linear -> T.Tensor -> T.Tensor
model state input = squeezeAll $ linear state input

groundTruth :: T.Tensor -> T.Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)

printParams :: T.Linear -> IO ()
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)

main :: IO ()
main = do
  -- train :: [Float]
  train <- trainingData
  print $ take 5 train

  -- pairdata :: [([Float], Float)]
  let pairdata = [(take 7 (drop i train), train !! (i+7)) | i <- [0..(length train - 8)]]
  print $ take 5 pairdata

  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1}
  randGen <- defaultRNG
  printParams init
  (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
    let (input, randGen') = randn' [batchSize, numFeatures] randGen
        (y, y') = (groundTruth input, model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newParam, _) <- runStep state optimizer loss 5e-3
    pure (newParam, randGen')
  printParams trained
  pure ()
  where
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000
    numFeatures = 3