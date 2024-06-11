{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.Vector as V
import Data.Maybe (isJust)

-- PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
data Passenger = Passenger {
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
    parseNamedRecord r = Passenger <$> r Csv..: "Survived"
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

main :: IO ()
main = do
  trainData <- readDataFromFile "/home/acf16408ip/hasktorch-projects/app/titanicClassification/data/train.csv"
  print trainData
