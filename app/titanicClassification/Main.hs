{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import qualified Data.Vector as V
import Data.Maybe (isJust)

-- PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
data Passenger = Passenger {
  survived :: Maybe BL.ByteString,
  pclass :: Maybe BL.ByteString,
  sex :: Maybe BL.ByteString,
  age :: Maybe BL.ByteString,
  sibSp :: Maybe BL.ByteString,
  parch :: Maybe BL.ByteString,
  fare :: Maybe BL.ByteString,
  embarked :: Maybe BL.ByteString
} deriving Show

instance Csv.FromNamedRecord Passenger where
    parseNamedRecord r = Passenger <$> r Csv..: "Survived"
                                   <*> r Csv..: "Pclass"
                                   <*> r Csv..: "Sex"
                                   <*> r Csv..: "Age"
                                   <*> r Csv..: "SibSp"
                                   <*> r Csv..: "Parch"
                                   <*> r Csv..: "Fare"
                                   <*> r Csv..: "Embarked"

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
  trainData <- readDataFromFile "data/train.csv"
  print trainData
