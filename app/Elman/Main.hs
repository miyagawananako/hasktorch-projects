{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad.State.Strict
import Data.List (foldl', intersperse, scanl')
import RecurrentLayer
import Torch

data ElmanSpec = ElmanSpec {in_features :: Int, hidden_features :: Int}

data ElmanCell = ElmanCell
  { input_weight :: Parameter,
    hidden_weight :: Parameter,
    bias :: Parameter
  }

-- RecurrentCell型クラスのインスタンスElmanCell
-- {..}はElmanCellのフィールドのパターンマッチング
-- nextState :: ElmanCell -> Tensor -> Tensor -> Tensor
-- gate関数が呼び出されています。この関数は、入力テンソル、隠れ状態テンソル、活性化関数（ここではTorch.tanh）、およびElmanCellのフィールド（input_weight、hidden_weight、bias）を受け取り、次の隠れ状態を計算します。Torch.tanhは、双曲線正接関数であり、ニューラルネットワークの活性化関数としてよく使用されます。
instance RecurrentCell ElmanCell where
  nextState ElmanCell {..} input hidden =
    gate input hidden Torch.tanh input_weight hidden_weight bias

-- =<<はHaskellにおけるモナディックな結合演算子の一つで、bind演算子とも呼ばれます。この演算子は、モナドの文脈で関数を適用するために使用されます。具体的には、モナドの値を取り、その値を引数として関数に渡し、結果として新しいモナドの値を返します。
-- Randomizable型クラスのインスタンス、ElmanSpecとElmanCell
-- sample :: ElmanSpec -> IO ElmanCell
-- randnIO' [in_features, hidden_features]は、指定された形状のランダムなテンソルを生成します。
-- makeIndependentは、そのテンソルを独立したテンソルに変換します。
-- 同様に、隠れ層から隠れ層への重みテンソルを生成し、w_hhに格納します。
-- バイアステンソルを生成し、bに格納します。
-- | このようにして、ElmanSpecからElmanCellをランダムに生成するための具体的な実装が提供されます。これは、ニューラルネットワークの初期化やランダムな重みの設定に非常に有用です。
instance Randomizable ElmanSpec ElmanCell where
  sample ElmanSpec {..} = do
    w_ih <- makeIndependent =<< randnIO' [in_features, hidden_features]
    w_hh <- makeIndependent =<< randnIO' [hidden_features, hidden_features]
    b <- makeIndependent =<< randnIO' [1, hidden_features]
    return $ ElmanCell w_ih w_hh b

instance Parameterized ElmanCell where
  flattenParameters ElmanCell {..} = [input_weight, hidden_weight, bias] -- ElmanCellのパラメータをリストに変換している
  _replaceParameters _ = do
    input_weight <- nextParameter  -- nextParameter関数を使用して新しいパラメータを生成する
    hidden_weight <- nextParameter
    bias <- nextParameter
    return $ ElmanCell {..}

instance Show ElmanCell where
  show ElmanCell {..} =
    (show input_weight) ++ "\n"
      ++ (show hidden_weight)
      ++ "\n"
      ++ (show bias)
      ++ "\n"