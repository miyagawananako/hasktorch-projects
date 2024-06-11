{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Prelude hiding (tanh) 
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
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

-- この行は、訓練データを定義しています。訓練データは、入力と出力のペアのリストです。
-- cycle関数は、与えられたリストを無限に繰り返すリストを生成します。take 10は、その無限リストの最初の10要素を取得します。
trainingData :: [([Float],Float)]
trainingData = take 10 $ cycle [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

main :: IO()
main = do
  let iter = 1500::Int -- 訓練のイテレーター数
      device = Device CPU 0  -- 使用するデバイス
      hypParams = MLPHypParams device 2 [(3,Sigmoid),(1,Sigmoid)]  -- ニューラルネットワークのハイパーパラメータ。入力層のノード数は2。隠れ層の3ノードのSigmoid活性化関数。出力層は1ノードのSigmoid活性化関数を持つMLPを定義している。
  initModel <- sample hypParams  -- hyperParamsに従って、初期モデルをサンプリングする。
  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do  -- 各エポックでモデルを更新し、損失を蓄積。
    let loss = sumTensors $ for trainingData $ \(input,output) ->
                  let y = asTensor'' device output
                      y' = mlpLayer model $ asTensor'' device input
                  in mseLoss y y'  -- 平均二乗誤差を計算
        lossValue = (asValue loss)::Float  -- 消失テンソルをFloat値に変換
    showLoss 10 epoc lossValue  -- エポック数と損失数を表示。10は表示の間隔。
    u <- update model opt loss 1e-1  -- モデルを更新する
    return (u, lossValue)  --更新されたモデルと損失値を返す
  drawLearningCurve "graph-xor.png" "Learning Curve" [("",reverse losses)]
  forM_ ([[1,1],[1,0],[0,1],[0,0]::[Float]]) $ \input -> do  -- リスト[[1,1],[1,0],[0,1],[0,0]]の各要素（入力）に対して以下の操作を行います。
    putStr $ show $ input
    putStr ": "
    putStrLn $ show ((mlpLayer trainedModel $ asTensor'' device input))  -- 入力値をテンソルに変換し、訓練済みモデルを使用して出力を生成します。その出力を文字列に変換して表示します。
  -- print trainedModel
  where for = flip map  -- map関数の引数の順序を反転したもの

