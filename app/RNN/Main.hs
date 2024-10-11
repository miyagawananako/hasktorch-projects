{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

{- Throuout the comments, a tensor of shape [a,b,..] is written as <a,b,...> -}

-- module Torch.Layer.RNN (
--   RnnHypParams(..)
--   , RnnParams(..)
--   , singleRnnLayer
--   , rnnLayers
--   , InitialStatesHypParams(..)
--   , InitialStatesParams(..)
--   ) where 
module Main where

import Prelude hiding   (tanh) 
import GHC.Generics              --base
import Data.Function    ((&))    --base
import Data.Maybe       (isJust) --base
import Data.List        (scanl',foldl',scanr) --base
import Control.Monad    (forM,unless)     --base
import System.IO.Unsafe (unsafePerformIO) --base
--hasktorch
import Torch.Tensor      (Tensor(..),shape,select,sliceDim,reshape)
import Torch.Functional  (Dim(..),add,sigmoid,cat,stack,dropout,transpose)
import Torch.Device      (Device(..))
import Torch.NN          (Parameterized(..),Randomizable(..),Parameter,sample)
import Torch.Autograd    (makeIndependent)
--hasktorch-tools
import Torch.Tensor.Util (unstack)
import Torch.Tensor.TensorFactories (randintIO')
import Torch.Layer.Linear (LinearHypParams(..),LinearParams(..),linearLayer)
import Torch.Layer.NonLinear (ActName(..),decodeAct)

(.->) :: a -> (a -> b) -> b
(.->) = (&)

data RnnHypParams = RnnHypParams {
  dev :: Device
  , bidirectional :: Bool -- ^ True if BiLSTM, False otherwise
  , inputSize :: Int  -- ^ The number of expected features in the input x（入力テンソルの特徴量の数）
  , hiddenSize :: Int -- ^ The number of features in the hidden state h
  , numLayers :: Int  -- ^ Number of recurrent layers（RNNの再帰層の数）
  , hasBias :: Bool   -- ^ If False, then the layer does not use bias weights b_ih and b_hh.
  -- , batch_first :: Bool -- ^ If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
  } deriving (Eq, Show)

newtype SingleRnnParams = SingleRnnParams {
    rnnGate :: LinearParams
    } deriving (Show, Generic)
instance Parameterized SingleRnnParams

-- cat関数を使用して、入力テンソルxtと隠れ状態テンソルhtを次元0（行方向）に沿って連結しています。これにより、2つのテンソルが1つのテンソルに結合されます。
-- 連結されたテンソルに対して、rnnGateを使用した線形変換を適用しています。linearLayer関数は、線形変換を実行する関数であり、rnnGateはそのパラメータを提供します。
-- このようにして、rnnCell関数は、現在の隠れ状態と入力を受け取り、それらを連結して線形変換を適用することで、次の隠れ状態を計算します。これは、RNNセルの基本的な動作を実装するための重要なステップです。
-- | 単一のRNNセルの計算を行う。現在の隠れ状態と入力を受け取り、次の隠れ状態を計算する。
rnnCell :: SingleRnnParams 
  -> Tensor -- ^ ht of shape <hDim>   （<hDim>は隠れ状態の次元数を示す）
  -> Tensor -- ^ xt of shape <iDim/oDim>
  -> Tensor -- ^ ht' of shape <hDim>
rnnCell SingleRnnParams{..} ht xt = linearLayer rnnGate $ cat (Dim 0) [xt,ht]

-- 一つのRNNレイヤーの計算を行う
-- 入力シーケンスに対しRNNセルを繰り返し適用し、全時間ステップの出力と最終隠れ状態を返す。
-- | inputのlistから、(cellState,hiddenState=output)のリストを返す
-- | （rnLayersのサブルーチン。外部から使う想定ではない）
-- | scanl'の型メモ :: ((h,c) -> input -> (h',c')) -> (h0,c0) -> [input] -> [(hi,ci)]
singleRnnLayer :: Bool -- ^ bidirectional (True if bidirectioal, False otherwise)
  -> Int               -- ^ stateDim (=hDim)
  -> ActName           -- ^ actname (the name of nonlinear function)
  -> SingleRnnParams   -- ^ singleRnnParams
  -> Tensor            -- ^ h0: <1,hDim> for one-directional and <2,hDim> for BiLSTM
  -> Tensor            -- ^ inputs: <seqLen,iDim/oDim> for the 1st-layer/the rest
  -> (Tensor,Tensor)   -- ^ an output pair (<seqLen,D*oDim>,<D*oDim>)
singleRnnLayer bidirectional stateDim actname singleRnnParams h0 inputs = unsafePerformIO $ do
  let h0shape = shape h0
      [seqLen,_] = shape inputs  -- inputsは<seqLen, iDim（入力の次元数）>
      d = if bidirectional then 2 else 1
      actf = decodeAct actname  -- 活性化関数を実際に取得
  unless (h0shape == [d,stateDim]) $ ioError $ userError $ "illegal shape of h0: " ++ (show h0shape) 
  if bidirectional -- check the well-formedness of the shapes of h0 and c0
    then do -- the case of BiRNN
      let h0f = select 0 0 h0 -- | pick the first h0 for the forward cells
          h0b = select 0 1 h0 -- | pick the second h0 for the backward cells
          hsForward = inputs  -- | <seqLen,iDim/oDim> 
            .-> unstack       -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (rnnCell singleRnnParams) h0f -- | [<hDim>] of length seqLen+1
            .-> tail          -- | [<hDim>] of length seqLen (by removing h0f)
            .-> stack (Dim 0) -- | <seqLen, hDim>
          hsBackward = inputs -- | <seqLen,iDim/oDim> 
            .-> unstack       -- | [<iDim/oDim>] of length seqLen
            .-> scanr (flip $ rnnCell singleRnnParams) h0b -- | [<hDim>] of length seqLen+1
            .-> init          -- | [<hDim>] of length seqLen (by removing h0b)
            .-> stack (Dim 0) -- | <seqLen, hDim>
          output = [hsForward, hsBackward]   -- | [<seqLen, hDim>] of length 2
            .-> stack (Dim 0)                -- | <2, seqLen, hDim>
            .-> actf                         -- | <2, seqLen, hDim> ??
            .-> transpose (Dim 0) (Dim 1)    -- | <seqLen, 2, hDim>
            .-> reshape [seqLen, 2*stateDim] -- | <seqLen, 2*hDim>
          hLast = output                           -- | <seqLen, 2*hDim>
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, 2*hDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <2*hDim> 
      return (output, hLast)
    else do -- the case of RNN
      let h0f = select 0 0 h0
          output = inputs
            .-> unstack          -- | [<iDim/oDim>] of length seqLen
            .-> scanl' (rnnCell singleRnnParams) h0f -- | [<hDim>] of length seqLen+1  RNNセルを各時間ごとに適用
            .-> tail             -- | [<hDim>] of length seqLen (by removing h0)  先頭を削除
            .-> stack (Dim 0)    -- | <seqLen, hDim>  
            .-> actf             -- | <seqLen, hDim> ??
          hLast = output                           -- | <seqLen, hDim>  最終的な隠れ状態
            .-> sliceDim 0 (seqLen-1) seqLen 1     -- | <1, hDim>
            .-> (\o -> reshape (tail $ shape o) o) -- | <hDim>
      return (output, hLast)

data RnnParams = RnnParams {
  firstRnnParams :: SingleRnnParams    -- ^ a model for the first RNN layer
  , restRnnParams :: [SingleRnnParams] -- ^ models for the rest of RNN layers
  } deriving (Show, Generic)
instance Parameterized RnnParams

instance Randomizable RnnHypParams RnnParams where
  sample RnnHypParams{..} = do
    let xDim = inputSize  -- 入力の次元数
        hDim = hiddenSize  -- 隠れ層の次元数
        xh1Dim = xDim + hDim  -- RNNの最初のレイヤーに渡される次元（入力と前の隠れ状態の結合）
        d = if bidirectional then 2 else 1
        xh2Dim = (d * hDim) + hDim  -- 最初のRNN層の次の層のための次元。双方向性RNNの場合、出力が2倍になるためdをかけている
    RnnParams
      <$> (SingleRnnParams <$> sample (LinearHypParams dev hasBias xh1Dim hDim)) -- gate  1つのRNNレイヤーのパラメータを表している
      <*> forM [2..numLayers] (\_ ->  -- 2番目以降の層をここで形成
        SingleRnnParams <$> sample (LinearHypParams dev hasBias xh2Dim hDim)
        )

-- | The main function for RNN layers
rnnLayers :: RnnParams -- ^ parameters (=model)
  -> ActName         -- ^ name of nonlinear function(非線形関数の名前を指定します。例えば、ReLUやTanhなどの活性化関数が考えられます。)
  -> Maybe Double    -- ^ introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.（ドロップアウトの確率）
  -> Tensor          -- ^ an initial tensor: <D*numLayers,hDim>（RNNの初期状態、Dは双方向性RNNの場合2で一方向RNNの場合1）
  -> Tensor          -- ^ an input tensor <seqLen,iDim>（シーケンスの長さと入力次元）
  -> (Tensor,Tensor) -- ^ an output of (<seqLen,D*oDim>,(<D*numLayers,oDim>,<D*numLayers,cDim>))
rnnLayers RnnParams{..} actname dropoutProb h0 inputs = unsafePerformIO $ do
  let numLayers = length restRnnParams + 1  -- RNNレイヤーの数
      (dnumLayers:(hiddenSize:_)) = shape h0 -- 初期RNNのshapeをとって[Int]を割り当ててる
  unless (dnumLayers == numLayers * 2 || dnumLayers == numLayers) $ 
    ioError $ userError $ "illegal shape of h0: dnumLayers = " ++ (show dnumLayers) ++ "\nnumLayers = " ++ (show numLayers) 
  let bidirectional | dnumLayers == numLayers * 2 = True  -- 双方向RNN（順伝播と逆伝播両方）かどうかを確認している（なぜこれで確認できるの？）
                    | dnumLayers == numLayers = False
                    | otherwise = False -- Unexpected
      d = if bidirectional then 2 else 1
      (h0h:h0t) = [sliceDim 0 (d*i) (d*(i+1)) 1 h0 | i <- [0..numLayers]] -- h0テンソルをnumLayers+1個の部分に分割
      firstLayer = singleRnnLayer bidirectional hiddenSize actname firstRnnParams h0h -- cellstate, hiddenStateのペア（Tensor, Tensor)
      restOfLayers = map (uncurry $ singleRnnLayer bidirectional hiddenSize actname) $ zip restRnnParams h0t -- uncurryは2つの引数を取る関数（今回はsingleRnnLayer）にタプルを渡せるようにする関数
      dropoutLayer = case dropoutProb of
                       Just prob -> unsafePerformIO . (dropout prob True)  -- ドロップアウト確率が指定されている場合、ドロップアウトを行う
                       Nothing -> id
      stackedLayers = \inputTensor -> -- 出力は各レイヤーの出力のリスト
                        scanr  -- リストに対して右からの畳み込み
                          (\nextLayer ohc -> nextLayer $ dropoutLayer $ fst ohc)  -- 各レイヤーの処理を表す関数
                          (firstLayer inputTensor) -- (<seqLen,D*oDim>,(<D*oDim>,<D*cDim>)) 最初のレイヤーの処理結果
                          restOfLayers -- 残りのレイヤーのリスト
      (outputList, hn) = inputs -- | <seqLen,iDim>  outputListは各レイヤーの全時間ステップの出力のリスト、hnは各レイヤーの最終時間ステップの隠れ状態のリスト
        .-> stackedLayers       -- | [(<seqLen, D*oDim>,<D*oDim>)] of length numLayers
        .-> unzip               -- | ([<seqLen, D*oDim>] of length numLayers, [(<D*oDim>,<D*cDim>)] of length numLayers)
      output = head outputList  -- | [<seqLen, D*oDim>] of length numLayers
      rh = hn                                      -- | 
           .-> stack (Dim 0)                       -- | <numLayers, D*oDim/cDim>  テンソルのリストを1つのテンソルに積み重ねる
           .-> reshape [d * numLayers, hiddenSize] -- | <D*numLayers, oDim/cDim>  [d * numLayers, hiddenSize] の形に形状を変更
  return (output, rh)

data InitialStatesHypParams = InitialStatesHypParams {
  dev :: Device
  , bidirectional :: Bool
  , hiddenSize :: Int
  , numLayers :: Int
  } deriving (Eq, Show)

newtype InitialStatesParams = InitialStatesParams {
  h0s :: Parameter 
  } deriving (Show, Generic)
instance Parameterized InitialStatesParams

-- 初期状態のハイパーパラメータから、実際の初期隠れ状態をランダムに生成するためのインスタンス
instance Randomizable InitialStatesHypParams InitialStatesParams where
  sample InitialStatesHypParams{..} = 
    InitialStatesParams
      <$> (makeIndependent =<< randintIO' dev (-1) 1 [(if bidirectional then 2 else 1) * numLayers, hiddenSize])  -- -1から1の乱数を生成(randintIO'~1)、[]は生成される乱数のテンソルの形状の指定

main :: IO ()
main = do
  putStrLn "Hello, this is the RNN module!"