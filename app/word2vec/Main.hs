{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import GHC.Generics
import qualified Data.ByteString.Lazy as B -- add bytestring to dependencies in package.yaml
import Data.Word (Word8)
import qualified Data.Map.Strict as M -- add containers to dependencies in package.yaml
import Data.List (nub)

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding')
import Torch.NN (Parameterized(..), Parameter, linear)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Torch.TensorFactories (eye', zeros', full)
import Torch.Functional (Dim(..), mseLoss, softmax, stack)
import Torch.Optim        (foldLoop, GD(..), Loss, runStep, LearningRate)
import Torch.NN (Linear(..), sample, LinearSpec(..))
import Torch.Functional (relu, matmul)
import Torch.Control      (mapAccumM)
import Torch.TensorOptions (defaultOpts)

import System.Random.Shuffle (shuffleM)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools

-- your text data (try small data first)
-- textFilePath = "app/word2vec/data/sample.txt"
textFilePath = "app/word2vec/data/sample_mini.txt"
modelPath =  "app/word2vec/data/sample_embedding.params"
wordLstPath = "app/word2vec/data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

-- 使わない
data MLP = MLP
  { layers :: [Linear],  -- 入力テンソルに対して線形変換を適用する役割を持つ
    nonlinearity :: Tensor -> Tensor  -- 非線形活性化関数
  } deriving (Generic, Parameterized)

-- 使わない
-- Probably you should include model and Embedding in the same data class.
data Model = Model {
    mlp :: MLP
    -- w_in :: Embedding,
    -- w_out :: Embedding,
  } deriving (Generic, Parameterized)

isUnnecessaryChar :: 
  Word8 ->
  Bool
isUnnecessaryChar str = 
  (str /= 39 && str /= 45) &&  -- '-以外
  ((str >= 33 && str <= 47) ||  -- !"#$%&'()*+,-./
  (str >= 58 && str <= 64) ||  -- :;<=>?@
  (str >= 91 && str <= 96) ||  -- [\]^_`
  (str >= 123 && str <= 126))   -- {|}~

toLowerWord8 :: Word8 -> Word8
toLowerWord8 w
  | w >= 65 && w <= 90 = w + 32  -- ASCII 'A'-'Z' to 'a'-'z'
  | otherwise = w

preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    lowercaseTexts = B.map toLowerWord8 texts
    filteredtexts = B.pack $ filter (not . isUnnecessaryChar) (B.unpack lowercaseTexts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))

toyEmbedding ::
  EmbeddingSpec ->
  Tensor           -- embedding
toyEmbedding EmbeddingSpec{..} = 
  eye' wordNum wordDim

setAt :: Int -> a -> [a] -> [a]
setAt idx val lst = take idx lst ++ [val] ++ drop (idx + 1) lst

oneHotEncode :: Int -> Int -> Tensor
oneHotEncode index size = asTensor $ setAt index 1 (zeros :: [Float])
  where
    zeros = replicate size 0

vecBinaryAddition :: Tensor -> Tensor -> Tensor
vecBinaryAddition vec1 vec2 = vec1 + vec2

-- CBOW（input: 周辺4単語, output: 中心単語）
initDataSets :: [[B.ByteString]] -> [B.ByteString] -> [(Tensor, Tensor)]
initDataSets wordLines wordlst = pairs
  where
      dictLength = Prelude.length wordlst
      wordToIndex = wordToIndexFactory $ nub wordlst  -- indexを生成
      input = concatMap createInputPairs wordlst
      output = concatMap createOutputPairs wordlst
      pairs = zip input output
      createInputPairs word =
        let indices = [wordToIndex word - 1, wordToIndex word + 1, wordToIndex word - 2, wordToIndex word + 2]
            validIndices = filter (\i -> i >= 0 && i < Prelude.length wordlst) indices
            vectors = map (\i -> oneHotEncode (wordToIndex (wordlst !! i)) dictLength) validIndices
        in [foldl1 vecBinaryAddition vectors]
      createOutputPairs word = [oneHotEncode (wordToIndex word) dictLength]


-- MLPの初期化（これが違う）
initMLP :: IO MLP
initMLP = do
  layer1 <- sample $ LinearSpec 9 64
  layer2 <- sample $ LinearSpec 64 370  -- 370は単語数
  let layers = [layer1, layer2]
  let nonlinearity = relu  -- 入力値が0以下の場合は0より上の場合には出力値が入力値と同じ値となる関数
  return $ MLP layers nonlinearity

-- フォワードパスの実装
predict :: Model -> Tensor -> Embedding -> IO Tensor
predict model input embedding = do
  let emb = wordEmbedding embedding  -- 埋め込み行列を取得
  -- print emb 出力 IndependentTensor {toDependent = Tensor Float [370,9] [[ 1.0000   ,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
  -- print input  -- 出力できない  Not implemented for Tensor-typeの部分だった
  let embeddedInput = matmul input (toDependent emb)  -- 入力テンソルと埋め込み行列の行列乗算を行う（toDependentは通常テンソルに戻す）
  -- print embeddedInput  -- 出力されず　→できた
  let mlpLayers = layers (mlp model)  -- モデルのMLPレイヤーを取得
  let nonlin = nonlinearity (mlp model)  -- 非線形活性化関数を取得
  let output = foldl (\acc layer -> nonlin (linear layer acc)) embeddedInput mlpLayers  -- mlpLayers（リスト）の各要素のmlpレイヤーを適用し、非線形変換を行う。accの初期値はembeddedInput
  return output

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath  -- texts :: B.Internal.ByteString

  -- create word lst (unique)
  let wordLines = preprocess texts -- wordLines :: [[B.ByteString]]
      wordlst = nub $ concat wordLines  -- wordlst :: [B.ByteString]
      wordToIndex = wordToIndexFactory wordlst  -- wordToIndex :: B.ByteString -> Int
  -- print wordlst

  -- create embedding(wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 9} -- emsddingSpec :: EmbeddingSpec
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec -- wordEmb :: IndependentTensor
  let w_in = Embedding { wordEmbedding = wordEmb } -- w_in :: Embedding

  mlp <- initMLP
  let initModel = Model mlp

  -- trainingData :: [(Tensor, Tensor)]
  let trainingData = initDataSets wordLines wordlst
  -- print $ trainingData !! 8

  let optimizer = GD
      numIters = ((length trainingData) `div` batchsize)
      learningRate = asTensor [0.01::Float]
      batchsize = 32

  initRandamTrainData <- shuffleM trainingData

  print "before training"

  -- train（エラーを吐く部分）  1個ずつ出力していく、確信を増やしていく。とりあえず直す。
  ((trainedModel, _, _, _),losses) <- mapAccumM [1..numIters] (initModel, optimizer, initRandamTrainData, 0) $ \epoc (model, opt, randamTrainData, index) -> do  -- 各エポックでモデルを更新し、損失を蓄積。
    let batchIndex = (index - 1) * batchsize
    let dataList = take batchsize $ drop batchIndex randamTrainData -- [(Tensor, Tensor)]
    let (input, target) = unzip dataList
    -- print $ shape (head input)  -- input::[Tensor]  [370]と出力された
    -- print $ length input  -- 32
    -- print $ shape target[0]  -- [Tensor]
    output <- predict model (stack (Dim 0) input) w_in  -- このw_inが更新されるべきじゃないか
    let loss = mseLoss (stack (Dim 0) target) output  -- loss :: Tensor
    --  print lossの結果、　The size of tensor a (32) must match the size of tensor b (370) at non-singleton dimension 1
    let newIndex = index + 1
    -- let lossTensor = asTensor loss
    -- let learningRateTensor = asTensor learningRate
    (newModel, _) <- runStep model optimizer (loss :: Torch.Optim.Loss) learningRate  -- loss :: Torch.Optim.Loss
    let lossValue = (asValue loss)::Float
    return ((newModel, opt, randamTrainData, newIndex), lossValue)  --更新されたモデルと損失値を返す, embも渡す？？？

  drawLearningCurve "/home/acf16408ip/hasktorch-projects/app/word2vec/graph/learning_curve.png" "Learning Curve" [("",reverse losses)]

  print "after training"

  -- trainedModelだけ保存しているのがおかしい。
  -- w_inを取得したい。w_inはembedding'関数に入れて、単語の分散表現を獲得したい

  -- save params
  saveParams w_in modelPath
  -- save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- load params（さっきのモデルをloadする）
  initWordEmb <- makeIndependent $ zeros' [1]  -- initWordEmb :: IndependentTensor
  let initEmb = Embedding {wordEmbedding = initWordEmb}  -- initEmb :: Embedding
  loadedEmb <- loadParams initEmb modelPath  -- loadedEmb :: Embedding
  -- print loadedEmb

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing" -- sampleTxt :: B.ByteString
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)  -- idxes :: [[Int]]
  -- convert to embedding
      embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)  -- embTxt :: Tensor？
      -- embedding' :: Tensor -> Tensor -> Tensor
      -- toDependent :: IndependentTensor -> Tensor
  print sampleTxt
  print idxes  -- [[27,1,369],[369,1,369]]。Thisが27, isが1, awesomeが369。

  -- TODO: train models with initialized embeddings
  
  return ()