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
-- import qualified Data.HashSet as HashSet
import qualified Data.Set as Set

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', Dim(..), mseLoss, softmax, stack, squeezeDim, relu, matmul, split, transpose)
import Torch.NN (Parameterized(..), Parameter, linear)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Torch.TensorFactories (eye', zeros', full)
import Torch.Optim        (foldLoop, GD(..), Loss, runStep, LearningRate)
import Torch.NN (Linear(..), sample, LinearSpec(..))
import Torch.Control      (mapAccumM)
import Torch.TensorOptions (defaultOpts)

import System.Random.Shuffle (shuffleM)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools

-- your text data (try small data first)
textFilePath = "app/word2vec/data/sample.txt"
-- textFilePath = "app/word2vec/data/sample_mini.txt"
modelPath =  "app/word2vec/data/sample_embedding.params"
wordLstPath = "app/word2vec/data/sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

data Model = Model {
    w_in :: Embedding,
    w_out :: Embedding
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
-- inputのTensor を 4*len(wordlst)にした
initDataSets :: [[B.ByteString]] -> [B.ByteString] -> IO [(Tensor, Tensor)]
initDataSets wordLines wordlst = do
  let dictLength = Prelude.length wordlst
      wordToIndex = wordToIndexFactory $ nub wordlst  -- indexを生成
      input = concatMap createInputPairs wordlst
      output = concatMap createOutputPairs wordlst
      pairs = zip input output
      createInputPairs word =
        let indices = [wordToIndex word - 1, wordToIndex word + 1, wordToIndex word - 2, wordToIndex word + 2]
            validIndices = filter (\i -> i >= 0 && i < Prelude.length wordlst) indices
            vectors = map (\i -> oneHotEncode (wordToIndex (wordlst !! i)) dictLength) validIndices
        in [stack (Dim 0) vectors]   -- []リストなのか危うい。リスト外したほうがいいかも
      createOutputPairs word = [oneHotEncode (wordToIndex word) dictLength]
  return pairs

-- フォワードパスの実装
-- inputのTensor を [4*len(wordlst), batchSize]にする
predict :: Model -> Tensor -> IO Tensor
predict model input = do
  let emb_in = wordEmbedding (w_in model)
  -- print (shape input)  -- [32,4,370]
  -- print (shape (toDependent emb_in))  -- [370,9]
  let embeddedInputs = split 1 (Dim 1) (matmul input (toDependent emb_in))
  let sumTensor = foldl1 vecBinaryAddition embeddedInputs -- ここの処理を変える
  -- print (shape sumTensor)  -- [32,1,9] TODO: ここの形を[32,1,9]に変えたい dim0からdim1に変えた
  let avgTensor = sumTensor / 4
  -- print (shape avgTensor)  -- [32,1,9]
  let nonlin = softmax (Dim 0)
  let emb_out = wordEmbedding (w_out model)
  -- print (shape avgTensor)  -- [32,1,9]
  -- print (shape (toDependent emb_out)) -- [370,9]
  let output = nonlin (matmul avgTensor (transpose (Dim 0) (Dim 1) (toDependent emb_out)))  -- (32x9 and 9×370)
  -- let output = foldl (\acc layer -> nonlin (linear layer acc)) embeddedInput mlpLayers  -- mlpLayers（リスト）の各要素のmlpレイヤーを適用し、非線形変換を行う。accの初期値はembeddedInput
  return output  -- おそらく[32,1,370]

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath  -- texts :: B.Internal.ByteString

  -- create word lst (unique)
  let wordLines' = preprocess texts -- wordLines :: [[B.ByteString]]
  let (wordLines, _) = splitAt (length wordLines' * 1 `div` 100) wordLines'
  let wordlst = Set.toList . Set.fromList . concat $ wordLines
  let wordToIndex = wordToIndexFactory wordlst  -- wordToIndex :: B.ByteString -> Int

  -- create embedding(wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 9} -- emsddingSpec :: EmbeddingSpec
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec -- wordEmb :: IndependentTensor
  let initW_in = Embedding { wordEmbedding = wordEmb } -- w_in :: Embedding
      initW_out = Embedding { wordEmbedding = wordEmb }
      initModel = Model { w_in = initW_in, w_out = initW_out }

  -- trainingData :: [(Tensor, Tensor)]
  trainingData' <- initDataSets wordLines wordlst
  let trainingData = drop 2 (take (length trainingData' - 2) trainingData')  -- 最初と最後だけ削除
  -- print $ trainingData !! 8 -- ここの出力がなかなか出てこない（データ10%だと大丈夫だった）

  let optimizer = GD
      numIters = 5
      learningRate = asTensor [0.1::Float]
      batchsize = 2048

  -- train 1個ずつ出力していく、確信を増やしていく。とりあえず直す。
  (trainedModel', _, losses') <- foldLoop (initModel, optimizer, []) numIters $ \(model', opt, lossesList) i -> do
    initRandamTrainData <- shuffleM trainingData
    ((trainedModel, _, _, _),losses) <- mapAccumM [1..((length trainingData) `div` batchsize)] (model', opt, initRandamTrainData, 0) $ \epoc (model, opt, randamTrainData, index) -> do
      let batchIndex = (index - 1) * batchsize
      let dataList = take batchsize $ drop batchIndex randamTrainData -- [(Tensor, Tensor)]
      let (input, target) = unzip dataList
      output <- predict model (stack (Dim 0) input)
      let loss = (mseLoss (stack (Dim 0) target) (squeezeDim 1 output)) / (asTensor (length dataList))  -- loss :: Tensor lossはバッチサイズで割るべきなのでは。→割った
      let newIndex = index + 1
      print index
      (newModel, _) <- runStep model optimizer loss learningRate  -- loss :: Torch.Optim.Loss（バッチごとに更新する）
      let lossValue = (asValue loss)::Float
      return ((newModel, opt, randamTrainData, newIndex), lossValue)  --更新されたモデルと損失値を返す, embも渡す？？？
    let avgLoss = sum losses / fromIntegral (length losses)
    -- print (w_in model)  -- ちゃんと更新はされている
    pure (trainedModel, opt,  lossesList ++ [avgLoss]) 

  drawLearningCurve "/home/acf16408ip/hasktorch-projects/app/word2vec/graph/learning_curve.png" "Learning Curve" [("",reverse losses')]

  -- w_inはembedding'関数に入れて、単語の分散表現を獲得したい

  -- save params
  saveParams (w_in trainedModel') modelPath
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