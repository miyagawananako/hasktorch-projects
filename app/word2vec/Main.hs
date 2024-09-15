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
import Torch.Tensor (Tensor, asTensor)
import Torch.TensorFactories (eye', zeros')
import qualified Torch.Layer.MLP as MLP
import Torch.Functional (Dim(..), mseLoss, softmax)
import Torch.Optim        (foldLoop, GD(..))
import Torch.NN (Linear(..), sample, LinearSpec(..))
-- import Torch.Train        (update)
import Torch.Tensor (Tensor)
import Torch.Functional (relu)

import Torch.Functional (matmul)

import Torch.Optim (GD(..), runStep)
import Torch.Autograd (grad)
import Torch.Functional (mseLoss)

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

data MLP = MLP
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  } deriving (Generic, Parameterized)

-- Probably you should include model and Embedding in the same data class.
data Model = Model {
    mlp :: MLP
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

-- input: 周辺4単語, output: 中心単語
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


-- MLPの初期化
initMLP :: IO MLP
initMLP = do
  layer1 <- sample $ LinearSpec 128 64
  layer2 <- sample $ LinearSpec 64 32
  let layers = [layer1, layer2]
  let nonlinearity = relu
  return $ MLP layers nonlinearity

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath  -- texts :: B.Internal.ByteString

  -- create word lst (unique)
  let wordLines = preprocess texts -- wordLines :: [[B.ByteString]]
      wordlst = nub $ concat wordLines  -- wordlst :: [B.ByteString]
      wordToIndex = wordToIndexFactory wordlst  -- wordToIndex :: B.ByteString -> Int
  print wordlst

  -- create embedding(wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 9} -- emsddingSpec :: EmbeddingSpec
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec -- wordEmb :: IndependentTensor
  let emb = Embedding { wordEmbedding = wordEmb } -- emb :: Embedding

  mlp <- initMLP
  let initModel = Model mlp

  let trainingData = initDataSets wordLines wordlst
  print $ trainingData !! 8

  -- save params
  saveParams emb modelPath
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