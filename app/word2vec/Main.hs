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
import Data.List (nub, sortOn)
import qualified Data.Set as Set
import Data.Maybe (fromMaybe)
import Control.Monad (forM_)  -- For forM_ function

import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', Dim(..), softmax, stack, squeezeDim, matmul, split, transpose, binaryCrossEntropyLoss')
import Torch.NN (Parameterized(..), Parameter)
import Torch.Serialize (saveParams, loadParams)
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Torch.TensorFactories (zeros', randnIO')
import Torch.Optim        (foldLoop, GD(..), runStep)
import Torch.Control      (mapAccumM)

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

-- -- 絵文字の範囲を判定する関数
-- isEmoji :: Word8 -> Bool
-- isEmoji w =
--   (w >= 0xF0 && w <= 0xF7) ||  -- 基本的な絵文字をカバー
--   (w >= 0xE2 && w <= 0xE3)     -- 拡張絵文字の範囲

-- -- 絵文字を除去するフィルタリング関数
-- filterEmojis :: B.ByteString -> B.ByteString
-- filterEmojis = B.filter (not . isEmoji)


preprocess ::
  B.ByteString -> -- input
  [[B.ByteString]]  -- wordlist per line
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    lowercaseTexts = B.map toLowerWord8 texts
    -- filteredEmojis = filterEmojis lowercaseTexts
    filteredtexts = B.pack $ filter (not . isUnnecessaryChar) (B.unpack lowercaseTexts)
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))

-- 単語→インデックス変換
createWordToIndexMap :: [B.ByteString] -> M.Map B.ByteString Int
createWordToIndexMap wordlst = M.fromList $ zip wordlst [0..]

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
initDataSets :: [B.ByteString] -> IO [(Tensor, Tensor)]
initDataSets wordlst = do
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
-- inputのTensor  [4*len(wordlst), batchSize]
predict :: Model -> Tensor -> IO Tensor
predict model input = do
  let emb_in = wordEmbedding (w_in model)
  -- print (shape input)  -- [32,4,370]
  -- print (shape (toDependent emb_in))  -- [370,9]
  let embeddedInputs = split 1 (Dim 1) (matmul input (toDependent emb_in))
  let sumTensor = foldl1 vecBinaryAddition embeddedInputs
  -- print (shape sumTensor)  -- [32,1,9] TODO: ここの形を[32,1,9]に変えたい dim0からdim1に変えた
  let avgTensor = sumTensor / 4
  -- print (shape avgTensor)  -- [32,1,9]
  let nonlin = softmax (Dim 0)
  let emb_out = wordEmbedding (w_out model)
  -- print (shape avgTensor)  -- [32,1,9]
  -- print (shape (toDependent emb_out)) -- [370,9]
  let output = nonlin (matmul avgTensor (transpose (Dim 0) (Dim 1) (toDependent emb_out)))  -- (32x9 and 9×370)
  -- print output
  return output  -- おそらく[32,1,370]

  -- 単語のベクトル表現を取得する関数
getWordVector :: Embedding -> M.Map B.ByteString Int -> B.ByteString -> Maybe Tensor
getWordVector emb wordToIndexMap word = do
    wordIdx <- M.lookup word wordToIndexMap
    let wordTensor = oneHotEncode wordIdx (M.size wordToIndexMap)
    return $ matmul wordTensor (toDependent $ wordEmbedding emb)

-- コサイン類似度を計算する関数
cosineSimilarity :: Tensor -> Tensor -> Float
cosineSimilarity v1 v2 = 
    let dot = asValue $ (v1 * v2)
        norm1 = sqrt $ asValue $ (v1 * v2)
        norm2 = sqrt $ asValue $  (v1 * v2)
    in if norm1 == 0 || norm2 == 0 then 0
       else dot / (norm1 * norm2)

-- 最も類似度の高いN個の単語を見つける関数
findMostSimilarWords :: Int -> Embedding -> M.Map B.ByteString Int -> B.ByteString -> IO [(B.ByteString, Float)]
findMostSimilarWords n emb wordToIndexMap targetWord = do
    case getWordVector emb wordToIndexMap targetWord of
        Nothing -> return []
        Just targetVec -> do
            -- すべての単語との類似度を計算
            similarities <- sequence 
                [ do
                    case getWordVector emb wordToIndexMap word of
                        Nothing -> return (word, -1.0)
                        Just wordVec -> return (word, cosineSimilarity targetVec wordVec)
                | word <- M.keys wordToIndexMap
                ]
            -- 類似度でソートして上位N個を返す
            return $ take n $ reverse $ sortOn snd $ filter (\(w, s) -> w /= targetWord) similarities

-- メイン関数に追加するテスト用コード
testSimilarity :: Embedding -> M.Map B.ByteString Int -> IO ()
testSimilarity emb wordToIndexMap = do
    let testWords = ["computer", "data", "program", "system", "network", "king", "queen", "drink"]
    forM_ testWords $ \word -> do
        putStrLn $ "\nFinding similar words for: " ++ show word
        similar <- findMostSimilarWords 10 emb wordToIndexMap (B.pack $ encode word)
        forM_ similar $ \(w, score) -> 
            putStrLn $ "  " ++ show w ++ ": " ++ show (score * 100) ++ "%"

main :: IO ()
main = do
  -- load text file
  texts <- B.readFile textFilePath  -- texts :: B.Internal.ByteString

  -- create word lst (unique)
  let wordLines' = preprocess texts -- wordLines :: [[B.ByteString]]
  let (wordLines, _) = splitAt (length wordLines' * 1 `div` 10) wordLines'
  let wordlst = Set.toList . Set.fromList . concat $ wordLines
  let wordToIndex = wordToIndexFactory wordlst  -- wordToIndex :: B.ByteString -> Int
      wordToIndexMap = createWordToIndexMap wordlst
  print wordToIndexMap -- [(word, index)]

  -- create embedding(wordDim × wordNum)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 9} -- emsddingSpec :: EmbeddingSpec
  initRandomTensor <- randnIO' [wordNum embsddingSpec, wordDim embsddingSpec]
  wordEmb <- makeIndependent initRandomTensor
  let initW_in = Embedding { wordEmbedding = wordEmb } -- w_in :: Embedding
      initW_out = Embedding { wordEmbedding = wordEmb }
      initModel = Model { w_in = initW_in, w_out = initW_out }

  -- trainingData :: [(Tensor, Tensor)]
  trainingData' <- initDataSets wordlst
  let trainingData = drop 2 (take (length trainingData' - 2) trainingData')  -- 最初と最後だけ削除

  let optimizer = GD
      numIters = 10
      learningRate = asTensor (0.1::Float)
      batchsize = 2048

  -- -- train 1個ずつ出力していく、確信を増やしていく。とりあえず直す。
  (trainedModel', losses') <- foldLoop (initModel, []) numIters $ \(model', lossesList) i -> do
    initRandamTrainData <- shuffleM trainingData
    ((trainedModel, _, _),losses) <- mapAccumM [1..((length trainingData) `div` batchsize)] (model', initRandamTrainData, 0) $ \epoc (model, randamTrainData, index) -> do
      let batchIndex = (index - 1) * batchsize
      let dataList = take batchsize $ drop batchIndex randamTrainData
      let (input, target) = unzip dataList
      output <- predict model (stack (Dim 0) input)
      let loss = binaryCrossEntropyLoss' (stack (Dim 0) target) (squeezeDim 1 output)
      let newIndex = index + 1
      (newModel, _) <- runStep model optimizer loss learningRate
      let lossValue = (asValue loss)::Float
      print newIndex
      return ((newModel, randamTrainData, newIndex), lossValue)
    let avgLoss = sum losses / fromIntegral (length losses)
    pure (trainedModel, lossesList ++ [avgLoss]) 

  drawLearningCurve "/home/acf16408ip/hasktorch-projects/app/word2vec/graph/learning_curve.png" "Learning Curve" [("",reverse losses')]

  -- save params
  saveParams (w_in trainedModel') modelPath
  -- save word list
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  
  -- load params（さっきのモデルをloadする）
  -- initWordEmb <- makeIndependent $ zeros' [1]  -- initWordEmb :: IndependentTensor
  -- let initEmb = Embedding {wordEmbedding = initWordEmb}  -- initEmb :: Embedding
  -- loadedEmb <- loadParams initEmb modelPath  -- loadedEmb :: Embedding
  -- print loadedEmb
  let loadedEmb = w_in trainedModel'

  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing" -- sampleTxt :: B.ByteString
  -- convert word to index
      idxes = map (map wordToIndex) (preprocess sampleTxt)  -- idxes :: [[Int]]
  -- convert to embedding
      embTxt = embedding' (toDependent $ wordEmbedding loadedEmb) (asTensor idxes)  -- embTxt :: Tensor？
      -- embedding' :: Tensor -> Tensor -> Tensor
      -- toDependent :: IndependentTensor -> Tensor
  print sampleTxt
  print idxes  -- [[27,1,369],[369,1,369]]。Thisが27, isが1, awesomeが369。
  print embTxt

  testSimilarity loadedEmb wordToIndexMap
  
  return ()