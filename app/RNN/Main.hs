import Torch.Layer.RNN (RnnHypParams)

hyperParams :: RnnHypParams
hyperParams = RnnHypParams {
    dev = Device CPU 0,
    bidirectional = False, -- ^ True if BiLSTM, False otherwise
    inputSize = iDim, -- ^ The number of expected features in the input x
    hiddenSize = hDim, -- ^ The number of features in the hidden state h
    numLayers = numLayers, -- ^ Number of recurrent layers
    hasBias = True -- ^ If False, then the layer does not use bias weights b_ih and b_hh. Default: True
}

main :: IO ()
main = do
    print [1, 2, 3]