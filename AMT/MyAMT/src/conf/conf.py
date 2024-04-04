class Config:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 4
        self.learning_rate = 0.001

        self.sr = 44100  # Sample rate
        self.hop_length = int(self.sr * (1/64))
        self.n_fft = 2048  # FFT window size for Mel-spectrogram
        self.n_mels = 229  # Number of Mel bins
        self.target_duration = 300  # Target duration of audio clips in seconds

        self.input_size = self.n_mels  # Input feature dimension (Mel bins)
        self.hidden_size = 512  # LSTM hidden layer size
        self.num_layers = 3  # Number of LSTM layers
        self.bidirectional = True  # Whether to use a bidirectional LSTM
        self.output_size = 88  # Number of output nodes
        self.dropout = 0.3

        self.gamma = 3.0
        self.alpha = 0.70