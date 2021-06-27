'''
Code adapted from https://github.com/ChunML/NLP/blob/32a52dc6a252175c60b44389a020fda17a6339b7/text_generation/train_pt.py#L24
Blog: https://trungtran.io/2019/02/08/text-generation-with-pytorch/
'''

import torch
import torch.nn as nn
import numpy as np
from collections import Counter

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size, lstm_num_layers=1, lstm_bidirectional=True, lstm_dropout=0, weights=None):
        super(RNNModule, self).__init__()

        # Hyperparameters
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.embedding_size = weights.shape[1] if weights is not None else embedding_size

        # Layers configuration
        ## Embedding layer
        self.embedding = nn.Embedding.from_pretrained(weights) if weights is not None else nn.Embedding(n_vocab, embedding_size)
        ## LSTM modules
        self.lstm = nn.LSTM(self.embedding_size,
                            lstm_size,
                            batch_first=True,
                            bidirectional=lstm_bidirectional,
                            dropout=lstm_dropout,
                            num_layers=self.lstm_num_layers)      
        ## Dense layer
        self.dense = nn.Linear(lstm_size*2 if self.lstm_bidirectional else lstm_size, n_vocab)

    def forward(self, x, prev_state):
        # print(prev_state[0].shape, prev_state[1].shape)
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        layers = self.lstm_num_layers*2 if self.lstm_bidirectional else self.lstm_num_layers

        return (torch.zeros(layers, batch_size, self.lstm_size),
                torch.zeros(layers, batch_size, self.lstm_size))

class RNNGenerator():

    def __init__(self, max_len=100, seq_size=32, batch_size=16, embedding_size=64, lstm_size=64, 
                 lstm_num_layers=1, lstm_bidirectional=True, lstm_dropout=0.5, gradients_norm=5, 
                 predict_top_k=5, training_epocs=200, lr=0.001, weights=None):
    
    # Hyperparameters
        self.max_len = max_len
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_dropout = lstm_dropout
        self.gradients_norm = gradients_norm
        self.predict_top_k = predict_top_k
        self.epochs = training_epocs
        self.lr = lr
        self.weights = weights
        self.vocab_to_int = None
        self.int_to_vocab = None

    def get_data_from_file(self, train_file, batch_size, seq_size):
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.split()

        word_counts = Counter(text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
        vocab_to_int = {w: k for k, w in int_to_vocab.items()}
        n_vocab = len(int_to_vocab)

        print('Vocabulary size', n_vocab)

        int_text = [vocab_to_int[w] for w in text]
        num_batches = int(len(int_text) / (seq_size * batch_size))
        in_text = int_text[:num_batches * batch_size * seq_size]
        out_text = np.zeros_like(in_text)
        out_text[:-1] = in_text[1:]
        out_text[-1] = in_text[0]
        in_text = np.reshape(in_text, (batch_size, -1))
        out_text = np.reshape(out_text, (batch_size, -1))

        self.int_to_vocab = int_to_vocab
        self.vocab_to_int = vocab_to_int

        return n_vocab, in_text, out_text


    def get_batches(self, in_text, out_text, batch_size, seq_size):
        num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
        for i in range(0, num_batches * seq_size, seq_size):
            yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


    def get_loss_and_train_op(self, net):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        return criterion, optimizer


    def predict(self, device, net, n_sentences=100):
        net.eval()
        sentences = []

        for _ in range(n_sentences):
            rand_seq = np.random.randint(1,self.max_len)
            words = np.random.choice(list(self.int_to_vocab.values()), np.random.randint(1, 3)).tolist()
            state_h, state_c = net.zero_state(1)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            for w in words:
                ix = torch.tensor([[self.vocab_to_int[w]]]).to(device)
                output, (state_h, state_c) = net(ix, (state_h, state_c))

            _, top_ix = torch.topk(output[0], k=self.predict_top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])

            words.append(self.int_to_vocab[choice])
            num_generated_words = rand_seq - len(words)
            print(num_generated_words)

            for _ in range(num_generated_words):
                ix = torch.tensor([[choice]]).to(device)
                output, (state_h, state_c) = net(ix, (state_h, state_c))

                _, top_ix = torch.topk(output[0], k=self.predict_top_k)
                choices = top_ix.tolist()
                choice = np.random.choice(choices[0])
                words.append(self.int_to_vocab[choice])

            generated_sentence = ' '.join(words)
            sentences.append(generated_sentence)

        return sentences


    def train(self, device, train_file):
        n_vocab, in_text, out_text = self.get_data_from_file(
            train_file, self.batch_size, self.seq_size)

        net = RNNModule(n_vocab=n_vocab, seq_size=self.seq_size, embedding_size=self.embedding_size, 
                        lstm_size=self.lstm_size, lstm_num_layers=self.lstm_num_layers, 
                        lstm_bidirectional=self.lstm_bidirectional, lstm_dropout=self.lstm_dropout, weights=self.weights
                        )
        net = net.to(device)

        criterion, optimizer = self.get_loss_and_train_op(net)

        if APEX_AVAILABLE:
            net, optimizer = amp.initialize(net, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
                                               loss_scale="dynamic")
        iteration = 0

        for e in range(self.epochs):
            batches = self.get_batches(in_text, out_text, self.batch_size, self.seq_size)
            state_h, state_c = net.zero_state(self.batch_size)
            state_h = state_h.to(device)
            state_c = state_c.to(device)

            for x, y in batches:
                iteration += 1
                net.train()

                optimizer.zero_grad()

                x = torch.tensor(x).to(device)
                y = torch.tensor(y).to(device)

                logits, (state_h, state_c) = net(x, (state_h, state_c))
                loss = criterion(logits.transpose(1, 2), y)

                loss_value = loss.item()

                if APEX_AVAILABLE:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                state_h = state_h.detach()
                state_c = state_c.detach()

                _ = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.gradients_norm)

                optimizer.step()

                if iteration % 100 == 0:
                    print('Epoch: {}/{}'.format(e, self.epochs),
                        'Iteration: {}'.format(iteration),
                        'Loss: {}'.format(loss_value))
        return net

                # if iteration % 1000 == 0:
                #     predict(device, net, flags.initial_words, n_vocab,
                #             vocab_to_int, int_to_vocab, top_k=5)
                #     # To save the model....
                #     # torch.save(net.state_dict(),
                #     #            'checkpoint_pt/model-{}.pth'.format(iteration))


####### Generation ##########

## Download pre-trained embeddings
# ! wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
# ! unzip '/content/wiki-news-300d-1M.vec.zip'
# import gensim.models.wrappers.fasttext
# model = gensim.models.KeyedVectors.load_word2vec_format('/content/wiki-news-300d-1M.vec')
# word_vectors = model.wv

# import torch
# import torch.nn as nn

# weights = torch.FloatTensor(word_vectors.vectors)
# embedding = nn.Embedding.from_pretrained(weights)


# file_path = '/content/neuraltextgen/data/wiki103.5k.txt'
# generator = RNNGenerator(training_epocs=1, lstm_num_layers=10, lr=0.1, lstm_bidirectional=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trained_net = generator.train(device, file_path)
# # list of sentences
# sentences = generator.predict(device, trained_net, n_sentences=100)