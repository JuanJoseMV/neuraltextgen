import torch
import torch.nn as nn
import numpy as np
from collections import Counter

class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)                      
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

class RNNGenerator():

    def __init__(self, train_file, seq_size=32, batch_size=16, embedding_size=64, lstm_size=64, gradients_norm=5, predict_top_k=5, training_epocs=200, lr=0.001):
    
    # Hyperparameters
        self.train_file = train_file,
        self.seq_size = seq_size,
        self.batch_size = batch_size,
        self.embedding_size = embedding_size,
        self.lstm_size = lstm_size,
        self.gradients_norm = gradients_norm,
        self.predict_top_k = predict_top_k
        self.epochs = training_epocs
        self.lr = lr
        self.vocab_to_int = None
        self.int_to_vocab = None

    def get_data_from_file(self, train_file, batch_size, seq_size):
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.split()
        text = text[:int(len(text) * 0.1)]

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


    def predict(self, device, net, n_sentences=100, top_k=5):
        net.eval()
        sentences = []

        for _ in range(n_sentences):
            words = np.random.choice(list(self.int_to_vocab.values()), np.random.randint(1, 3)).tolist()
            state_h, state_c = net.zero_state(1)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            for w in words:
                ix = torch.tensor([[self.vocab_to_int[w]]]).to(device)
                output, (state_h, state_c) = net(ix, (state_h, state_c))

            _, top_ix = torch.topk(output[0], k=top_k)
            choices = top_ix.tolist()
            choice = np.random.choice(choices[0])

            words.append(self.int_to_vocab[choice])

            for _ in range(100):
                ix = torch.tensor([[choice]]).to(device)
                output, (state_h, state_c) = net(ix, (state_h, state_c))

                _, top_ix = torch.topk(output[0], k=top_k)
                choices = top_ix.tolist()
                choice = np.random.choice(choices[0])
                words.append(self.int_to_vocab[choice])

        generated_sentence = ' '.join(words).encode('utf-8')
        sentences.append(generated_sentence)
        
        return sentences


    def train(self):
        epochs = 200
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_vocab, in_text, out_text = self.get_data_from_file(
            self.train_file, self.batch_size, self.seq_size)

        net = RNNModule(n_vocab, self.seq_size,
                        self.embedding_size, self.lstm_size)
        net = net.to(device)

        criterion, optimizer = self.get_loss_and_train_op(net, 0.01)

        iteration = 0

        for e in range(epochs):
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

                loss.backward()

                state_h = state_h.detach()
                state_c = state_c.detach()

                _ = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), self.gradients_norm)

                optimizer.step()

                if iteration % 100 == 0:
                    print('Epoch: {}/{}'.format(e, 200),
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

# file_path = '/content/neuraltextgen/data/wiki103.5k.txt'
# generator = RNNGenerator(train_file=file_path)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# trained_net = generator.train(device)
# list of sentences
# sentences = generator.predict(device, trained_net, ["One"], n_sentences=100, top_k=5)