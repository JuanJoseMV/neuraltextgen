import math
import random
import time
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, BertConfig, AutoConfig

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLS = '[CLS]'
SEP = '[SEP]'
MASK = '[MASK]'


def tokenize_batch(batch, tokenizer):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch, tokenizer):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def detokenize(sent, tokenizer):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    s = tokenizer.convert_tokens_to_string(sent)

    for tok in [tokenizer.mask_token, tokenizer.cls_token, tokenizer.sep_token]:
        s = s.replace(tok, '')
    return s


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def read_sents(in_file, should_detokenize=False):
    sents = [sent.strip().split() for sent in open(in_file).readlines()]
    if should_detokenize:
        sents = [detokenize(sent) for sent in sents]
    return sents


def write_sents(out_file, sents, should_detokenize=False):
    with open(out_file, "w") as out_fh:
        for sent in sents:
            sent = detokenize(sent[1:-1]) if should_detokenize else sent
            ###print("%s\n" % " ".join(sent))
            out_fh.write("%s\n" % " ".join(sent))


class BertTextGenerator:
    def __init__(self, model_version, device=DEFAULT_DEVICE, use_apex=False):
        self.device = device
        self.model_version = model_version
        self.model = AutoModelForMaskedLM.from_pretrained(model_version)
        self.model.to(self.device)

        if use_apex and APEX_AVAILABLE:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
                                                   loss_scale="dynamic")

        self.tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case="uncased" in model_version)

    def generate(self, save_to_path=None, n_samples=100, seed_text="", batch_size=10, max_len=25, sample=True,
                 top_k=100, temperature=1.0, burnin=200, max_iter=500, print_every=1, init_method='masked',masked_portion=0.15,
                 generation_method="parallel sequential", verbose=True):
        '''

        :param save_to_path:
        :param n_samples:
        :param seed_text:
        :param batch_size:
        :param max_len:
        :param sample:
        :param top_k:
        :param temperature:
        :param burnin:
        :param max_iter:
        :param print_every:
        :param init_method:
        :param generation_method:
        :param verbose:
        :return:
        '''
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()

        sentences = []

        for batch_n in range(n_batches):
            batch = self.generate_batch(seed_text, batch_size, max_len, top_k, temperature, max_iter,burnin, masked_portion,print_every, verbose, init_method='masked', generation_method=generation_method)
            sentences += [detokenize(sent, self.tokenizer) for sent in batch]

            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()


        if save_to_path is not None:
            with open(save_to_path, 'a') as f:
                for sent in sentences:
                    f.write(sent + '\n')

        return sentences


    def generate_batch(self, seed_text, batch_size, max_len=15, top_k=0, temperature=None, max_iter=300,
                       burnin=200, masked_portion=0.15, print_every=50, verbose=True, init_method='masked', generation_method='parallel'):
        """ Generate for one random position at a timestep
        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """

        seed_text = self.tokenizer.tokenize(seed_text)
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size, method=init_method)


        num_mask = 1 if generation_method == 'parallel sequential' else int(max_len * masked_portion)
        p = np.full() if generation_method == 'attention' else None


        with torch.no_grad():
            for ii in range(max_iter):

                print(f"ITER [{ii}]")
                # 1. Select indices to replace
                idx_to_replace = self.__select_tokens_to_replace(num_mask, max_len,batch_size, ii, seed_len,  p=p, generation_method=generation_method)
                # print(f'\t indices = {idx_to_replace[0]}')

                # 2. Replace with mask
                self.__replace_tokens(batch, idx_to_replace, tokens=self.tokenizer.mask_token_id)
                print('\t'+self.tokenizer.decode(batch[0]))

                # 3. Sample new tokens
                out = self.model(batch)
                logits = out['logits']

                if generation_method == 'attention':
                    attentions = out['attentions']
                    p = self.__generate_probs()

                # del out

                topk = top_k if (ii >= burnin) else 0
                idxs = self.generate_step(logits, gen_idx=idx_to_replace, top_k=topk, temperature=temperature, sample=(ii < burnin))

                # 4. Replace tokens
                self.__replace_tokens(batch, idx_to_replace, tokens=idxs)
                print('\t'+self.tokenizer.decode(batch[0]))
                print('\n\n')

                # if verbose and np.mod(ii + 1, print_every) == 0:
                #     print("iter", ii + 1, detokenize(self.tokenizer.convert_ids_to_tokens(batch[0]), self.tokenizer))

        return untokenize_batch(batch, self.tokenizer)


    def __select_tokens_to_replace(self, num_mask=None, max_len=None, batch_size=None, ii=None, seed_len=None, p = None, generation_method='parallel sequential'):
        if generation_method == "sequential":
            kk = [[ii % max_len] for _ in range(batch_size)]
        elif generation_method == "attention":
            kk = np.random.choice(0, max_len, (batch_size, num_mask), p=p)
        else:
            kk = np.random.randint(0, max_len, (batch_size, num_mask))

        return np.array(kk) + seed_len


    def __replace_tokens(self, batch, idx_to_replace, tokens):
        rows_idx = np.repeat(range(len(batch)), idx_to_replace.shape[-1]).reshape(idx_to_replace.shape)

        if type(tokens) is not int:
            tokens = tokens.reshape(idx_to_replace.shape)
        batch[rows_idx, idx_to_replace] = tokens

    def __generate_probs(self):
        return None


    def get_init_text(self, seed_text, max_len, batch_size=1, method='', masked_prob=0.9):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """

        seed_text = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token] + seed_text)

        if method == 'masked':
            batch = [seed_text + [self.tokenizer.mask_token_id] * max_len + [self.tokenizer.sep_token_id] for _ in range(batch_size)]
        elif method == 'random':
            batch = [seed_text + np.random.randint(0, self.tokenizer.vocab_size, max_len).tolist() + [self.tokenizer.sep_token_id] for _ in range(batch_size)]
        elif method == 'mixed':
            p = [(1 - masked_prob) / (self.tokenizer.vocab_size - 1)] * self.tokenizer.vocab_size
            p[self.tokenizer.mask_token_id] = masked_prob

            batch = [seed_text + np.random.choice(np.arange(self.tokenizer.vocab_size), max_len, p=p).tolist() + [self.tokenizer.sep_token_id] for _ in range(batch_size)]


        return torch.tensor(batch).to(self.device)


    def generate_step(self, out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]
        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        if type(gen_idx) is int:
            gen_idx = np.array(gen_idx)

        rows_idx = np.repeat(range(len(out)), gen_idx.shape[-1]).reshape(gen_idx.shape)

        logits = out[rows_idx, gen_idx]

        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=-1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)

        return idx

    def predict_masked(self, text, target=None):
        tokenized_text = self.tokenizer.tokenize(text)
        # tokenized_text = self.tokenizer.encode(text)

        if target is not None:
            masked_index = tokenized_text.index(self.tokenizer.tokenize(target)[0])
            tokenized_text[masked_index] = self.tokenizer.mask_token

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)
        for i in range(len(segments_ids)):
            segments_ids[i] = 0
            if tokenized_text[i] == self.tokenizer.sep_token:
                break
        segments_ids

        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)

        with torch.no_grad():
            out = self.model(tokens_tensor, segments_tensors)['logits']

            predicted_index = torch.argmax(out[0, masked_index]).item()
            predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])

            print("\n\nOriginal:", text)
            print("Masked:", " ".join(tokenized_text))

            print("Predicted token:", predicted_token)
            print("Other options:")
            # just curious about what the next few options look like.
            for i in range(10):
                out[0, masked_index, predicted_index] = -11100000
                predicted_index = torch.argmax(out[0, masked_index]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
                print(predicted_token)


if __name__ == '__main__':

    # model initialization
    en_bert_model = BertTextGenerator('bert-base-uncased')
    it_bert_model = BertTextGenerator("Musixmatch/umberto-wikipedia-uncased-v1")

    # masked prediction
    en_text = "He was sitting. [SEP] although he had already eaten a large meal, he was still very hungry."
    en_target = "meal"
    en_bert_model.predict_masked(en_text, target=en_target)

    it_text = 'In geometria, la curva di Peano è una curva che "ricopre" interamente un quadrato. [SEP] È stata la prima curva con questa proprietà ad essere scoperta da Giuseppe Peano nel 1890'
    it_target = "curva"
    it_bert_model.predict_masked(it_text, target=it_target)

    # text generation
    parameters = {'n_samples': 10,  # 1000
                  'batch_size': 5,  # 50
                  'max_len': 15,
                  'top_k': 100,
                  'temperature': 1,
                  'burnin': 100,
                  'sample': True,
                  'max_iter': 100,
                  'seed_text': "",
                  'init_method': 'masked',
                  'generation_method': "sequential"
                  }

    # "key1=val1_key2=val2_...txt"
    # file_path = "_".join([f"{k}={v}" for k, v in parameters.items()])+".txt"
    file_path = None
    print('\n\n ENGLISH TEXT GENERATION')
    en_bert_sents = en_bert_model.generate(save_to_path=file_path, **parameters)
    print("\nEnglish text generated: ")
    for sent in en_bert_sents:
        print(f"\t{sent}")

    # print('\n\n ITALIAN TEXT GENERATION')
    # it_bert_sents = it_bert_model.generate(**parameters)
    # print("\nItalian text generated: ")
    # for sent in it_bert_sents:
    #    print(f"\t{sent}")