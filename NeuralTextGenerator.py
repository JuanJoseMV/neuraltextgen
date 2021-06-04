import math
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
    def __init__(self, model_version, device = DEFAULT_DEVICE, use_apex = False):
        self.device = device
        self.model_version = model_version
        self.model = AutoModelForMaskedLM.from_pretrained(model_version)
        self.model.to(self.device)

        if use_apex and APEX_AVAILABLE:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
                                                   loss_scale="dynamic")

        self.tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case= "uncased" in model_version)


    def parallel_sequential_generation(self, seed_text, batch_size, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                       cuda=False, print_every=50, verbose=True, init_method='masked'):
        """ Generate for one random position at a timestep
        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_text = self.tokenizer.tokenize(seed_text)
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size, method=init_method)

        for ii in range(max_iter):
            kk = np.random.randint(0, max_len)

            for jj in range(batch_size):

                batch[jj][seed_len + kk] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]

            with torch.no_grad():
                inp = torch.tensor(batch).to(self.device)
                out = self.model(inp)['logits']
                topk = top_k if (ii >= burnin) else 0
                idxs = self.generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
                for jj in range(batch_size):
                    batch[jj][seed_len + kk] = idxs[jj]

                if verbose and np.mod(ii + 1, print_every) == 0:
                    for_print = self.tokenizer.convert_ids_to_tokens(batch[0])
                    for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]
                    print("iter", ii + 1, " ".join(for_print))
                    print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))

        return untokenize_batch(batch, self.tokenizer)
    
    def parallel_generation(self, batch_size, seed_text, max_len=15, top_k=0, temperature=None, max_iter=300, sample=True, 
                        cuda=False, print_every=10, verbose=True, init_method='masked'):
        """ Generate for all positions at a time step """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size, method = init_method)
    
        for ii in range(max_iter):
            inp = torch.tensor(batch).to(self.device)
            out = model(inp)
            for kk in range(max_len):
                idxs = generate_step(out, gen_idx=seed_len+kk, top_k=top_k, temperature=temperature, sample=sample)
                for jj in range(batch_size):
                    batch[jj][seed_len+kk] = idxs[jj]
            
            if verbose and np.mod(ii, print_every) == 0:
                print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))
    
        return untokenize_batch(batch)
            
    def sequential_generation(self, seed_text, batch_size, max_len=15, leed_out_len=15, 
                          top_k=0, temperature=None, sample=True, cuda=False, verbose=True, print_every=10, init_method = "masked"):
        """ Generate one word at a time, in L->R order """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size, method = init_method)
            
        for ii in range(max_len):
            inp = [sent[:seed_len+ii+leed_out_len]+[sep_id] for sent in batch]
            inp = torch.tensor(batch).to(self.device)
            out = model(inp)
            idxs = generate_step(out, gen_idx=seed_len+ii, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len+ii] = idxs[jj]

            if verbose and np.mod(ii, print_every) == 0:
                print("iter", ii+1, " ".join(tokenizer.convert_ids_to_tokens(batch[0])))
        
        return untokenize_batch(batch)

    def generate(self, save_to_path=None, n_samples=100, seed_text="", batch_size=10, max_len=25, sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500, print_every=1, init_method='masked', generation_method = "parallel sequential", verbose = False):

        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()

        for batch_n in range(n_batches):
            if generation_method == "parallel sequential":
                batch = self.parallel_sequential_generation(self.tokenizer.cls_token+seed_text, max_len=max_len, top_k=top_k, batch_size=batch_size, 
                                                            temperature=temperature, burnin=burnin, max_iter=max_iter, verbose=verbose, init_method=init_method)
            elif generation_method == "sequential":
                batch = self.sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, temperature=temperature, leed_out_len=leed_out_len,
                                              sample=sample, init_method= init_method)
            elif generation_method == "parallel":
                batch = self.parallel_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k, temperature=temperature, sample=sample, max_iter=max_iter, init_method = init_method)

            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()

        sentences = [detokenize(sent, self.tokenizer) for sent in batch]

        if save_to_path is not None:
            with open(save_to_path, 'a') as f:
                for sent in sentences:
                    f.write(sent+'\n')

        return sentences



    def get_init_text(self, seed_text, max_len, batch_size=1, method='', masked_prob=0.9):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """

        if method == 'masked':
            batch = [seed_text + [self.tokenizer.mask_token] * max_len + [self.tokenizer.sep_token] for _ in range(batch_size)]
        elif method == 'random':
            batch = [seed_text + np.random.choice(list(self.tokenizer.vocab.keys()), max_len).tolist() + [self.tokenizer.sep_token] for _ in range(batch_size)]
        elif method == 'mixed':
            p = [(1 - masked_prob) / (self.tokenizer.vocab_size - 1)] * self.tokenizer.vocab_size
            p[self.tokenizer.mask_token_id] = masked_prob
            batch = [seed_text + self.tokenizer.convert_ids_to_tokens(
                np.random.choice(np.arange(self.tokenizer.vocab_size), max_len, p=p)) + [self.tokenizer.sep_token] for _
                     in range(batch_size)]


        return tokenize_batch(batch, self.tokenizer)


    def generate_step(self, out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]

        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx


    def predict_masked(self, text, target=None):
        tokenized_text= self.tokenizer.tokenize(text)
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
    en_bert_model= BertTextGenerator('bert-base-uncased')
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
                  'burnin': 50,
                  'sample': True,
                  'max_iter': 100,
                  'seed_text': "",
                  'init_method':'random'
                  }

    # "key1=val1_key2=val2_...txt"
    # file_path = "_".join([f"{k}={v}" for k, v in parameters.items()])+".txt"
    file_path = None
    print('\n\n ENGLISH TEXT GENERATION')
    en_bert_sents = en_bert_model.generate(save_to_path=file_path, **parameters)
    print("\nEnglish text generated: ")
    for sent in en_bert_sents:
        print(f"\t{sent}")


    print('\n\n ITALIAN TEXT GENERATION')
    it_bert_sents = it_bert_model.generate(**parameters)
    print("\nItalian text generated: ")
    for sent in it_bert_sents:
        print(f"\t{sent}")


