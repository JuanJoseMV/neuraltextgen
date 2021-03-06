import math
import time
import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from textprocessing import *
from utils import *

try:
    from apex import amp

    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertTextGenerator:
    def __init__(self, model_version, device=DEFAULT_DEVICE, use_apex=APEX_AVAILABLE, use_fast=True,
                 do_basic_tokenize=True):
        """
        Wrapper of a BERT model from AutoModelForMaskedLM from huggingfaces.
        This class implements methods to generate text with the BERT module
        Parameters
        ----------
        model_version : str
            The name of the BERT model to initialize form AutoModelForMaskedLM
        device : str
            Type of pytorch device to adopt. By default is set to DEFAULT_DEVICE
            that is 'cuda' if cuda is available otherwise is 'cpu'
        use_apex : boolean
            Flag to adopt nvidia apex
        """
        self.device = device
        self.model_version = model_version
        self.model = AutoModelForMaskedLM.from_pretrained(model_version, output_attentions=True)
        self.model.to(self.device)
        self.use_apex = use_apex

#         Move to finetune
        if use_apex:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
                                                   loss_scale="dynamic")

        self.tokenizer = AutoTokenizer.from_pretrained(model_version, do_lower_case="uncased" in model_version,
                                                       use_fast=use_fast,
                                                       do_basic_tokenize=do_basic_tokenize)  # added to avoid splitting of unused tokens
        self.num_attention_masks = len(self.model.base_model.base_model.encoder.layer)
        self.has_format_tokenizer = False

    def generate(self, save_to_path=None, n_sentences=100, seed_text="", batch_size=10, max_iter=500, verbose=False,
                 print_every=50, max_len=40, min_len=4, avg_len=20, std_len=4, init_mask_prob=1,
                 generation_method="parallel", masked_portion=1, temperature=1.0, sample=True, top_k=100, burnin=None):
        '''
        Principal method of the class, used to generate sentences. The methodology used to generate a batch of sentences
        can be decomposed into 3 main points:
            1) Initialization: each batch is initialized as a matrix of tokens where each row represent a sentence
            2) Selection: for each iteration and for each sentence one or more tokens are selected and masked
            3) Sampling: for each iteration BERT is used to compute logits of the masked tokens that are then used to sample
                        new tokens that will replace the masked ones
        Parameters
        ==============================
        (General)
        ------------------------------
        save_to_path: str, default = None
            path of txt file where to store the sentences generated
        n_sentences: int, default = 100
            total number of sentences to generate
        seed_text: str, default = ""
            Initial text used to generate the sentences
        batch_size: int, default = 10
            number of sentences for each batch
        max_iter: int, default = 300
            number of iterations
        verbose: boolean, default = False
        print_every:int, default = 50
            print a sample from the batch every print_every iteration.Used only if verbose is True
        (Length of the sentences)
        ------------------------------
        The method can generated sentences with different length. For each batch the len of the sentences in it
        is sampled from a normal distribution N(avg_len, std_len) and then rounded to the closest int.
        max_len and min_len are used to clip the length
        max_len: int, default = 40
            maximum length of each sentence
        min_len: int, default = 4
            minimum length of each sentence
        avg_len: float or int, default = 20
            average length of the sentences
        std_len: float or int, default = 4
            standard deviation of the sentences
        (Initialization)
        ------------------------------
        Each batch is initialized as a matrix of tokens of dimension (batch_size x batch_len + 2), where batch_len is
        selected as described above. At the beginning of each sentences is added a cls_token and at the end a sep_token.
        Each other token is selected based on the value of init_mask_prob:
            - if init_mask_prob == 1  -> each token is [MASK] with probability 1 (the batch is whole [MASK]s)
            - if init_mask_prob == 0  -> each token is selected as a random token in the tokenizer vocabulary (the batch is init as random sentences)
            - if init_mask_prob in (0, 1) -> each token is sampled as [MASK] with prob init_mask_prob or with probability
                                        (1 - init_mask_prob) as any other token in the tokenizer vocabulary
        init_mask_prob: float in [0,1], default = 1
            probability of the mask token
        (Selection)
        ------------------------------
        generation_method: str, default = "parallel"
            method used to select the tokens to replace at each iteration
            - 'parallel': for each sentence is selected randomly one token or a percentage of tokens based on the value of masked_portion
            - 'sequential': the tokens are selected sequentially. At iteration i the token in position i % batch_len is selected
            - 'attention': At the first iteration one token is selected randomly for each sentence. In later iterations
                        for each sentence the token is selected with probabilty distribution based on the attention mask
                        of the token sampled in the previous iteration
        masked_portion: int or float in [0, 1], default = 1
            percentage of tokens to mask for each sentence. Used only if generation_method is 'parallel'
        (Sampling)
        ------------------------------
        temperature: float, default = 1
            temperature for logits ( logits <- logits/temperature)
        sample: boolean, default = True
            when sample is True each masked token is replaced sampling randomly according to the corresponding logits
        top_k: int or None, default = 100
             when top_k > 0  each masked token is replaced sampling randomly according to the logits considering
             only the top_k tokens. If setted to None all the tokens will be considered
        burnin: int, default = None
            after burnin iterations the tokens will be chosen determinsitically selecting the one with maximum
            logit score
        Returns
        -------
        list
            a list of sentences (str) already detokenized and cleaned
        '''

        n_batches = math.ceil(n_sentences / batch_size)

        if burnin is None:
            burnin = max_iter

        sentences = []

        for batch_n in range(n_batches):
            batch_sentence_len = np.round(np.random.normal(avg_len, std_len))
            batch_sentence_len = int(np.clip(batch_sentence_len, min_len, max_len))

            # Generate and append batch of sentences
            sentences += self.generate_batch(seed_text, batch_size, max_iter, verbose=verbose, print_every=print_every,
                                             sent_len=batch_sentence_len, init_mask_prob=init_mask_prob,
                                             generation_method=generation_method,
                                             masked_portion=masked_portion, temperature=temperature, sample=sample,
                                             top_k=top_k, burnin=burnin)

            # Print if verbose
            if verbose and (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()

        # Store results
        if save_to_path is not None:
            with open(save_to_path, 'w') as f:
                for sent in sentences:
                    f.write(sent + '\n')

        return sentences

    def generate_batch(self, seed_text, batch_size, max_iter, verbose, print_every, sent_len, init_mask_prob,
                       generation_method, masked_portion, temperature, sample, top_k, burnin):

        # Init batch
        seed_text = self.tokenizer.tokenize(
            self.tokenizer.cls_token + seed_text)  # add [CLS] token at the beggining of the seed_text
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, sent_len, batch_size, init_mask_prob)

        # Init sampling parameters
        if generation_method == "parallel":
            if type(masked_portion) is int:
                num_mask = masked_portion
            else:
                num_mask = int(np.round(sent_len * masked_portion))
            list_probs = None
        elif generation_method == "sequential":
            list_probs = None
            num_mask = 1
        else:
            # One probability distribution for each sentence in the batch (initially uniform among all tokens)
            num_mask = 1
            list_probs = [np.full(sent_len, 1.0 / sent_len)] * batch_size
            counter = np.zeros((batch_size, sent_len))

        with torch.no_grad():
            for ii in range(max_iter):

                # 1. Select indices to replace
                idx_to_replace = self.__select_tokens_to_replace(generation_method, sent_len, batch_size, num_mask, ii,
                                                                 seed_len, list_probs)

                # 2. Replace with mask
                self.__replace_tokens(batch, idx_to_replace, tokens=self.tokenizer.mask_token_id)

                # 3. Sample new tokens
                out = self.model(batch)
                logits = out['logits']

                if generation_method == 'attention':
                    counter[np.arange(batch_size), idx_to_replace.flatten() - seed_len] += 1
                    attentions = torch.stack(out['attentions'])
                    list_probs = self.__compute_probs(attentions, batch_size, idx_to_replace, seed_len, counter)

                sample = False if ii >= burnin else sample
                idxs = self.generate_step(logits, gen_idx=idx_to_replace, temperature=temperature, sample=sample,
                                          top_k=top_k)

                # 4. Replace tokens
                self.__replace_tokens(batch, idx_to_replace, tokens=idxs)

                if verbose and ii % print_every == 0:
                    print_batch(self.tokenizer, batch, 3)

        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def get_init_text(self, seed_text, sent_len, batch_size, init_mask_prob):
        """ Get initial sentence by padding seed_text with either masks or random words to sent_len """

        seed_text = self.tokenizer.convert_tokens_to_ids(seed_text)

        if init_mask_prob == 1:
            batch = [seed_text + [self.tokenizer.mask_token_id] * sent_len + [self.tokenizer.sep_token_id] for _ in
                     range(batch_size)]
        elif init_mask_prob == 0:
            batch = [seed_text + np.random.randint(0, self.tokenizer.vocab_size, sent_len).tolist() + [
                self.tokenizer.sep_token_id] for _ in range(batch_size)]
        else:
            p = [(1 - init_mask_prob) / (self.tokenizer.vocab_size - 1)] * self.tokenizer.vocab_size
            p[self.tokenizer.mask_token_id] = init_mask_prob

            batch = [seed_text + np.random.choice(np.arange(self.tokenizer.vocab_size), sent_len, p=p).tolist() + [
                self.tokenizer.sep_token_id] for _ in range(batch_size)]

        return torch.tensor(batch).to(self.device)

    def __select_tokens_to_replace(self, generation_method, sent_len, batch_size, num_mask, ii, seed_len, list_probs):
        if generation_method == "sequential":
            kk = [[ii % sent_len] for _ in range(batch_size)]
        elif generation_method == "attention":
            kk = [np.random.choice(range(sent_len), num_mask, p=p).tolist() for p in list_probs]
        elif generation_method == 'parallel':
            #             kk = np.random.randint(0, sent_len, (batch_size, num_mask))
            x = np.random.randint(0, sent_len)
            kk = [[x] for _ in range(batch_size)]
        #         elif generation_method == 'parallel original':
        #             x = np.random.randint(0, sent_len)
        #             kk = [[x] for _ in range(batch_size)]

        return np.array(kk) + seed_len

    def __replace_tokens(self, batch, idx_to_replace, tokens):
        rows_idx = np.repeat(range(len(batch)), idx_to_replace.shape[-1]).reshape(idx_to_replace.shape)

        if type(tokens) is not int:
            tokens = tokens.reshape(idx_to_replace.shape)

        batch[rows_idx, idx_to_replace] = tokens

    def __compute_probs(self, attentions, batch_size, idx, seed_len, counter):
        ''' compute probabilities from attention masks'''
        # list_probs = []
        #
        # # attentions has dimension (batch_size, num_attention_masks, sentence_len, sentence_len)
        # for i in range(batch_size):
        #     average_prob = attentions[i, :, idx[i], :].mean(axis=0).flatten().cpu().numpy()
        #     average_prob = average_prob[seed_len:-1]  # avoid seed_text and last token ([SEP])
        #     average_prob = average_prob / average_prob.sum()  # normalize
        #     list_probs.append(average_prob)
        #
        # return list_probs

        avg_attentions = attentions.mean(axis=(0, 2)).cpu().detach().numpy() # mean through encoders and attention masks
        avg_attentions = avg_attentions[np.arange(batch_size),seed_len:-1,idx.flatten()]  # for each sentence extract the
                                                                                  # attention corresponding to the
                                                                                  # masked token (avoiding special tokens and seed)


        c = counter + 1
        prob = avg_attentions / c

        return prob / prob.sum(axis=1)[:, np.newaxis]
    # def counter_penalization(attention, idx_mask, counter, **kwargs):
    #     a = attention.mean(
    #         axis=(0, 1)).cpu().detach().numpy()  # mean over ax0 that is encoders and ax1 that is attention_mask
    #     a = a[1:-1, idx_mask].reshape(-1, 1)
    #     c = np.array(counter) + 1
    #     prob = a.flatten() / c
    #     prob = prob / sum(prob)
    #     return prob


    def generate_step(self, out, gen_idx, temperature=1, sample=True, top_k=None):
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

        if sample:
            # general sampling
            if top_k is None:
                dist = torch.distributions.categorical.Categorical(logits=logits)
                idx = dist.sample().squeeze(-1)
            # top_k sampling
            else:
                kth_vals, kth_idx = logits.topk(top_k, dim=-1)
                dist = torch.distributions.categorical.Categorical(logits=kth_vals)
                idx = kth_idx.gather(dim=-1, index=dist.sample().unsqueeze(-1)).squeeze(-1)

        # burnin - deterministic
        else:
            idx = torch.argmax(logits, dim=-1)

        return idx

    def finetune(self, sentences, labels=None, encoded_dict=None, mask_percentage=0.15, epochs=4, batch_size=32,
                 optimizer=AdamW, optimizer_parameters=dict(lr=2e-5, eps=1e-8),
                 scheduler=get_linear_schedule_with_warmup, scheduler_parameters=dict(num_warmup_steps=0),
                 num_tokens_per_class=3
                 ):

        if encoded_dict is None:
            # set encoder
            if labels is None:
                self.encoder = Encoder(self.tokenizer)
                encoded_dict = self.encoder.encode(sentences)
            else:
                classes = np.unique(labels)
                self.encoder = LabelEncoder(self.model, self.tokenizer, classes=classes,
                                            num_tokens_per_class=num_tokens_per_class)
                encoded_dict = self.encoder.encode(sentences, labels)


        # Retrieve tokenized sentences and attention masks
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

        # Setting optimizer and scheduler
        optimizer = optimizer(self.model.parameters(), **optimizer_parameters)
        if self.use_apex:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O2", keep_batchnorm_fp32=True,
                                                   loss_scale="dynamic")

        total_steps = len(dataloader) * epochs
        scheduler = scheduler(optimizer, num_training_steps=total_steps, **scheduler_parameters)

        # TODO add stats
        training_stats = []
        test_stats = []
        total_t0 = time.time()

        self.model.train()

        for epoch_i in range(0, epochs):

            print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
            print('Training...')

            t0 = time.time()
            total_train_loss = 0

            for step, batch in enumerate(dataloader):

                if step % 25 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))

                batch_input = batch[0].to(self.device)
                batch_attention = batch[1].to(self.device)

                # 512 to truncate max bert input
                if len(batch[0]) > 512:
                    batch_input = batch_input[:, :512]
                    batch_attention = batch_attention[:, :512]

                # Computing number to tokens to mask based on mask_percentage
                num_sent, num_tokens = batch_input.shape
                num_tokens_to_mask = int(mask_percentage * num_tokens)

                # Generating randomly num_tokens_to_mask to mask for each sentence, considering only real tokens
                # (not [CLS] nor label-tokens that are at the beginning of the sentence)
                start_id = 1 + num_tokens_per_class    # mask only
                batch_mask_ids = torch.randint(start_id, num_tokens - 1, size=(num_sent, num_tokens_to_mask))

                #  Each sentence needs to be indexed num_tokens_to_mask times.
                #  This array is of the type [0,0,0 ..., 1,1,1, ..., 2,2,2, ... num_sentences -1]
                sentence_ids =  np.repeat(np.arange(len(batch_input)), num_tokens_to_mask)

                # Retrieve the original tokens to mask:
                batch_masked_tokens = batch_input[sentence_ids, batch_mask_ids.flatten()]

                # Mask the tokens
                batch_input[sentence_ids, batch_mask_ids.flatten()] = self.tokenizer.mask_token_id


                # Forward pass
                self.model.zero_grad()
                result = self.model(batch_input, attention_mask=batch_attention, return_dict=True)
                logits = result['logits']

                # Retrieve logits only for masked tokens. logits is a tensor of dim [batch_size, num_tokens, len_vocab]
                # logits = logits[np.concatenate([[i] * batch_mask_ids.shape[1] for i in range(len(batch_mask_ids))], 0),
                #          batch_mask_ids.flatten(), :]
                logits = logits[sentence_ids, batch_mask_ids.flatten(), :]


                loss = F.cross_entropy(logits, batch_masked_tokens.flatten())
                total_train_loss += loss.item()

                # Backward pass
                if self.use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                # Restoring masked tokens
                batch_input[sentence_ids, batch_mask_ids.flatten()] = batch_masked_tokens.flatten()

            avg_train_loss = total_train_loss / len(dataloader)
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == '__main__':

    # model initialization
    en_bert_model = BertTextGenerator('bert-base-uncased')

    # text generation
    parameters = {'n_sentences': 10,  # 1000
                  'seed_text': "",
                  'batch_size': 10,  # 50
                  'max_iter': 150,
                  'init_mask_prob': 1,
                  'generation_method': "attention",
                  'masked_portion': 1,
                  'temperature': 1,
                  'sample': True,
                  'top_k': 100,
                  }

    file_path = None
    print('\n\n ENGLISH TEXT GENERATION')
    en_bert_sents = en_bert_model.generate(save_to_path=file_path, **parameters)
    print("\nEnglish text generated: ")
    for sent in en_bert_sents:
        print(f"\t{sent}")