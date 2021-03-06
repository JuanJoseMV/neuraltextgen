import re
import numpy as np








class Formatter():

    def __init__(self, replace_tokens, unused_type='[unusedi]'):
        self.dict_token_replace = {k: ' ' + unused_type.replace('i', str(i + 1)) + ' ' for i, k in
                                   enumerate(replace_tokens)}

    def format(self, path, pattern):
        lines = []

        re_line = re.compile(pattern)
        with open(path, 'r') as f:
            for match in re_line.finditer(''.join(f.readlines())):
                line = match[0]

                # Replace
                for k, v in self.dict_token_replace.items():
                    line = line.replace(k, v)

                lines.append(line)

        return lines

    def unformat(self, sentences):
        unformatted_sentences = []
        for sent in sentences:
            # Replace
            for k, v in self.dict_token_replace.items():
                sent = sent.replace(v.strip(), k)

            unformatted_sentences.append(sent)

        return unformatted_sentences

class Encoder():
    def __init__(self, tokenizer):
        self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_model(self, model):
        self.model = model

    def encode(self, lines):

        encoded_dict = self.tokenizer.batch_encode_plus(
            lines,  # Sentence to encode.
            padding=True,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        return encoded_dict




class LabelEncoder(Encoder):
    def __init__(self, model, tokenizer, classes=[], num_tokens_per_class=3):

        super().__init__(tokenizer)
        self.set_model(model)

        # Preparing special tokens related to labels.
        # Each token is of the type 'cls-k' where cls is a class in classes and
        # k is an integer value in range(0, num_tokens_per_class)
        self.num_tokens_per_class = num_tokens_per_class
        self.label_special_tokens_dict = {cls: [f'[{cls}-{i}]' for i in range(num_tokens_per_class)] for cls in classes}
        self.label_special_tokens_list = np.concatenate([list(x) for x in self.label_special_tokens_dict.values()]).tolist()


        # Addd special tokens and replace vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.label_special_tokens_list})
        self.model.resize_token_embeddings(len(self.tokenizer))


    def encode(self, lines, labels):
        labeled_lines = [' '.join(self.label_special_tokens_dict[label]) + ' ' + line for line, label in zip(lines, labels)]
        return super().encode(labeled_lines)
