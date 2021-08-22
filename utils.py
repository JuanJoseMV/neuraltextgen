
import numpy as np
import datetime


def print_batch(tokenizer, batch, n, header=None):
    '''
    print a batch of tokens. Used mainly for debugging
    Parameters
    ------------
    tokenizer : Tokenizer (https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.Tokenizer)

    batch : List of List[int]

    n : int
        number of sentences to print from the batch
    header : str
        header of the batch printed before the sentences
    '''
    print(f'=== {header or "Batch"} ===')
    print(tokenizer.batch_decode(batch[:n], skip_special_tokens=True))
    print('...\n' if n < len(batch) else '')



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))