import os
import requests
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp

# --- init the tokenizer --- #
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token id, 50256

def tokenize(doc): # tokenize a single document, and return a numpy array of uint16 token
    tokens = [eot] # end of text token to separate documents
    tokens.extend(enc.encode_ordinary(doc["text"])) # encode the text and add to tokens list
    tokens_np = np.array(tokens)
    
    # data type optimization: since the token ids are all less than 65536, we can use uint16 to save memory
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "[WARNING] token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np): # write a numpy array of tokens to a binary file(more efficient than text file)
    with open(filename, 'wb') as f:
        f.write(tokens_np.tobytes())

# --- main data preparation function --- #
def prepare_data(name='tiny_shakespeare'):
    # --- Tiny Shakespeare dataset preparation --- #
    if name == 'tiny_shakespeare':
        base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        data_path = os.path.join(base_dir, 'data/Tiny_Shakespeare/input.txt')

        if not os.path.exists(data_path): # Download the dataset if it doesn't exist
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(data_path, 'w') as f:
                f.write(requests.get(text_url).text)

        print(f"[INFO] [Tiny Shakespeare] dataset downloaded and saved to: {data_path}")
    
    # --- FineWeb-Edu dataset preparation --- #
    if name == 'fineweb-edu-10BT':
        """
        FineWeb-Edu dataset preparation script
        https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
        Download the dataset and save it to the specified path.
        Run simply with:
        $ python prepare.py
        """
        local_dir = "data/FineWeb-edu-10BT"
        remote_name = "sample-10BT" # Choose 10BT size
        shard_size = int(1e8) # 100MB per shard, shard for that the dataset is too large to fit in memory, we need to split it into smaller chunks
        DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        # download the dataset using HuggingFace Datasets library when the dataset is not already downloaded and cached
        if not os.path.exists(os.path.join(DATA_CACHE_DIR, remote_name)):
            fineweb = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
        print(f"[INFO] [FineWeb-Edu-10BT] dataset loaded, starting tokenization...")

        # tokenize all docuements and write outputs shards
        nprocs = max(1, mp.cpu_count() // 2) # leave at least one CPU core free for other tasks
        with mp.Pool() as pool: # mp to speed up tokenization by using multiple CPU cores, since the dataset is large and tokenization can be time-consuming
            shard_id = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16) # pre-allocate a large numpy array for the current shard
            tokens_count = 0 # count the number of tokens in the current shard
            progress_bar = None
            for tokens in pool.imap(tokenize, fineweb, chunksize=16): # tokenize documents in parallel
                if tokens_count + len(tokens) < shard_size: # simply append tokens to the current shard if it doesn't exceed the shard size
                    all_tokens_np[tokens_count:tokens_count+len(tokens)] = tokens
                    tokens_count += len(tokens)
                    # update the progress bar after processing each document
                    if progress_bar is None: # initialize the progress bar on the first iteration
                        progress_bar = tqdm(total=shard_size, unit='tokens', desc=f"Shard {shard_id}") if progress_bar is None else progress_bar
                    progress_bar.update(len(tokens))
                else:
                    # write the current shard to a file and start a new shard if adding the current tokens would exceed the shard size
                    split = 'val' if shard_id == 0 else 'train' # use the first shard as validation set, and the rest as training set
                    filename = os.path.join(DATA_CACHE_DIR, f"{remote_name}_{split}_shard{shard_id:06d}.npy") # save as .npy file for faster loading later
                    remainder = shard_size - tokens_count # calculate how many tokens can be added to the current shard
                    progress_bar.update(remainder) # update the progress bar for the current shard
                    all_tokens_np[tokens_count:tokens_count+remainder] = tokens[:remainder] # fill the current shard with as many tokens as possible
                    write_datafile(filename, all_tokens_np) # write the current shard to a file
                    shard_id += 1 # move to the next shard
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:] # start the new shard with the remaining tokens
                    tokens_count = len(tokens) - remainder # update the token count for the new shard
               
            # write the last shard if there are any remaining tokens after processing all documents
            if tokens_count != 0:
                split = 'val' if shard_id == 0 else 'train'
                filename = os.path.join(DATA_CACHE_DIR, f"{remote_name}_{split}_shard{shard_id:06d}.bin")
                write_datafile(filename, all_tokens_np[:tokens_count]) # write only the valid portion of the last shard
        print(f"[INFO] Finished tokenization and sharding for {name} dataset.")

if __name__ == "__main__":
    prepare_data(name='fineweb-edu-10BT')
