import requests
import numpy as np
# from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder, ProtTransT5XLU50Embedder
from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import concurrent.futures
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool


ENCODERS = {
    # "seqvec": SeqVecEmbedder(),
    # "prottrans_bert_bfd": ProtTransBertBFDEmbedder(),
    # "prottrans_t5_xl_u50": ProtTransT5XLU50Embedder(),
    "prot_t5": {
        "tokenizer": T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False),
        "model": T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
    },
    "prost_t5": {
        "tokenizer": T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False),
        "model": T5EncoderModel.from_pretrained("Rostlab/ProstT5")
    }
}


def drugbank2smiles(drugbank_id):
    url = f"https://go.drugbank.com/drugs/{drugbank_id}.smiles"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        # print(f"Failed to get SMILES for {drugbank_id}")
        return None


def uniprot2sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        # Extract sequence from FASTA format
        sequence = "".join(response.text.split("\n")[1:])
        return sequence
    else:
        # print(f"Failed to get sequence for {uniprot_id}")
        return None


def encode_sequences(sequences: list, encoder: str):
    if encoder not in ENCODERS.keys():
        raise ValueError(f"Invalid encoder: {encoder}")
    
    model = ENCODERS[encoder]["model"]
    tokenizer = ENCODERS[encoder]["tokenizer"]  

    # Cache for storing encoded sequences
    cache = {}

    def encode_sequence(sequence: str):
        if sequence is None:
            return None
        if len(sequence) <= 3:
            raise ValueError(f"Invalid sequence: {sequence}")
        # Check if the sequence is already in the cache
        if sequence in cache:
            return cache[sequence]
        else:
            # Encode the sequence and store it in the cache
            try:
                encoded_sequence = model.embed(sequence)
                encoded_sequence = np.mean(encoded_sequence, axis=0)
                cache[sequence] = encoded_sequence
                return encoded_sequence
            except Exception as e:
                print(f"Failed to encode sequence: {sequence}")
                print(e)
                return None
                
    def encode_sequence_device_failover(sequence: str, function, timeout: int = 120):
        if sequence is None:
            return None
        
        if sequence in cache:
            return cache[sequence]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        try:
            # Try to process using GPU
            result = function(sequence, device)
        except RuntimeError as e:
            print(e)
            return None
            if "CUDA out of memory." in str(e):
                print("Trying on CPU instead.")
                device = torch.device("cpu")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(function, sequence, device)
                    try:
                        result = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        print(f"CPU encoding timed out.")
                        cache[sequence] = None
                        return None
            else:
                cache[sequence] = None
                raise Exception(e)
        except Exception as e:
            print(f"Failed to encode sequence: {sequence}")
            cache[sequence] = None
            return None
        
        cache[sequence] = result
        return result

    def encode_sequence_hf_3d(sequence, device):
        sequence_1d_list = [sequence]
        model.full() if device == "cpu" else model.half()
        model.to(device)

        ids = tokenizer.batch_encode_plus(
            sequence_1d_list,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding = model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )

        # Skip the first token, which is the special token for the entire sequence and mean pool the rest
        assert embedding.last_hidden_state.shape[0] == 1

        encoded_sequence = embedding.last_hidden_state[0, 1:-1, :]
        encoded_sequence = encoded_sequence.mean(dim=0).cpu().numpy().flatten()

        assert encoded_sequence.shape[0] == 1024
        return encoded_sequence
            
    def encode_sequence_hf(sequence, device):
        sequence_1d_list = [sequence]
        model.full() if device == "cpu" else model.half()
        model.to(device)

        ids = tokenizer.batch_encode_plus(
            sequence_1d_list,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding = model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )
        
        assert embedding.last_hidden_state.shape[0] == 1

        encoded_sequence = embedding.last_hidden_state[0, :-1, :]
        encoded_sequence = encoded_sequence.mean(dim=0).cpu().numpy().flatten()

        assert encoded_sequence.shape[0] == 1024
        return encoded_sequence

    # Use list comprehension to encode all sequences, utilizing the cache
    if encoder == "seqvec":
        raise NotImplementedError("SeqVec is not supported")
        seq = encoder_function.embed(list(sequences))
        seq = np.sum(seq, axis=0)

    if encoder == "prost_t5":
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        # The direction of the translation is indicated by two special tokens:
        # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
        # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
        sequences = ["<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in sequences]
        seq = [encode_sequence_device_failover(sequence, encode_sequence_hf_3d) for sequence in tqdm(sequences, desc="Encoding sequences")]

    elif encoder == "prot_t5":
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        seq = [encode_sequence_device_failover(sequence, encode_sequence_hf) for sequence in tqdm(sequences, desc="Encoding sequences")]

    else:
        raise NotImplementedError("SeqVec is not supported")
        seq = [encode_sequence(sequence) for sequence in sequences]
        
    return np.array(seq)


class SequenceEncoder:
    def __init__(self, encoder: str):
        if encoder not in ENCODERS:
            raise ValueError(f"Invalid encoder: {encoder}")
        self.encoder = encoder
        self.model = ENCODERS[encoder]["model"]
        self.tokenizer = ENCODERS[encoder]["tokenizer"]
        self.cache = {}

    def encode_sequence(self, sequence: str):
        if sequence is None:
            return None
        if len(sequence) <= 3:
            raise ValueError(f"Invalid sequence: {sequence}")
        
        if sequence in self.cache:
            return self.cache[sequence]
        
        try:
            encoded_sequence = self.model.embed(sequence)
            encoded_sequence = np.mean(encoded_sequence, axis=0)
            self.cache[sequence] = encoded_sequence
            return encoded_sequence
        except Exception as e:
            print(f"Failed to encode sequence: {sequence}")
            print(e)
            return None
    
    def encode_sequence_device_failover(self, sequence: str, function, timeout: int = 5):
        if sequence is None:
            return None
        
        if sequence in self.cache:
            return self.cache[sequence]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        
        try:
            result = function(sequence, device)
        except RuntimeError as e:
            return None
            print(e)
            if "CUDA out of memory." in str(e):
                print("Trying on CPU instead.")
                device = torch.device("cpu")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(function, sequence, device)
                    try:
                        result = future.result(timeout=timeout)
                    except:
                        print(f"CPU encoding timed out.")
                        self.cache[sequence] = None
                        return None
                    finally:
                        executor.shutdown(wait=False)
            else:
                self.cache[sequence] = None
                return None
        except Exception as e:
            print(f"Failed to encode sequence: {sequence}")
            self.cache[sequence] = None
            return None
        
        self.cache[sequence] = result
        return result

    def encode_sequence_hf_3d(self, sequence, device):
        sequence_1d_list = [sequence]
        self.model.full() if device == "cpu" else self.model.half()
        self.model.to(device)

        ids = self.tokenizer.batch_encode_plus(
            sequence_1d_list,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding = self.model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )

        assert embedding.last_hidden_state.shape[0] == 1

        encoded_sequence = embedding.last_hidden_state[0, 1:-1, :]
        encoded_sequence = encoded_sequence.mean(dim=0).cpu().numpy().flatten()

        assert encoded_sequence.shape[0] == 1024
        return encoded_sequence

    def encode_sequence_hf(self, sequence, device):
        sequence_1d_list = [sequence]
        self.model.full() if device == "cpu" else self.model.half()
        self.model.to(device)

        ids = self.tokenizer.batch_encode_plus(
            sequence_1d_list,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            embedding = self.model(
                ids.input_ids,
                attention_mask=ids.attention_mask
            )
        
        assert embedding.last_hidden_state.shape[0] == 1

        encoded_sequence = embedding.last_hidden_state[0, :-1, :]
        encoded_sequence = encoded_sequence.mean(dim=0).cpu().numpy().flatten()

        assert encoded_sequence.shape[0] == 1024
        return encoded_sequence

    def encode_sequences(self, sequences: list):
        if self.encoder == "seqvec":
            raise NotImplementedError("SeqVec is not supported")
            seq = self.encoder_function.embed(list(sequences))
            seq = np.sum(seq, axis=0)

        elif self.encoder == "prost_t5":
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
            sequences = ["<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in sequences]
            seq = [self.encode_sequence_device_failover(sequence, self.encode_sequence_hf_3d) for sequence in tqdm(sequences, desc="Encoding sequences")]

        elif self.encoder == "prot_t5":
            sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
            seq = [self.encode_sequence_device_failover(sequence, self.encode_sequence_hf) for sequence in tqdm(sequences, desc="Encoding sequences")]

        else:
            raise NotImplementedError("SeqVec is not supported")
            seq = [self.encode_sequence(sequence) for sequence in sequences]

        if any([x is None for x in seq]):
            return seq
        else:
            return np.array(seq)
