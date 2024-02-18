import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_trg,src_lang,trg_lang,seq_len)->None:
        
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.long)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.long)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.long)

    def __len__(self):
        return len(self.ds)
    def __getitem__(self,idx):
        src_target_pair = self.ds[idx]

        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.trg_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_trg.encode(trg_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 :
            raise ValueError("Padding Encoding tokens are not enough")
        if dec_num_padding_tokens < 0:
            raise ValueError("Padding Decoding tokens are not enough")
        
        encoder_input = torch.cat(
            [
               self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
            ]
        
        )
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ]
        )
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input":encoder_input,
            "decoder_input":decoder_input,
            "encoder_mask":(encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask":(decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len,seq_len)
            "label":label,
            "src_text":src_text,
            "trg_text":trg_text
        }

def causal_mask(seq_len:int)->torch.Tensor:
    mask = torch.tril(torch.ones(seq_len,seq_len),diagonal=1).type(torch.int)
    return mask == 0