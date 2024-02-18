import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset,causal_mask
from pathlib import Path
from model import Transformer
from config import get_config,get_weights_file_path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config,ds,lang):
    
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        print(f"Loading tokenizer for {lang}...")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset("opus_books",f'{config["lang_src"]}-{config["lang_tgt"]}',split="train")
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_trg = get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    # Keep 90% of the data for training and 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds= BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_trg,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_trg,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_length_src=0
    max_length_trg=0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_tgt']]).ids
        max_length_src = max(max_length_src,len(src_ids))
        max_length_trg = max(max_length_trg,len(trg_ids))

    print(f"Max length src: {max_length_src}")
    print(f"Max length trg: {max_length_trg}")

    train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_trg

    
def get_model(config, tokenizer_src, tokenizer_trg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_pad_idx = tokenizer_src.token_to_id("[PAD]")
    trg_pad_idx = tokenizer_trg.token_to_id("[PAD]")
    src_vocab_size = tokenizer_src.get_vocab_size()
    trg_vocab_size = tokenizer_trg.get_vocab_size()
    
    # Ensure the Transformer class __init__ method is compatible with these arguments
    model = Transformer(
        src_vocab_size=src_vocab_size,  # Vocabulary size of the source language tokenizer
        trg_vocab_size=trg_vocab_size,  # Vocabulary size of the target language tokenizer
        src_pad_idx=src_pad_idx,        # Pad token index for the source language tokenizer
        trg_pad_idx=trg_pad_idx,        # Pad token index for the target language tokenizer
        max_length=config['seq_len'],   # Maximum length of the source and target sequences
        device=device                   # Device to run the model on ('cuda' or 'cpu')
    ).to(device)  # Ensure the model is on the right device
    
    return model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_pad_idx = tokenizer_src.token_to_id("[PAD]")
    trg_pad_idx = tokenizer_trg.token_to_id("[PAD]")
    src_vocab_size = tokenizer_src.get_vocab_size()
    trg_vocab_size = tokenizer_trg.get_vocab_size()
    
    model = Transformer(
    src_vocab_size=src_vocab_size,  # Vocabulary size of the source language tokenizer
    trg_vocab_size=trg_vocab_size,  # Vocabulary size of the target language tokenizer
    src_pad_idx=src_pad_idx,        # Pad token index for the source language tokenizer
    trg_pad_idx=trg_pad_idx,        # Pad token index for the target language tokenizer
    device=device                   # Device to run the model on ('cuda' or 'cpu')
        ).to(device)  # Ensure the model is on the right device
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_trg = get_ds(config)
    print(f"Source tokenizer vocab size: {tokenizer_src.get_vocab_size()}")
    print(f"Target tokenizer vocab size: {tokenizer_trg.get_vocab_size()}")

    model= get_model(config,tokenizer_src,tokenizer_trg).to(device)
    # Tensorboard
    writer = SummaryWriter(config['expierment_name'])

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch']+1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_trg.token_to_id("[PAD]"),label_smoothing=0.1).to(device)    

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc=f"Epoch {epoch+1}/{config['num_epochs']}",unit="batch")
        for i,batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through the model
            # print(f"encoder_input: {encoder_input.shape}")
            # print(f"decoder_input: {decoder_input.shape}")
            # print(f"encoder_mask: {encoder_mask.shape}") 
            # print(f"decoder_mask: {decoder_mask.shape}")
            # print("Max token ID in encoder_input:", encoder_input.max().item())

            encoder_output = model.encoder(encoder_input,encoder_mask)
            decoder_output = model.decoder(decoder_input,encoder_output,encoder_mask,decoder_mask)

            label= batch['label'].to(device)
            loss = loss_fn(decoder_output.reshape(-1,decoder_output.shape[-1]),label.reshape(-1))
            batch_iterator.set_postfix(loss=loss.item())
            writer.add_scalar("Loss/train",loss,global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model weights
    model_filename = get_weights_file_path(config,str(epoch))
    print(f"Saving model weights to {model_filename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    },model_filename)

if __name__ == "__main__":
    config = get_config()
    train_model(config)