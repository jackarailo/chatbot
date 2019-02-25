import os
import argparse

import torch

import model
import data
import train
import query

parser = argparse.ArgumentParser(
            description=r"""Chatbot influenced by the tutorial 
            on https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
            and pytorch official examples on gihub""")
parser.add_argument('--mode', type=str, default='train',
                    help='mode to run main (train, query)')
parser.add_argument('--device', type=str, default='best',
                    help='device to use (cpu, cuda, best)')
parser.add_argument('--datadir', type=str, 
                    default='./data/cornell_movie-dialogs_corpus/',
                    help='directory of the data corpus')
parser.add_argument('--epochs', type=int, default=100,
                    help=r"Number of epochs to train the model")
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help=r"Learning rate")
parser.add_argument('--batch_size', type=int, default=64,
                    help=r"Batch size during training")
parser.add_argument('--max_word_length', type=int, default=20,
                    help=r"Max word length (required if batch size > 1)")
parser.add_argument('--min_count', type=int, default=1000,
                    help=r"Min number of times word is repeated to be added")
parser.add_argument('--print_every', type=int, default=10,
                    help=r"Generate words every print_every steps during train")
parser.add_argument('--pretrained_word_vectors', type=str, default="",
                    help="optional directory to pretrained word vectors")
parser.add_argument('--checkpoint', type=str, default='./model_state_dict.pt',
                    help='directory to state_dict checkpoint to use')
parser.add_argument('--query', type=str, default=None,
                    help='pass your query to the bot')
parser.add_argument('--use_existing_checkpoint', type=int, default=1,
                    help='use existing checkpoint if it exists')

args = parser.parse_args()
# Globals
DATA_DIR = args.datadir
if args.pretrained_word_vectors != "":
    WORD_VECTORS_FILE = args.pretrained_word_vectors
else:
    WORD_VECTORS_FILE = None
VALID_MODES = ['train', 'query']
MODE = args.mode
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PRINT_EVERY = args.print_every
LEARNING_RATE = args.learning_rate
MAX_WORD_LEN = args.max_word_length
MIN_COUNT = args.min_count
if args.device == 'best':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = args.device
CHECKPOINT_FILE = args.checkpoint
USE_EXISTING_CHECKPOINT = False if args.use_existing_checkpoint == 0 else True
QUERY = args.query
RAW_DATA = False

def main():
    # Get the word tokens
    wvectors = None
    if MODE == 'train':
        sinputs, stargets = data.get_sentences(DATA_DIR, MAX_WORD_LEN,
                                               MIN_COUNT)
    if WORD_VECTORS_FILE is not None:
        tokens, wvectors = data.get_tokens_from_vectors_file(WORD_VECTORS_FILE)
        sinputs, stargets = data.clear_not_found(tokens, sinputs, stargets)
        tokens = tokens[4:]
    elif os.path.isfile(CHECKPOINT_FILE) and USE_EXISTING_CHECKPOINT:
        tokens = data.get_tokens_from_checkpoint_file(CHECKPOINT_FILE)
    else:
        # get list of lists for each word, sentence in inputs and targets
        tokens = data.get_tokens_from_data(sinputs, stargets)
        SDATA = True
    # Populate the vocab based on the word tokens
    vocab = data.Vocab(tokens)
    # Create the model and move it to device
    net = model.get_model(vocab, wvectors=wvectors, 
                          checkpoint_file=CHECKPOINT_FILE, 
                          use_existing_checkpoint=USE_EXISTING_CHECKPOINT,
                          device=DEVICE, outwlen=MAX_WORD_LEN)
    if MODE == 'train':
        inputs, mask_inputs, targets, mask_targets = data.get_tensors(
                                            vocab, [sinputs, stargets], 
                                            MAX_WORD_LEN)
        print("-----------Starting training--------------")
        train.step(net, inputs, mask_inputs, targets, mask_targets, vocab, DEVICE,
                  EPOCHS, BATCH_SIZE, CHECKPOINT_FILE, LEARNING_RATE,
                  PRINT_EVERY)
    elif MODE == 'query':
        print("Pass a query or type exit to terminate")
        q = input("> ")
        while q != 'exit':
            query.respond(q, net, vocab, DEVICE)
            q = input("> ")

def test():
    sinputs, stargets = data.get_sentences(DATA_DIR, MAX_WORD_LEN, MIN_COUNT)
    tokens = data.get_tokens_from_data(sinputs, stargets)
    vocab = data.Vocab(tokens)
    inputs, mask_inputs, targets, mask_targets = data.get_tensors(
                                        vocab, [sinputs, stargets], 
                                        MAX_WORD_LEN)
    return sinputs, stargets, vocab

if __name__ == "__main__":
    main()


