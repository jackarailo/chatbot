# Chatbot implementation in pytorch

Pass a query or type exit to terminate
> hello!
hello . 
> what's your name?
my . . . 
> do you have a name?
no . 
> how old are you?
twenty . 
> exit

## Instructions for chatting on Linux / OSX

1. Load a python3 environment with numpy & torch
1. git clone https://github.com/jackarailo/chatbot.git
1. chmod +x query_naive_bot.sh
1. ./query_naive_bot.sh
1. Type your query or exit to terminate

## Info

1. Seq-to-seq LSTM with bi-directional Encoder, Global Attention, and Greedy Selection Decoder.
1. Option to pass pretrained word vectors. Please download the glove vectors from https://nlp.stanford.edu/projects/glove/
1. Please download the glove vectors from https://nlp.stanford.edu/projects/glove/

## Acknowledgments

1. Minh-Thang Luong Hieu Pham Christopher D. Manning, 2015. Effective Approaches to Attention-based Neural Machine Translation 
1. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
1. Pytorch tutorial from Matthew Inkawhich
