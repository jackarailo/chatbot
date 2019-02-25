import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def get_model(vocab, **kwargs):
    checkpoint_file = kwargs.pop('checkpoint_file', None)
    use_existing_checkpoint = kwargs.pop('use_existing_checkpoint', False)
    ninp = kwargs.pop('ninp', 50)
    nhid = kwargs.pop('nhid', 200)
    nlayers = kwargs.pop('nlayers', 1)
    outwlen = kwargs.pop('outwlen', 10)
    wvectors = kwargs.pop('wvectors', None)
    ntoken = vocab.ntokens
    stoken = vocab['<START>']
    etoken = vocab['<END>']
    device = kwargs.pop('device', 'cpu')
    net = BotRNN(ntoken, ninp, nhid, nlayers, stoken, 
                  etoken, outwlen, device, vocab, wvectors)
    if use_existing_checkpoint:
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'])
    return net

class BotRNN(nn.Module):
    """Container class for the Bot model"""
    def __init__(self, ntoken, ninp, nhid, nlayers, stoken, etoken,
                 outwlen, device, vocab, wvectors=None):
        super(BotRNN, self).__init__()
        #self.attn = Attn()
        self.ntoken = ntoken # no of tokens in the vocabulary
        self.ninp = ninp # no of word vector dims
        self.nhid = nhid # no of hidden layer dims
        self.nlayers = nlayers # no of stacked LSTM layers
        self.stoken = stoken # start token
        self.etoken = etoken # end token
        self.outwlen = outwlen # output sentence len
        self.device = device
        self.vocab = vocab
        # Flag to generate words until <END> token for the decoder
        self.generate = True if outwlen == 0 else False

        # Submodules
        self.embed = nn.Embedding(self.ntoken, self.ninp)
        if wvectors is not None:
            self.embed.load_state_dict({'weight': wvectors})
            self.embed.weight.requires_grad = False
        self.encoderLSTM = nn.LSTM(self.ninp, self.nhid, self.nlayers,
                                   bidirectional=True)
        self.decoderLSTM = nn.LSTM(self.ninp, self.nhid, self.nlayers)
        self.attnOut = nn.Linear(self.nhid, self.nhid)
        self.attnConcat = nn.Linear(self.nhid*2, self.nhid)
        self.decoderOut = nn.Linear(self.nhid, self.ntoken)

        # Initialize weights
        #self._init_weights()

    def switch_mode(self, mode):
        if mode == 'train':
            self.generate = False
        elif mode == 'query':
            self.generate = True

    def _init_weights(self):
        # NOTE: LSTM encoder/decoder weights are
        # NOT initialized with this method!
        initrange = 0.1
        if self.wvectors is None:
            self.embed.weight.data.zero_()
        else:
            # Initialize embed from pretrained word vectors
            self.load_state_dict(self.wvectors)
        # initialize output linear layer
        self.attnOut.weight.data.uniform_(-initrange, initrange)
        self.attnOut.bias.data.zero_()
        self.decoderOut.weight.data.uniform_(-initrange, initrange)
        self.decoderOut.bias.data.zero_()

    def init_hidden(self, bsz):
        # NOTE: Method is not strictly required as the LSTM resets the hidden 
        # state to zero if not provided
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid))

    def forward(self, inp, linp):
        nwords, bsz = inp.shape
        ## ENCODER
        # Get embedings from inputs
        embEnc = self.embed(inp) # (nwords, bsz, ninp)
        # pack padded sequence #
        packed = nn.utils.rnn.pack_padded_sequence(embEnc, linp)
        # Initialize LSTM hidden state
        ehid = self.init_hidden(bsz)
        # run packed sequence through LSTM
        # ehid -> (nlayersX2, bsz, nhid)
        # eout -> (nwords, bsz, nhidX2)
        packed, ehid = self.encoderLSTM(packed, ehid)
        # unpack sequence
        eout, _ = nn.utils.rnn.pad_packed_sequence(packed)
        # sum bidirectional LSTM outputs
        # dhid -> (nlayers, bsz, nhid)
        dhid = []
        for h in ehid:
            dhid.append(h[:self.nlayers, :, :] + h[self.nlayers:, :, :])
        dhid = tuple(dhid)
        # eout (nwords, bsz, nhid) 
        eout = eout[:, :, :self.nhid] + eout[:, :, self.nhid:]
        ## DECODER WITH ATTENTION and Greedy selection
        wout = self.outwlen if self.generate else 0
        token = torch.tensor([self.stoken]*bsz, dtype=torch.long).reshape(1,-1).to(self.device)
        out = []
        while (self.generate or self.outwlen>wout):
            embDec = self.embed(token) # (1, bsz)
            # dout (1,bsz, nhid) 
            # dhid (nlayers, bsz, nhid)
            dout, dhid = self.decoderLSTM(embDec, dhid)

            dout = dout.squeeze(0)
            output = self.decoderOut(dout)
            output = F.softmax(output, dim=1)
            output = torch.log(output)
            token = self.greedy_select(output)
            out.append(output)
            if not self.generate:
                wout += 1
            elif token==self.etoken:
                self.generate = False
        return out

    def greedy_select(self, out):
        x =  torch.argmax(out, dim=1).reshape(1,-1).to(self.device)
        return x

