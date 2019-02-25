import re

import torch

import data

def respond(q, net, vocab, device):
    if q == '':
        print("Sorry, you didn't pass anything") 
        return None
    qe = re.findall(r"[\w']+|[.,!?;]", q)
    inputs = [vocab[q.lower()] for q in qe]
    for i, t in enumerate(inputs):
        if t is None:
            print(f"Sorry didn't understand word {qe[i]}") 
            return None
    net = net.to(device)
    net.switch_mode('query')
    net.eval()
    with torch.no_grad():
        bx = torch.tensor(inputs).reshape(-1,1).to(device)
        bxm = torch.tensor([len(inputs)], dtype=torch.long).to(device)
        out = net(bx, bxm)
        lout = len(out)
        tp = torch.zeros([lout, net.ntoken])
        for i, tk in enumerate(out):
            tp[i, :] = tk
        _, tokens = tp.topk(k=1, dim=1)
        tokens = tokens[:-1]
        response = ""
        for word in tokens:
            response += vocab[word.item()]
            response += " "
        print(response)
        return None

def test():
    checkpoint = torch.load('test.pth')
    tokens = checkpoint['tokens']
    vocab = data.Vocab(tokens)
    q = "HHhello, world"
    qe = respond(q, None, vocab)
    return qe
