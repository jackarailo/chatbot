import numpy as np
import torch
from torch import nn

def step(net, inputs, mask_inputs, targets, mask_targets, vocab, device,
         epochs, bsz, checkpoint_file, lr, print_every=10):
    net = net.to(device)
    net.train()
    optim = torch.optim.Adam(net.parameters(), lr = lr)
    k = 0
    for e in range(epochs):
        batch_data = generate_batches(inputs, mask_inputs, targets, 
                                      mask_targets, bsz, device)
        for bx, bxm, by, bym in batch_data:
            optim.zero_grad()
            out = net(bx, bxm)
            mask = torch.zeros_like(by, dtype=torch.uint8).to(device)
            for i in range(mask.shape[1]):
                mask[:bym[i], i] = 1
            loss = 0
            for yhat, y, m in zip(out, by, mask):
                loss += CELoss(yhat, y, m)
            loss /= len(out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
            optim.step()
            k += 1
            if k % print_every == 0:
                net.eval()
                qs = bx.clone().detach()
                pred = net(qs, bxm.clone().detach())
                lp = len(pred)
                tp = torch.zeros([lp, by.shape[1], net.ntoken])
                for i, tk in enumerate(pred):
                    tp[i, :, :] = tk
                _, tokens = tp.topk(k=1, dim=2)
                tokens = tokens.squeeze(2).permute(1,0)
                generate_qa_pairs(vocab, qs.permute(1,0), tokens, by.permute(1,0))
                net.train()
                print(f"Training Loss {loss.item()}")
    if checkpoint_file is not None:
            print(f"Checkpoint saved on {checkpoint_file}")
            torch.save({'model_state_dict': net.state_dict(),
                        'tokens': vocab.get_tokens()},
                       checkpoint_file)

def CELoss(yhat, y, mask):
    l = y.shape[0]
    pred = yhat[torch.arange(l), y]
    pred = pred.masked_select(mask.view(-1))
    loss = -torch.sum(pred) / len(pred) if len(pred) > 0 else 0
    return loss

def generate_batches(inputs, mask_inputs, targets, mask_targets, bsz, device):
    wlen, ninputs = inputs.shape
    inputs = inputs.to(device)
    mask_inputs = mask_inputs.to(device)
    targets = targets.to(device)
    for b in range(0, ninputs, bsz):
        bx = inputs[:, b:b+bsz]
        bxm = mask_inputs[b:b+bsz]
        by = targets[:, b:b+bsz]
        bym = mask_targets[b:b+bsz]
        yield bx, bxm, by, bym

def generate_qa_pairs(vocab, questions, answers, canswers):
    nq, lenq = questions.shape
    _, lena = answers.shape
    _, lenca = canswers.shape
    for i in range(nq):
        q = ""
        for j in range(lenq):
            qq = int(questions[i,j].detach().cpu().numpy())
            q += vocab[qq]
            q += " "
        a = ""
        for j in range(lena):
            aa = int(answers[i,j].cpu().numpy())
            a += vocab[aa]
            a += " "
        qa = ""
        for j in range(lenca):
            ca = int(canswers[i,j].cpu().numpy())
            qa += vocab[ca]
            qa += " "
        print(f"Question: {q} --- Answer: {a} ----- Correct Answer: {qa}")

