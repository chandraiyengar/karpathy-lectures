import torch
import torch.nn.functional as F

# note: we are now taking 2 character strings as input, and outputting 1 character.
# in the bigram model the input space was the same as the ouput space, but that is not true now.
# So we need to:
# 1. find the input vectors, i.e. all 2 character strings in the training data + make a map
# 2. create the dataset. It will contain the same labels as before, but x will now be all 2 character strings.
# 3. do gradient descent
# 4 sample using the new approach

words = open("names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

bigrams = set()
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigrams.add((ch1 + ch2))

bigramstoi = {b: i + 1 for i, b in enumerate(bigrams)}
itobigrams = {i: b for b, i in bigramstoi.items()}

# for w in words:
#     chs = ["."] + list(w) + ["."]
#     for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
#         bigrams.add((ch1 + ch2, ch3))
# # create the dataset
# xs, ys = [], []
# for w in words:
#     chs = ["."] + list(w) + ["."]
#     for ch1, ch2 in zip(zip(chs, chs[1:]), chs[2:]):
#         ch1 = ch1[0] + ch1[1]  # now ch1 is two consecutive chars.
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         xs.append(ix1)
#         ys.append(ix2)
# xs = torch.tensor(xs)
# ys = torch.tensor(ys)
# num = xs.nelement()
# print("number of examples: ", num)

# # initialize the 'network'
# g = torch.Generator().manual_seed(2147483647)
# W = torch.randn((27, 27), generator=g, requires_grad=True)

# # gradient descent
# for k in range(150):

#     # forward pass
#     xenc = F.one_hot(
#         xs, num_classes=27
#     ).float()  # input to the network: one-hot encoding
#     logits = xenc @ W  # predict log-counts
#     counts = logits.exp()  # counts, equivalent to N
#     probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
#     loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
#     print(loss.item())

#     # backward pass
#     W.grad = None  # set to zero the gradient
#     loss.backward()

#     # update
#     W.data += -50 * W.grad


# # finally, sample from the 'neural net' model
# g = torch.Generator().manual_seed(2147483647)

# for i in range(5):

#     out = []
#     ix = 0
#     while True:

#         # ----------
#         # BEFORE:
#         # p = P[ix]
#         # ----------
#         # NOW:
#         xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
#         logits = xenc @ W  # predict log-counts
#         counts = logits.exp()  # counts, equivalent to N
#         p = counts / counts.sum(1, keepdims=True)  # probabilities for next character
#         # ----------

#         ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         out.append(itos[ix])
#         if ix == 0:
#             break
#     print("".join(out))
