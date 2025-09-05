import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from optimizer import SGD
from dataset import ptb
import numpy as np
from rnnlm import SinpleRnnlm, LSTMRnnlm
from layers import *

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
learning_rate = 0.1
max_epoch = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('corpus size : %d, vocabulary size: %d' % (corpus_size, vocab_size))

max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

model = LSTMRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(learning_rate)

jump = (corpus_size - 1)
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):

        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        for t in range(time_size):
            for i , offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch+1, ppl))

    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.title('Perplexity over Epochs')
plt.title('Perplexity over Epochs')
plt.savefig('simple_rnn_train_LSTM.png')
print("グラフを 'perplexity_graph.png' として保存しました。")