import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)


class Plot2Vec(nn.Module):

    def __init__(self, vocab_size, embedding_dim, genres_size):
        super(Plot2Vec, self).__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, genres_size)

    def forward(self, inputs):
        out = F.relu(self.linear1(inputs))
        out =  F.relu(self.linear2(out))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def transform(self, X):
        return Variable(F.relu(self.linear1(torch.tensor(X.toarray(), dtype=torch.float)))).numpy()


def fit(X, y, epochs=5, n_features=100):
    losses = []
    loss_function = nn.NLLLoss()
    model = Plot2Vec(X.shape[1], n_features, y.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    #X = X[:1000, :]
    print(X.shape)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(X.shape[0]):
            genres = np.where(y[i] == 1)[0]
            for j in genres:
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                movie_idxs = torch.tensor(X[i].toarray(), dtype=torch.float)
                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                model.zero_grad()
                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(movie_idxs)
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # wrapped in a tensor)
                loss = loss_function(log_probs, torch.tensor(y[i, j].reshape(1), dtype=torch.long))
                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()
                #print(loss.item()*1/genres.shape[0])
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()*1/genres.shape[0]
        total_loss /= X.shape[0]
        losses.append(total_loss)
        print('loss in epoch', epoch, ':', total_loss)
    return model, losses

def transform(X, model):
    return model.transform(X)


def plot2vec(X, **kwargs):
    epochs = kwargs.get('epochs', 5)
    n_features = kwargs.get('n_features', 100)
    movies = kwargs['movies']
    genres = movies['genres'].str.get_dummies()
    y = genres.values
    model, losses = fit(X, y)
    return transform(X, model)

#def movie2vec(df_movies, *):
