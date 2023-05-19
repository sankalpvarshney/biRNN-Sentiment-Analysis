# Sentiment Analysis of IMDB comments using Bi-directioanl RNN

A bidirectional recurrent neural network (RNN) is a type of neural network architecture that processes sequential data in both forward and backward directions. Unlike traditional RNNs that only consider the past context of the sequence, bidirectional RNNs also incorporate future context by processing the sequence in reverse.

In a bidirectional RNN, the input sequence is fed into two separate RNNs: one RNN processes the sequence in the forward direction, starting from the beginning, while the other RNN processes the sequence in the reverse direction, starting from the end. The outputs of both RNNs are then combined or used independently to make predictions or extract features from the sequence.

By considering both past and future context, bidirectional RNNs can capture dependencies and patterns that may be missed by unidirectional RNNs. They are commonly used in tasks where the entire sequence is available from the beginning, such as natural language processing tasks like sentiment analysis, named entity recognition, and machine translation.

