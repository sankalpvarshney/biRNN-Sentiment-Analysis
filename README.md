# Sentiment Analysis of IMDB comments using Bi-directioanl RNN

A bidirectional recurrent neural network (RNN) is a type of neural network architecture that processes sequential data in both forward and backward directions. Unlike traditional RNNs that only consider the past context of the sequence, bidirectional RNNs also incorporate future context by processing the sequence in reverse.

In a bidirectional RNN, the input sequence is fed into two separate RNNs: one RNN processes the sequence in the forward direction, starting from the beginning, while the other RNN processes the sequence in the reverse direction, starting from the end. The outputs of both RNNs are then combined or used independently to make predictions or extract features from the sequence.

By considering both past and future context, bidirectional RNNs can capture dependencies and patterns that may be missed by unidirectional RNNs. They are commonly used in tasks where the entire sequence is available from the beginning, such as natural language processing tasks like sentiment analysis, named entity recognition, and machine translation.

![Screenshot from 2023-05-19 09-13-49](https://github.com/sankalpvarshney/biRNN-Sentiment-Analysis/assets/41926323/93a50eaf-cf0c-4db9-ae16-958df4211577)


## Installations

Create python virtual environment for avoiding the version conflictions.

```bash
conda create --prefix ./env python=3.8 -y
conda activate ./env
```

Install requirements.txt file to make sure correct versions of libraries are being used.

```bash
pip install -r requirements.txt
```

## Usage

### Training

For changing the dataset and tuning the hyper-parmeters of algorithm modify the config/config.yaml file. After the changes execute the below command.

```bash
python src/components/train.py
```

### Inferencing

To facilitate inference and utilize this feature via an API, we have developed an application that assists in deploying the API on a server. Additionally, it provides scalability by allowing control over the number of workers using uvicorn. This application proves highly beneficial in ensuring efficient deployment and scalability of the API.

For te deployment of this service execute the below command

```bash
uvicorn main:app --host '0.0.0.0' --port 3354
```
 You can use any available port on the sysytem


 ## Output

 ![Screenshot from 2023-05-19 10-09-16](https://github.com/sankalpvarshney/biRNN-Sentiment-Analysis/assets/41926323/bd18adc1-1cee-4808-b49b-058f697c9dda)
