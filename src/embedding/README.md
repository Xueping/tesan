# Medical Concept Embedding
## 1.Models
### 1.1.TeSAN
Our proposed model
### 1.2.Normal Self-attention
normal
### 1.3. Multi-Dimen Self-attention
sa
### 1.4.Delta
Time intervals
## 2.Baseline Methods
### 2.5.GloVe
### 2.6.Skip-gram
### 2.7.CBOW
### 2.8.CME
### 2.9.Med2vec

## 3. Dataset
First, we concatenate all medical concepts in patient's visits, and each concept has a time stamp 
from admission time.

### 3.1 Context
Given a skip window k and a target medical concept, the context is k medical concepts before and 
k concepts after the target concept.
### 3.2 Time Intervals
The context is given with a time stamp for each concept, the interval is a matrix with shape of 2k*2k,
 and each element is the difference between the time stamps of two context concepts.
 
### 3.3 Labals
The label is the target concept. 
### 3.4 Codes
src.embedding.concept.dataset.py

## 4.Methodology
### 4.1 Embed context
### 4.2 Embed time intervals
### 4.3 TeSA
### 4.4 Codes
src.embedding.concept.model.py

## 5.Train
### 5.1 Generate Batch data
### 5.2 Epoch and Steps based train
### 5.3 Codes
src.embedding.concept.train_tesan_sesa.py

## 6.Evaluation
### 6.1 Clustering and Nearest neighbour search
### 6.2 Mortality prediction
We use Gated Recurrent Units (GRU) with different embedding strategies to map visit
embedding sequence v(1),... , v(T) to a patient representation h.



