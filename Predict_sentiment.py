
# example: python ./Predict_sentiment.py "I love it" "I hate it" "You are great" "It had a very interesting story but I did not like the actors" "I feel ashamed"

from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel
bert = TFAutoModel.from_pretrained('bert-base-cased') # same model used to initialie tokenizer

# get input
import sys
sentences = sys.argv[1:]

# Load model
model_path = '/groups/stark/almeida/DeepLearning_practice/Sentiment_classification_BERT/Model_BERT_final_class.h5'
model = tf.keras.models.load_model(model_path, custom_objects={"TFBertModel": bert})

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
SEQ_LEN=50
def create_test(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_attention_mask=True, return_tensors='tf')
    return tokens

def sentiment_pred(seq):
    pred = model.predict([create_test(seq)['input_ids'], create_test(seq)['attention_mask']])[0]
    # convert class
    print(seq + ' --> ' + str(np.where(pred == np.amax(pred))[0][0]) + ' stars')

# Make prediction
print("\nPredictions ...\n")
for s in sentences:
    sentiment_pred(s)