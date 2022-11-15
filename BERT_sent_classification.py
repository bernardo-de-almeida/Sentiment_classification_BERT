
## run as: 
# my_bsub_gridengine -P g -G "gpu:1" -m 10 -T '07:00:00' -o log_training -n Training "python ./BERT_sent_classification.py"

from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

# Decide on length 50 just that we have as much data as possible
SEQ_LEN = 50

### Initialize tokenizer
print("\nInitialize tokenizer ...\n")

# BERT model + BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased') # case means that bert distinguiches difference between upper case and lower case

### Load data
print("\nLoad data ...\n")

with open('xids.npy', 'rb') as fp:
    Xids = np.load(fp)
with open('xmask.npy', 'rb') as fp:
    Xmask = np.load(fp)
with open('labels.npy', 'rb') as fp:
    labels = np.load(fp)
    
dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels)) # creates a generetaor that contains all data in the tupple-like format

# Create data structure
def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

dataset = dataset.map(map_func)

# shuffle and put samples into batches of 32
dataset = dataset.shuffle(100000).batch(32)
# split into training and validation set
DS_LEN = len(list(dataset))

SPLIT = 0.9 # train/val split
train = dataset.take(round(DS_LEN*SPLIT)) # select these
val = dataset.skip(round(DS_LEN*SPLIT)) # skip these

del dataset

### Build model architecture
print("\nBuild model architecture ...\n")

# initialize BERT
from transformers import TFAutoModel
# had to correct keras loading (https://github.com/huggingface/transformers/issues/18912)
bert = TFAutoModel.from_pretrained('bert-base-cased') # same model used to initialie tokenizer

# Define input layers: 2 --> input IDs and input attention mask
input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32') # name is very important - needs to match the dictionary names
mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

# pull the embeddings from the initialized BERT model
# bert will return 2 tensors:  1) last hidden-state, which is what we are interested in, 3D tensor with all information from last hidden-state. 2) puller output: after runing it through a FF and linear activation and pooled, 2D densor that can be used to classification (ignre)
embeddings = bert(input_ids, attention_mask=mask)[0]

# now you can experiment with adding any lstm layers, CNNs, or anything else
# we will keep it simple and add a gloabl maxpooling layer - converting 3D to 2D tensor
X = tf.keras.layers.GlobalMaxPool1D()(embeddings)
# normalize outputs
X = tf.keras.layers.BatchNormalization()(X)

# Fully Connected layers in charge of the classification
X = tf.keras.layers.Dense(128, activation='relu')(X)
X = tf.keras.layers.Dropout(0.1)(X)
X = tf.keras.layers.Dense(32, activation='relu')(X)
y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(X)

model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

# freeze the bert model to don't train it
model.layers[2].trainable = False

model.summary()

### Model training
optimizer = tf.keras.optimizers.Adam(0.01)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

EPOCHS = 250
checkpoint_filepath = 'my_best_model.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# fit the model
print("\nModel training ...\n")
history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)

# Save model
model.save('Model_BERT_final_class.h5')
# Save history
np.save('Model_BERT_final_class_history.npy',history.history) 
# history=np.load('Model_BERT_final_class_history.npy',allow_pickle='TRUE').item()


### Model evaluation
print("\nModel evaluation ...\n")
import matplotlib.pyplot as plt
# plot history
plt.ion()
fig = plt.figure(figsize=(14,4))
subfig = fig.add_subplot(122)
subfig.plot(history.history['accuracy'], label="training")
if history.history['val_accuracy'] is not None:
    subfig.plot(history.history['val_accuracy'], label="validation")
subfig.set_title('Model Accuracy')
subfig.set_xlabel('Epoch')
subfig.legend(loc='upper left')
subfig = fig.add_subplot(121)
subfig.plot(history.history['loss'], label="training")
if history.history['val_loss'] is not None:
    subfig.plot(history.history['val_loss'], label="validation")
subfig.set_title('Model Loss')
subfig.set_xlabel('Epoch')
subfig.legend(loc='upper left')
plt.ioff()
plt.savefig('Model_BERT_final_class_performance.pdf')  

### Model testing
print("\nModel testing ...\n")

def create_test(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_attention_mask=True, return_tensors='tf')
    return tokens

def sentiment_pred(seq):
    pred = model.predict([create_test(seq)['input_ids'], create_test(seq)['attention_mask']])[0]
    # convert class
    print(seq + ' --> sentiment: ' + str(np.where(pred == np.amax(pred))[0][0]) + ' stars')

# test
sentences = ['This is shit',
             'This is bad',
             'This is poor',
             'This is not too bad',
             'This is okay',
             'This is good',
             'This is interesting',
             'This is fine',
             'This is excelent',
             'This is great',
             'This is amazing',
             'This is the best ever',
             'I love it!!!',
             
             'I hated this movie',
             'It had a very interesting story',
             'It had a very interesting story but I did not like the actors',
             'I did not like the actors',
            
             'I love you']

for s in sentences:
    sentiment_pred(s)