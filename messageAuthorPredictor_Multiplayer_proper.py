#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Some notes on how to set up GPU for tensorflow in Jupyter notebook
#%env PATH="/home/aes/Datamining/.venv/bin:/usr/local/lib/nodejs/node-v20.11.0-linux-x64/bin:/home/aes/.local/bin:/usr/local/cuda-12.4/bin:/usr/java/jdk-21.0.2/bin:/opt/gradle/gradle-8.5/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/c/Program Files/NVIDIA/CUDNN/v9.1/bin:/mnt/c/Python311/Scripts/:/mnt/c/Python311/:/mnt/c/Windows/system32:/mnt/c/Windows:/mnt/c/Windows/System32/Wbem:/mnt/c/Windows/System32/WindowsPowerShell/v1.0/:/mnt/c/Windows/System32/OpenSSH/:/mnt/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/mnt/c/Program Files/Calibre2/:/mnt/c/Program Files/dotnet/:/mnt/c/Program Files/Git/cmd:/mnt/c/Program Files/nodejs/:/mnt/c/ProgramData/chocolatey/bin:/mnt/c/Program Files/MySQL/MySQL Server 8.0/bin:/mnt/c/Program Files/MySQL/MySQL Router 8.0/bin:/mnt/c/Program Files/MySQL/MySQL Shell 8.0/bin:/mnt/c/Program Files/yt-dlp:/mnt/c/Program Files (x86)/GnuPG/bin:/mnt/c/Program Files/Mullvad VPN/resources:/mnt/c/Program Files (x86)/Pulse Secure/VC142.CRT/X64/:/mnt/c/Program Files (x86)/Pulse Secure/VC142.CRT/X86/:/mnt/c/Program Files/Docker/Docker/resources/bin:/mnt/c/Strawberry/c/bin:/mnt/c/Strawberry/perl/site/bin:/mnt/c/Strawberry/perl/bin:/mnt/c/Program Files/NVIDIA Corporation/NVIDIA NvDLISR:/mnt/c/Users/aesun/AppData/Local/Programs/Python/Python312/Scripts/:/mnt/c/Users/aesun/AppData/Local/Programs/Python/Python312/:/mnt/c/Users/aesun/AppData/Local/Microsoft/WindowsApps:/mnt/c/Users/aesun/AppData/Local/Programs/Microsoft VS Code/bin:/mnt/c/Users/aesun/AppData/Roaming/npm:/mnt/c/Users/aesun/AppData/Local/Programs/MiKTeX/miktex/bin/x64/:/mnt/c/Program Files/Neovim/bin:/snap/bin:/usr/local/cuda-12.4/bin"
#%env LD_LIBRARY_PATH="/home/aes/Datamining/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64:/usr/local/cuda-12.4/lib64"
#%env CUDNN_PATH="/home/aes/Datamining/.venv/lib/python3.10/site-packages/nvidia/cudnn"

# You should set up cudnn path in your system to get GPU to work
# export CUDNN_PATH="/home/aes/Datamining/.venv/lib/python3.10/site-packages/nvidia/cudnn"
# and
# export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda-12.4/lib64"

# In[4]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import urllib.request
from imblearn.over_sampling import SMOTE


# In[5]:


# Load the dataset
data = pd.read_csv('cleaned_data/uber_boat_cleaned.csv')
data2 = pd.read_csv('cleaned_data/ch_cleaned.csv')

# Concatenate the two datasets
data = pd.concat([data, data2])
author = data['author']
message = data['message']
header = data.columns

# Count the number of unique authors
author_id = author.unique()
print('Number of unique authors:', len(author_id))


# In[6]:


# Set tensorflow debugging on or off
tf.debugging.set_log_device_placement(False)

# Check if tensorflow is using the GPU
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# If you see num GPUs available as 0 then you should be worried.


# In[12]:


# Tokenize the message
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(message)
message_sequence = tokenizer.texts_to_sequences(message)

# Pad the tokenized message sequence
max_len = max([len(seq) for seq in message_sequence])
print('Max length of message sequence:', max_len)
message_sequence = tf.keras.utils.pad_sequences(message_sequence, maxlen=max_len)

author_label = tf.keras.utils.to_categorical(author)
print(author_label[0])

# Count how many messages per author
# First create a dictionary of authors and int
author_id = author.unique() 
author_to_messages_count = {auth_id: 0 for auth_id in author_id}

# Count the number of messages per author
for a in author_label:
    author_to_messages_count[np.argmax(a)] += 1

print("Sample size:", len(author_label))
print('Number of messages per author before oversampling:', author_to_messages_count)

# Oversample the dataset
smote = SMOTE()
message_os, author_os = smote.fit_resample(message_sequence, author_label)

# Count the number of messages per author after oversampling
author_to_messages_count = {auth_id: 0 for auth_id in author_id}
for a in author_os:
    author_to_messages_count[np.argmax(a)] += 1

print('Number of messages per author after oversampling:', author_to_messages_count)


# In[8]:


# Split the dataset into training, validation and testing sets
message_train, message_test, author_train, author_test = train_test_split(message_os, author_os, test_size=0.3, random_state=1, stratify=author_os)
message_test, message_val, author_test, author_val = train_test_split(message_test, author_test, test_size=0.5, random_state=1, stratify=author_test)


# In[9]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 128, name='embedding'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128), name='lstm'),
    tf.keras.layers.Dense(128, activation='relu', name='dense'),
    tf.keras.layers.Dropout(0.4, name='dropout'),
    tf.keras.layers.Dense(64, activation='relu', name='dense2'),
    tf.keras.layers.Dropout(0.4, name='dropout2'),
    tf.keras.layers.Dense(5, activation='softmax', name='output')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[10]:


print("Model stats before training")
test_pre_loss, test_pre_accuracy = model.evaluate(message_test, author_test)
print('Test loss:', test_pre_loss)
print(f"Test accuracy: {test_pre_accuracy*100:.2f}%")


# In[11]:


epochs = 4
model.fit(message_train, author_train, epochs=epochs, validation_data=(message_val, author_val))


# In[ ]:


# Test the model using the test data
test_loss, test_accuracy = model.evaluate(message_test, author_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Test recall
recall_loss, recall_accuracy = model.evaluate(message_train, author_train)
print('Recall loss:', recall_loss)
print('Recall accuracy:', recall_accuracy)


# In[ ]:


# Load the stop words
STOP_WORDS = urllib.request.urlopen("https://github.com/igorbrigadir/stopwords/blob/master/en/postgresql.txt").read().decode("utf-8").split("\n")


# In[ ]:


def clean_message(message):
    message = message.replace(".", "")
    message = message.replace(",", "")
    message = message.replace(";", "")
    message = message.replace("!", "")
    message = message.replace("?", "")
    message = message.replace("(", "")
    message = message.replace(")", "")
    message = message.replace("\\", "")
    message = message.replace("\"", "")

    # remove discord effects
    message = message.replace("*", "")
    message = message.replace("_", "")
    message = message.replace("~", "")
    message = message.replace("`", "")
    message = message.replace(">", "")
    message = message.replace("<", "")
    message = message.replace("||", "")
    message = message.replace("```", "")
    message = message.replace("~~", "")
    message = message.replace(":", "")
    message = message.replace("#", "")
    message = message.replace("@", "")


    # remove stopwords
    message = message.lower()
    message = ' '.join([word for word in message.split() if word not in STOP_WORDS or word in USER_NAMES])

    return message


# In[ ]:


participating_users = { "Shrimpy Raccoon" : 0,
                        "Stella" : 1,
                        "jeremy" : 2,
                        "Nosmo" : 3,
                        "matt_m_h" : 4
}
users = list(participating_users.keys())

def predict_string_author(test_message_raw):
    # remove stopwords
    test_message = clean_message(test_message_raw)
    test_message_sequence = tokenizer.texts_to_sequences([test_message])
    test_message_sequence = tf.keras.utils.pad_sequences(test_message_sequence, maxlen=max_len)
    prediction = model.predict(test_message_sequence)
    preds = []
    for i, name in enumerate(users):
        preds.append([name, prediction[0][i] * 100])
    
    return users[np.argmax(prediction)], prediction[0][np.argmax(prediction)] * 100


# 

# In[ ]:


test_message_raw = [
    ["Nyaa UwU I'm a femboy", None],
    ["i'm going to become physically violent", None],
    ["kill sedimentary fuck metamorphic marry igneous", None],
    ["kill me already", None],
    ["face planted into the corner of a table", None]
    ]

for test_message, actual_author in test_message_raw:
    username, prediction = predict_string_author(test_message)
    print(f"Message: {test_message}\nPredicted Author: {username} ({prediction:.2f}%)\nActual Author: {actual_author}\n{'!!! CORRECT !!!' if actual_author == username else ''}")


# In[ ]:


model_name = "uber_boat_and_CH_OS_128_128_64_4_epoch"

# Check if the model already exists if it already does add a number after it
if os.path.exists(model_name + '_model.keras'):
    model_name = model_name + '_1'
    while os.path.exists(model_name + '_model.keras'):
        model_name = model_name[:-1] + str(int(model_name[-1]) + 1)

# Save the model
model.save(model_name + '_model.h5', save_format = 'h5')

# tf.keras.saving.save_weights(model, 'chat_weights.keras')
model.save_weights(model_name + '.weights.h5')

# model.save_weights('chat_weights.h5')
tokenizer_json = tokenizer.to_json()
with open(model_name + '_tokenizer.json', 'w') as json_file:
    json_file.write(tokenizer_json)

# Output the model summary
with open(model_name + '_model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f"Epochs: {epochs}\n\n")
    f.write(f"Testing Summary: \n")
    f.write(f"Test loss: {test_loss}\n")
    f.write(f"Test accuracy: {test_accuracy}\n")
    f.write(f"Recall loss: {recall_loss}\n")
    f.write(f"Recall accuracy: {recall_accuracy}\n")
    f.write(f"\n")
    f.write(f"Model files saved as: \n")
    f.write(f"Model saved as {model_name}_model.keras\n")
    f.write(f"Weights saved as {model_name}_weights.weights.h5\n")
    f.write(f"Tokenizer saved as {model_name}_tokenizer.json\n")

print(f"Model saved as {model_name}_model.keras")
print(f"Weights saved as {model_name}_weights.weights.h5")
print(f"Tokenizer saved as {model_name}_tokenizer.json")
print(f"Model summary saved as {model_name}_model_summary.txt")
