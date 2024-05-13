# %% [code] {"id":"vVNAD6VNMMz8"}


# %% [markdown] {"id":"EJDa3TCdMQez"}
# # Preprocessing

# %% [markdown] {"id":"aU0yYT1rMOlC"}
# 

# %% [code] {"id":"Y8Y48vxwepJF","execution":{"iopub.status.busy":"2024-05-01T22:41:19.996104Z","iopub.execute_input":"2024-05-01T22:41:19.996496Z","iopub.status.idle":"2024-05-01T22:41:21.081411Z","shell.execute_reply.started":"2024-05-01T22:41:19.996465Z","shell.execute_reply":"2024-05-01T22:41:21.078206Z"}}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Assuming you have loaded your dataset into a DataFrame called 'df'
#df= pd.read_csv('volvo_daily_max.csv')
#df.isna().sum()
#df=df.dropna()
#df.isna().sum()


# %% [code] {"id":"ct6BDCUoMM-k","execution":{"iopub.status.busy":"2024-05-01T22:41:21.082291Z","iopub.status.idle":"2024-05-01T22:41:21.082783Z","shell.execute_reply.started":"2024-05-01T22:41:21.082585Z","shell.execute_reply":"2024-05-01T22:41:21.082603Z"}}
df = pd.read_csv('https://raw.githubusercontent.com/umar-farooq-khan/m-en-dataset/main/M-En%20Dataset.csv')


# %% [markdown]
# 

# %% [code] {"id":"ov5-VlrWuMfK","outputId":"bad3dd51-05bb-4840-f442-624c0a4173e4","execution":{"iopub.status.busy":"2024-05-01T22:41:21.083839Z","iopub.status.idle":"2024-05-01T22:41:21.084289Z","shell.execute_reply.started":"2024-05-01T22:41:21.084086Z","shell.execute_reply":"2024-05-01T22:41:21.084109Z"}}
#df=bmw_data
df

# %% [markdown]
# 

# %% [code] {"id":"HxcHH1wwMyA0","outputId":"e5bb2c55-29e9-4347-e1c4-d30835afae9c","execution":{"iopub.status.busy":"2024-05-01T22:41:21.086111Z","iopub.status.idle":"2024-05-01T22:41:21.086603Z","shell.execute_reply.started":"2024-05-01T22:41:21.086348Z","shell.execute_reply":"2024-05-01T22:41:21.086384Z"}}

y = df['target'].replace('Normal', 0).replace('Anomaly', 1)
X = df.drop(['target', df.columns[0]], axis=1)


X

# %% [code] {"id":"6dIrJO6fpYg9","outputId":"88f2b07f-019a-4e90-ca12-74e586d8fe7a","execution":{"iopub.status.busy":"2024-05-01T22:41:21.088833Z","iopub.status.idle":"2024-05-01T22:41:21.089248Z","shell.execute_reply.started":"2024-05-01T22:41:21.089034Z","shell.execute_reply":"2024-05-01T22:41:21.089051Z"}}
len(y)

# %% [code] {"id":"JMpmt674Mcvp","execution":{"iopub.status.busy":"2024-05-01T22:41:21.090236Z","iopub.status.idle":"2024-05-01T22:41:21.090686Z","shell.execute_reply.started":"2024-05-01T22:41:21.090469Z","shell.execute_reply":"2024-05-01T22:41:21.090487Z"}}

# Extract features (X) and target variable (y)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



# Split data into training and testing sets



# %% [markdown] {"id":"bzrLXXf0L-dD"}
# # GRU

# %% [markdown] {"id":"iLHH_TFnMLkN"}
# 

# %% [code] {"id":"4cx1vIe9Al8J","outputId":"9e7d1a69-cb32-457b-b248-0b7968dadadd","execution":{"iopub.status.busy":"2024-05-01T22:41:21.092495Z","iopub.status.idle":"2024-05-01T22:41:21.092958Z","shell.execute_reply.started":"2024-05-01T22:41:21.092710Z","shell.execute_reply":"2024-05-01T22:41:21.092728Z"}}
import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, GRU
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Assuming you have already defined X, y, X_with_clusters

X_train, x_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.metrics import precision_score

X_train_array = X_train.values
X_validate_array = x_validate.values

X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1, 1)
X_validate_reshaped = X_validate_array.reshape(X_validate_array.shape[0], -1, 1)

print("Original shapes:")
print("y shape:", y.shape)

print("After train-test split:")
print("X_train shape:", X_train.shape)
print("x_validate shape:", x_validate.shape)
print("y_train shape:", y_train.shape)
print("y_validate shape:", y_validate.shape)

# Reshape input data
print("After reshaping:")
print("X_train_reshaped shape:", X_train_reshaped.shape)
print("X_validate_reshaped shape:", X_validate_reshaped.shape)

# %% [code] {"id":"0ffMJ_1sF4ix","outputId":"53e58b91-c97d-4566-aa06-f2a83c479c2d","execution":{"iopub.status.busy":"2024-05-01T22:41:21.095473Z","iopub.status.idle":"2024-05-01T22:41:21.095932Z","shell.execute_reply.started":"2024-05-01T22:41:21.095700Z","shell.execute_reply":"2024-05-01T22:41:21.095719Z"}}
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validate = scaler.transform(x_validate)

# Define your model
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.8),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.8),

    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_nn.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validate, y_validate))
# Calculate evaluation metrics
# Calculate evaluation metrics


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-05-01T22:41:21.097147Z","iopub.status.idle":"2024-05-01T22:41:21.097595Z","shell.execute_reply.started":"2024-05-01T22:41:21.097373Z","shell.execute_reply":"2024-05-01T22:41:21.097392Z"}}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras import layers
# Plot the loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Predict probabilities for the test set
y_pred_proba = model.predict(X_test)

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# %% [markdown] {"id":"n5WwirhfM8Lo"}
# # LSTM

# %% [code] {"id":"Hbsv9KYPM_Jb","outputId":"0c79cd08-3ce3-43ad-fef6-70fd36a4fd0a","execution":{"iopub.status.busy":"2024-05-01T22:41:21.099584Z","iopub.status.idle":"2024-05-01T22:41:21.100039Z","shell.execute_reply.started":"2024-05-01T22:41:21.099797Z","shell.execute_reply":"2024-05-01T22:41:21.099816Z"}}
from tensorflow import keras
from tensorflow.keras import layers

# Reshape the input data
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1, 1)
X_validate_reshaped = X_validate_array.reshape(X_validate_array.shape[0], -1, 1)

# Build the LSTM model
model_lstm_simple = keras.Sequential([
    layers.LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
        layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(10, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_lstm_simple.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
history = model_lstm_simple.fit(
    X_train_reshaped, y_train,
    validation_data=(X_validate_reshaped, y_validate),
    batch_size=64, epochs=100
)
# Train the model
# Calculate evaluation metrics
# Calculate evaluation metrics
accuracy = accuracy_score(y_validate, y_pred)
precision = precision_score(y_validate, y_pred)
recall = recall_score(y_validate, y_pred)
f1 = f1_score(y_validate, y_pred)
conf_matrix = confusion_matrix(y_validate, y_pred)
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_bigrubilstm.png')  # Save the plot as PNG

plt.show()

# Print confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(conf_matrix))
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Fill confusion matrix cells with respective values
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('conf_bigrubilstm.png')  # Save the plot as PNG

plt.show()




# Calculate AUC
auc = roc_auc_score(y_validate, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve ')

plt.legend()
plt.savefig('roc_bigrubilstm.png')  # Save the plot as PNG

plt.show()
# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# %% [code] {"execution":{"iopub.status.busy":"2024-05-01T22:41:21.101262Z","iopub.status.idle":"2024-05-01T22:41:21.101741Z","shell.execute_reply.started":"2024-05-01T22:41:21.101513Z","shell.execute_reply":"2024-05-01T22:41:21.101532Z"}}
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('LSTM.png')  # Save the plot as PNG

plt.show()

# Predict probabilities for the test set
y_pred_proba = model_lstm_simple.predict(X_validate)

# Convert probabilities to binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(y_validate, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend()
plt.savefig('roc_lstm.png')  # Save the plot as PNG

plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-01T22:41:21.102926Z","iopub.status.idle":"2024-05-01T22:41:21.103336Z","shell.execute_reply.started":"2024-05-01T22:41:21.103132Z","shell.execute_reply":"2024-05-01T22:41:21.103151Z"}}
y_validate

# %% [code] {"id":"_JeBfxZl8R2P","execution":{"iopub.status.busy":"2024-05-01T22:41:21.104826Z","iopub.status.idle":"2024-05-01T22:41:21.105240Z","shell.execute_reply.started":"2024-05-01T22:41:21.105026Z","shell.execute_reply":"2024-05-01T22:41:21.105043Z"}}
print(X_train_reshaped.shape)
print(X_validate_reshaped.shape)
print(y_train.shape)
print(y_validate.shape)


# %% [code] {"id":"m3SmDcl7Nc8y","execution":{"iopub.status.busy":"2024-05-01T22:41:21.107618Z","iopub.status.idle":"2024-05-01T22:41:21.108139Z","shell.execute_reply.started":"2024-05-01T22:41:21.107875Z","shell.execute_reply":"2024-05-01T22:41:21.107897Z"}}
from tensorflow import keras
from tensorflow.keras import layers

# Reshape the input data
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1, 1)
X_validate_reshaped = X_validate_array.reshape(X_validate_array.shape[0], -1, 1)

# Build the Bidirectional GRU model
model_bigru = keras.Sequential([
    layers.Bidirectional(layers.GRU(64, return_sequences=True), input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    layers.Bidirectional(layers.LSTM(64)),

    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_bigru.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model_bigru.fit(
    X_train_reshaped, y_train,
    validation_data=(X_validate_reshaped, y_validate),
    batch_size=64, epochs=100
)


# %% [code] {"id":"WhJ7YRcybYQj","execution":{"iopub.status.busy":"2024-05-01T22:41:21.109745Z","iopub.status.idle":"2024-05-01T22:41:21.110200Z","shell.execute_reply.started":"2024-05-01T22:41:21.109971Z","shell.execute_reply":"2024-05-01T22:41:21.109990Z"}}
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense

# Define the model
def create_bilstm_bigru_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        layers.BatchNormalization(),

        Bidirectional(GRU(128)),
        Dense(num_classes, activation='sigmoid')
    ])
    return model

# Example usage:
input_shape = (20, 1)  # Example input shape (sequence_length, input_dimension)
num_classes = 1

# Create the model
model_bigru_bilstm = create_bilstm_bigru_model(input_shape, num_classes)

# Display the model summary
model_bigru_bilstm.summary()
model_bigru_bilstm.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
history= model_bigru_bilstm.fit(
    X_train_reshaped, y_train,
    validation_data=(X_validate_reshaped, y_validate),
    batch_size=64, epochs=100
)
#98
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('bigrubilstm.png')  # Save the plot as PNG

plt.show()

# Predict probabilities for the test set
y_pred_proba = model_lstm_simple.predict(X_validate)

# Convert probabilities to binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(y_validate, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve ')

plt.legend()
plt.savefig('roc_bigrubilstm.png')  # Save the plot as PNG

plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-05-01T22:41:21.111745Z","iopub.status.idle":"2024-05-01T22:41:21.112317Z","shell.execute_reply.started":"2024-05-01T22:41:21.111967Z","shell.execute_reply":"2024-05-01T22:41:21.111986Z"}}
accuracy = accuracy_score(y_validate, y_pred)
precision = precision_score(y_validate, y_pred)
recall = recall_score(y_validate, y_pred)
f1 = f1_score(y_validate, y_pred)
conf_matrix = confusion_matrix(y_validate, y_pred)
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_bigrubilstm.png')  # Save the plot as PNG

plt.show()

# Print confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(conf_matrix))
plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Fill confusion matrix cells with respective values
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('conf_bigrubilstm.png')  # Save the plot as PNG

plt.show()




# Calculate AUC
auc = roc_auc_score(y_validate, y_pred_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_validate, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve ')

plt.legend()
plt.savefig('roc_bigrubilstm.png')  # Save the plot as PNG

plt.show()
# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# %% [markdown] {"id":"18y55p9VMfu7"}
# # RNN

# %% [code] {"id":"dZhoLjNBwW0j","execution":{"iopub.status.busy":"2024-05-01T22:41:21.114055Z","iopub.status.idle":"2024-05-01T22:41:21.114524Z","shell.execute_reply.started":"2024-05-01T22:41:21.114272Z","shell.execute_reply":"2024-05-01T22:41:21.114291Z"}}
from tensorflow import keras
from tensorflow.keras import layers

# Reshape the input data
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1, 1)
X_validate_reshaped = X_validate_array.reshape(X_validate_array.shape[0], -1, 1)

# Build the RNN model
model_rnn_simple = keras.Sequential([
    layers.SimpleRNN(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    layers.BatchNormalization(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
#bhtttt ziadalagayers laga don to

# Compile the model
model_rnn_simple.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model_rnn_simple.fit(
    X_train_reshaped, y_train,
    validation_data=(X_validate_reshaped, y_validate),
    batch_size=64, epochs=100
)


# %% [markdown] {"id":"KPp5mNwWMjWK"}
# # Ensembleing

# %% [markdown] {"id":"bI2cfMbtMoRQ"}
# # CNN

# %% [code] {"id":"Lw6svgm8uRr1","execution":{"iopub.status.busy":"2024-05-01T22:41:21.115819Z","iopub.status.idle":"2024-05-01T22:41:21.116309Z","shell.execute_reply.started":"2024-05-01T22:41:21.116066Z","shell.execute_reply":"2024-05-01T22:41:21.116086Z"}}
from tensorflow import keras
from tensorflow.keras import layers

# Reshape the input data
X_train_reshaped = X_train_array.reshape(X_train_array.shape[0], -1, 1)
X_validate_reshaped = X_validate_array.reshape(X_validate_array.shape[0], -1, 1)

# Build the CNN model
model_cnn = keras.Sequential([
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_cnn.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model_cnn.fit(
    X_train_reshaped, y_train,
    validation_data=(X_validate_reshaped, y_validate),
    batch_size=64, epochs=100
)


# %% [code] {"id":"L1UfjCB1yeib","execution":{"iopub.status.busy":"2024-05-01T22:41:21.117650Z","iopub.status.idle":"2024-05-01T22:41:21.118091Z","shell.execute_reply.started":"2024-05-01T22:41:21.117861Z","shell.execute_reply":"2024-05-01T22:41:21.117880Z"}}
import numpy as np

# Predictions from the LSTM model
lstm = model_lstm_simple.predict(X_validate_reshaped)

# Predictions from the Bidirectional GRU model
nn = model_nn.predict(X_validate_reshaped)

# Predictions from the Simple RNN model
cnn = model_cnn.predict(X_validate_reshaped)

# Predictions from the CNN model
rnn = model_rnn_simple.predict(X_validate_reshaped)

bigru = model_bigru.predict(X_validate_reshaped)
model_bigru_bilstm = model_bigru_bilstm.predict(X_validate_reshaped)

# Average the predictions of all models
#ensemble_preds = (bigru + rnn + cnn + nn + lstm) / 5
ensemble_preds = (bigru +model_bigru_bilstm ) / 2

# Check the shape of ensemble_preds
print("Shape of ensemble_preds:", ensemble_preds.shape)

# Flatten ensemble_preds if necessary
ensemble_preds_flat = ensemble_preds.flatten()

# Convert probabilities to class labels (assuming binary classification)
ensemble_labels = np.round(ensemble_preds_flat)

# Check the shape of y_validate
print("Shape of y_validate:", y_validate.shape)

# Evaluate the accuracy of the ensemble predictions
ensemble_accuracy = np.mean(ensemble_labels == y_validate)
print("Ensemble Accuracy:", ensemble_accuracy)
#phle 0.964 thi

# %% [markdown] {"id":"fUOujoPkmsyW"}
# 

# %% [code] {"id":"Dhw9y1OCOP_C","execution":{"iopub.status.busy":"2024-05-01T22:41:21.119268Z","iopub.status.idle":"2024-05-01T22:41:21.119724Z","shell.execute_reply.started":"2024-05-01T22:41:21.119498Z","shell.execute_reply":"2024-05-01T22:41:21.119521Z"}}
weak_learners = [
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("AdaBoost", AdaBoostClassifier()),

]

from sklearn.ensemble import VotingClassifier


# Create a voting classifier
voting_classifier = VotingClassifier(estimators=weak_learners, voting='hard')

# Train the voting classifier
voting_classifier.fit(X_train, y_train)

# Evaluate the voting classifier
y_pred_ensemble = voting_classifier.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble (Voting) - Accuracy: {ensemble_accuracy:.4f}")
#Ensemble (Voting) - Accuracy: 0.9860
#Ensemble (Voting) - Accuracy: 0.9863
#Ensemble (Voting) - Accuracy: 0.9864 decsion,random,ada
#Ensemble (Voting) - Accuracy: 0.9868 clusters ke sath


# %% [markdown]
# 