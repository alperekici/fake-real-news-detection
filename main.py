import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight


# Veri setlerini yükle
trueNews = pd.read_csv('dataset/True.csv')
fakeNews = pd.read_csv('dataset/Fake.csv')

# Etiketle
fakeNews['label'] = 0
trueNews['label'] = 1

# Remove unnecessary columns
trueNews.drop(columns=["title", "date", "subject"], inplace=True)
fakeNews.drop(columns=["title", "date", "subject"], inplace=True)

# Merge datasets
data = pd.concat([trueNews, fakeNews], ignore_index=True)

# Null and Duplicate checking
data.drop_duplicates(inplace=True)


# Metin ön işleme
def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if len(word) > 1]
    return " ".join(words)


# Metinleri temizle
data['cleaned_text'] = data['text'].apply(process_text)

# Veri ve etiket ayır
X = data['cleaned_text']
y = data['label']

# Eğitim ve test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizer: tüm veride eğit
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

vocab_size = len(tokenizer.word_index)
print("Vocabulary size:", vocab_size)

# Metinleri dizilere çevir
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
maxlen = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Sınıf ağırlıkları
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Model mimarisi
input_layer = Input(shape=(maxlen,))
x = Embedding(vocab_size + 1, 100)(input_layer)
x = Dropout(0.3)(x)
x = LSTM(100, return_sequences=True)(x)
x = Dropout(0.3)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Eğitimi başlat
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    validation_data=(X_test_pad, y_test),
    class_weight=class_weights
)

# Eğitim grafikleri
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Test verisinde değerlendirme
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Confusion Matrix
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Tahmin fonksiyonu
def predict_news(news_text):
    cleaned = process_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen)
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        print("\n🟩 Tahmin: GERÇEK HABER (Real)")
    else:
        print("\n🟥 Tahmin: SAHTE HABER (Fake)")


# Kullanıcıdan giriş alma
while True:
    user_input = input("\nBir haber metni girin (çıkmak için 'q' yazın):\n> ")
    if user_input.lower() == 'q':
        print("Program sonlandırıldı.")
        break
    predict_news(user_input)
