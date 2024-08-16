import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import io

# Cài đặt cấu hình
st.title('Fashion MNIST Classifier')

# Tải dữ liệu và chuẩn bị dữ liệu
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Chuyển đổi dữ liệu
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train_ohe = to_categorical(y_train, num_classes=10)
y_test_ohe = to_categorical(y_test, num_classes=10)

# Tạo mô hình
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Huấn luyện mô hình
def train_model():
    model = create_model()
    model.fit(X_train, y_train_ohe, epochs=20, verbose=0)
    _, test_accuracy = model.evaluate(X_test, y_test_ohe, verbose=0)
    return model, test_accuracy

# Hiển thị kết quả huấn luyện
model, test_accuracy = train_model()
st.write(f"Test Accuracy: {test_accuracy:.2f}")

# Tải ảnh từ máy
st.subheader('Upload an image to predict')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Chuyển ảnh thành numpy array
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thay đổi shape thành (1, 28, 28)

    # Dự đoán
    y_pred_proba = model.predict(img_array)
    top_3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
    top_3_proba = y_pred_proba[0][top_3_indices]
    labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    top_3_labels = [labels[i] for i in top_3_indices]

    # Hiển thị kết quả dự đoán
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Top 3 predictions:")
    for label, proba in zip(top_3_labels, top_3_proba):
        st.write(f"{label}: {proba * 100:.2f}%")
