import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import io

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train_ohe = to_categorical(y_train, 10)
y_test_ohe = to_categorical(y_test, 10)

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.sidebar.title('Navigation')
option = st.sidebar.radio('Go to', ['Train', 'Inference'])

if option == 'Train':
    st.title('Train Model')
    
    st.subheader('Visualize Data')
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        ids = np.where(y_train == i)[0]
        for j in range(10):
            target = np.random.choice(ids)
            axs[i, j].imshow(X_train[target], cmap='gray')
            axs[i, j].axis('off')
    st.pyplot(fig)
    
    if st.button('Train'):
        model = create_model()
        st.write("Training...")
        history = model.fit(X_train, y_train_ohe, epochs=20, verbose=1, validation_data=(X_test, y_test_ohe))
        
        test_loss, test_acc = model.evaluate(X_test, y_test_ohe, verbose=2)
        st.write(f'Test accuracy: {test_acc:.4f}')
        
        model.save('trained_model.h5')
        
        st.subheader('Training History')
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Accuracy')
        ax.plot(history.history['loss'], label='Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.legend()
        st.pyplot(fig)

elif option == 'Inference':
    st.title('Image Inference')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L').resize((28, 28))
        img_array = 1 - np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        model = create_model()
        model.load_weights('trained_model.h5')
        y_pred_proba = model.predict(img_array)
        
        top_3_indices = np.argsort(y_pred_proba[0])[-3:][::-1]
        top_3_proba = y_pred_proba[0][top_3_indices]
        labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        top_3_labels = [labels[i] for i in top_3_indices]
        
        st.subheader('Prediction Results')
        for label, proba in zip(top_3_labels, top_3_proba):
            st.write(f"{label}: {proba * 100:.2f}%")
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
