import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from skopt import gp_minimize
from skopt.space import Real, Integer
import matplotlib.pyplot as plt

# Veri setini yükle
data = pd.read_csv("veriseti.csv", sep=';', decimal=',')
X = data[['B', 'S', 'H0', 'H', 'Q', 'W', 'R']].values
y = data['dB'].values

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Erken durdurma
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Model oluşturma fonksiyonu
def create_model(hidden_layers, hidden_units, learning_rate, dropout_rate):
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(hidden_units, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    for _ in range(hidden_layers - 1):
        x = Dense(hidden_units, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Bayes optimizasyonu için hedef fonksiyon
def objective(params):
    hidden_layers, hidden_units, learning_rate, dropout_rate = params
    model = create_model(hidden_layers, hidden_units, learning_rate, dropout_rate)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stopping])
    val_loss = min(history.history['val_loss'])
    return val_loss

# Bayes optimizasyonu için parametre alanları
space = [
    Integer(1, 5, name='hidden_layers'),
    Integer(100, 300, name='hidden_units'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Real(0.2, 0.4, name='dropout_rate')
]

# Bayes optimizasyonu
result = gp_minimize(objective, space, n_calls=20, random_state=42)

# En iyi hiperparametrelerle modeli oluşturma
best_model = create_model(result.x[0], result.x[1], result.x[2], result.x[3])

# Eğitim sırasında metrikleri takip etmek için özel callback
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_test, y_test, tolerance=5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.tolerance = tolerance

    def on_epoch_end(self, epoch, logs=None):
        # Eğitim ve test setlerinde tahminler
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_test_pred = self.model.predict(self.X_test, verbose=0)

        # R² skorları
        r2_train = r2_score(self.y_train, y_train_pred)
        r2_test = r2_score(self.y_test, y_test_pred)

        # Tolerans doğruluk metriği
        train_accuracy = np.mean(np.abs(self.y_train - y_train_pred.flatten()) <= self.tolerance)
        test_accuracy = np.mean(np.abs(self.y_test - y_test_pred.flatten()) <= self.tolerance)

        print(f"Epoch {epoch + 1}: R² (Eğitim) = {r2_train:.4f}, R² (Test) = {r2_test:.4f}, "
              f"Doğruluk (Eğitim) = {train_accuracy:.4f}, Doğruluk (Test) = {test_accuracy:.4f}")

# MetricsCallback oluşturma
metrics_callback = MetricsCallback(X_train, y_train, X_test, y_test, tolerance=5)

# Modeli eğitme
history = best_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, metrics_callback]
)

# Tahminler
y_pred = best_model.predict(X_test)

# Metrikler
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Eğitim ve doğrulama kaybı grafiği
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid()
plt.show()
