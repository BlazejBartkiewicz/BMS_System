import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import RandomSearch


def load_data(data_loc):
    data_frame = pd.read_csv(data_loc)
    return data_frame

#Loading data from file to data frame
data_frame = load_data('Data/Dane_rounded.csv')

print(data_frame.dtypes) #=======================================

#Dropping unused columns

X = data_frame.drop(['HourId',
                     'Day',
                     'Chiller Output (kW)',
                     'Chiller Input (kW)',
                     'Total Building Electric [kW]',
                     'Precool Coil Load (kW)',
                     'Preheat Coil Load (kW)',
                     'Terminal Cooling Coil Load (kW)',
                     'Terminal Heating Coil Load (kW)',
                     'Ventilation Fan (kW)',
                     'Exhaust Fan (kW)',
                     'Terminal Fan (kW)',
                     'Vent. Reclaim Device (kW)',
                     'Lighting (kW)',
                     'Electric Equipment (kW)'], axis=1)
print(X) #=======================================

X = data_frame[['Month', 'Week day', 'Hour', 'Dry-Bulb Temp (Â°C)']].values
y = data_frame['Heating Load (kW)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_heating_model():
    model = Sequential()

    model.add(Dense(64, input_dim=4, activation='relu'))

    model.add(Dense(96, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    return model


heating_model = create_heating_model()
history = heating_model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))
mse, mae = heating_model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}, Mean Absolute Error: {mae}')

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.legend(['Loss', 'Val_loss'], loc='upper right')
  plt.grid(True)

def plot_mae(history):
  plt.plot(history.history['mae'], label='mae')
  plt.plot(history.history['val_mae'], label='val_mae')
  #plt.xlabel('loss')
  plt.xlabel('Epoch')
  plt.legend(['mae', 'Val_mae'], loc='upper right')
  plt.grid(True)


plot_mae(history)

plot_loss(history)

heating_model.save("Heating_model_test.keras")