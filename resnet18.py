from allresnets import resnet18instance #call the particular resnet model instance you want to you here 
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.callbacks import CSVLogger, ModelCheckpoint
import pandas as pd
from datatf import train, val

resnet18instance.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy(), Precision(), Recall()])
resnet18instance.summary()
#define callbacks
logdir = 'logs'
csv_logger = CSVLogger('training_history.csv')
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

# #Train the Model
history = resnet18instance.fit(train, epochs=20, validation_data=val, callbacks=[csv_logger, checkpoint])

#save the history in a datframe
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history_df.csv', index=False)