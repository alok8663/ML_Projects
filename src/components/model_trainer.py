import os
import sys
import pickle
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Save the ANN model in .pkl format

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_ann_model(self, input_dim):
        """
        Build and compile an Artificial Neural Network model.
        """
        try:
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')  # Binary classification
            ])
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train the ANN model and evaluate it on the test set.
        """
        try:
            logging.info("Splitting training and testing data into inputs and targets.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f"Building the ANN model with input dimension: {X_train.shape[1]}")
            model = self.build_ann_model(input_dim=X_train.shape[1])

            logging.info("Training the ANN model.")
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

            logging.info("Evaluating the ANN model on test data.")
            y_pred = (model.predict(X_test) > 0.5).astype(int)  # Convert probabilities to binary predictions

            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {accuracy}")

            logging.info("Saving the trained ANN model in .pickle format.")
            with open(self.model_trainer_config.trained_model_file_path, 'wb') as file:
                pickle.dump(model, file)

            return accuracy
        except Exception as e:
            raise CustomException(e, sys)
