"""
Training utilities for the drowsiness detection model.
Provides a simple pipeline to prepare eye‑region datasets,
train a classifier, and export the model.
"""
import os
import json
import cv2
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models

from .inference import DrowsinessDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DrowsinessTrainer:
    """
    Simple training pipeline for a drowsiness detection model.
    The workflow is:
        1. Load eye‑region images from a structured dataset.
        2. Optionally extract features (e.g., EAR, MAR) or raw pixels.
        3. Train a classifier (RandomForest) or a small CNN.
        4. Persist the model and update configuration.
    """

    DEFAULT_TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def __init__(self, config_path: str):
        """
        Load configuration and set up output paths.

        Args:
            config_path: Path to JSON config that contains training settings
                         and the location of the processed dataset.
        """
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

        self.processed_dir = self.cfg.get('processed_dataset_path',
                                          'data/processed/eye_samples')
        self.model_output_path = self.cfg.get('model_path',
                                              'models/drowsiness_model.h5')
        self.report_path = self.cfg.get('training_report_path',
                                        'logs/training_report.json')

        self.model = None

    def _load_image_data(self):
        """
        Load images from a directory that is organized as:
            <processed_dir>/open/*.jpg
            <processed_dir>/closed/*.jpg
        Returns:
            images (np.ndarray): normalized pixel arrays, shape (N, H, W, 3)
            labels (list): 'open' or 'closed' strings
        """
        images, labels = [], []
        for label in ['open', 'closed']:
            folder = os.path.join(self.processed_dir, label)
            if not os.path.isdir(folder):
                logger.warning(f"Folder {folder} does not exist – skipping.")
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(folder, fname)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not read {img_path}")
                    continue
                img = cv2.resize(img, (128, 128))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(label)
        logger.info(f"Loaded {len(images)} images ({'open':0}{'closed':0})")
        return np.array(images), labels

    def _extract_ear_mar(self, img):
        """
        Dummy helper – in a real project you'd extract eye/mouth landmarks.
        Here we just return static values for illustration.
        """
        # Placeholder: assume a medium EAR, high MAR when closed
        return 0.3, 0.7

    def train_classifier(self):
        """
        Train a RandomForest on raw pixel data (or handcrafted features).
        Returns the trained model.
        """
        images, labels = self._load_image_data()
        if images.shape[0] == 0:
            raise ValueError("No training images found. Check the dataset path.")

        # Flatten each image for RandomForest
        flat = images.reshape(len(images), -1)

        logger.info("Training RandomForest classifier...")
        clf = RandomForestClassifier(
            n_estimators=self.cfg.get('rf_n_estimators', 200),
            max_depth=self.cfg.get('rf_max_depth', 15),
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(flat, labels)
        return clf

    def train_cnn(self, input_shape=(128, 128, 3), num_classes=2):
        """
        Train a tiny CNN (mostly pedagogical) on eye‑region crops.
        """
        images, labels = self._load_image_data()
        if images.shape[0] == 0:
            raise ValueError("No training images found.")

        # Convert string labels to numeric classes
        unique, inv = np.unique(labels, return_inverse=True)
        y = inv

        # Train‑test split
        X_train, X_val, y_train, y_val = train_test_split(
            images, y, test_size=self.DEFAULT_TEST_SIZE,
            stratify=y, random_state=self.RANDOM_STATE
        )

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        logger.info("Training CNN...")
        history = model.fit(
            X_train, y_train,
            epochs=self.cfg.get('cnn_epochs', 15),
            batch_size=self.cfg.get('cnn_batch_size', 32),
            validation_data=(X_val, y_val),
            verbose=1
        )

        # Simple evaluation
        val_acc = history.history['val_accuracy'][-1]
        logger.info(f"CNN validation accuracy: {val_acc:.3f}")

        self.model = model
        return model

    def save_model(self):
        """
        Persist the trained model to disk and remember the path in config.
        """
        if self.model is None:
            raise RuntimeError("No model has been trained yet.")
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        self.model.save(self.model_output_path)
        logger.info(f"Model saved to {self.model_output_path}")

        # Update config with model path (if not already present)
        if isinstance(self.cfg, dict):
            self.cfg['model_path'] = self.model_output_path
            with open(self.cfg.get('config_path', 'config.json'), 'w') as f:
                json.dump(self.cfg, f, indent=2)
            logger.info("Configuration updated with new model path")

    def generate_training_report(self, train_args=None, eval_metrics=None):
        """
        Produce a JSON report summarising the training session.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'processed_dataset': self.processed_dir,
            'model_path': self.model_output_path,
            'algorithm': train_args or 'RandomForest',
            'parameters': eval_metrics or {}
        }
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report written to {self.report_path}")

    # ------------------------------------------------------------------ #
    # Convenience entry point
    # ------------------------------------------------------------------ #
    @classmethod
    def run_from_cli(cls, config_path: str):
        """
        CLI entry point – train, evaluate, and persist a model.
        """
        trainer = cls(config_path)
        # Example: train a random forest on raw pixels
        model = trainer.train_classifier()
        # Optionally replace with CNN: trainer.train_cnn()
        trainer.save_model()
        trainer.generate_training_report()
        logger.info("Training pipeline completed.")


if __name__ == "__main__":
    import argparse, json, os
    parser = argparse.ArgumentParser(description="Train drowsiness detection model")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    DrowsinessTrainer.run_from_cli(args.config)