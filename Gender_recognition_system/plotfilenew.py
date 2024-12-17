import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Paths for the dataset and model
base_path = "/Users/sachetutekar/PycharmProjects/ECE 5831/Sachet/ECE 5831 Gender Recog Files"
image_dir = os.path.join(base_path, "img_align_celeba/img_align_celeba")
labels_file = os.path.join(base_path, "list_attr_celeba.csv")
model_path = 'gender_recognition_model.h5'

# Load labels
df = pd.read_csv(labels_file)
df.rename(columns={df.columns[0]: 'image_id'}, inplace=True)

# Filter dataframe to include only valid images
valid_images = os.listdir(image_dir)
df = df[df['image_id'].isin(valid_images)]

# Map gender labels (-1 for female, 1 for male -> 0 for female, 1 for male)
df['gender'] = df['Male'].apply(lambda x: 0 if x == -1 else 1)

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create data generators
img_size = 128
batch_size = 32
datagen = ImageDataGenerator(rescale=1. / 255)

train_data = datagen.flow_from_dataframe(
    train_df,
    directory=image_dir,
    x_col='image_id',
    y_col='gender',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='raw'
)

val_data = datagen.flow_from_dataframe(
    val_df,
    directory=image_dir,
    x_col='image_id',
    y_col='gender',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='raw'
)

test_data = datagen.flow_from_dataframe(
    test_df,
    directory=image_dir,
    x_col='image_id',
    y_col='gender',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

# Load the pre-trained model
model = load_model(model_path)

# Fine-tune the model
model.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Fine-tuning the model...")
history = model.fit(train_data, validation_data=val_data, epochs=1)

# Evaluate on test data
print("Evaluating on test data...")
test_predictions = (model.predict(test_data) > 0.5).astype(int).flatten()
test_accuracy = accuracy_score(test_df['gender'].values[:len(test_predictions)], test_predictions)
print(f"Test Accuracy: {test_accuracy:.2%}")

# Save the updated model
model.save('fine_tuned_gender_recognition_model.h5')
print("Fine-tuned model saved as 'fine_tuned_gender_recognition_model.h5'")

# Save the training history
with open('fine_tuning_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
