import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Paths for the dataset
base_path = "/Users/sachetutekar/PycharmProjects/ECE 5831/Sachet/ECE 5831 Gender Recog Files"
image_dir = os.path.join(base_path, "img_align_celeba/img_align_celeba")
labels_file = os.path.join(base_path, "list_attr_celeba.csv")

# Step 1: Load and preprocess the data
# Load labels
# Load labels
df = pd.read_csv(labels_file)
df.rename(columns={df.columns[0]: 'image_id'}, inplace=True)

# Check all valid image files in the directory
valid_images = os.listdir(image_dir)
print(f"Number of valid image files in directory: {len(valid_images)}")

# Filter dataframe to include only valid images
df = df[df['image_id'].isin(valid_images)]
print(f"Total rows in dataframe after filtering: {df.shape[0]}")

if df.empty:
    raise ValueError("Filtered dataframe is empty. Verify that image directory and CSV file align.")


# Map gender labels (-1 for female, 1 for male -> 0 for female, 1 for male)
df['gender'] = df['Male'].apply(lambda x: 0 if x == -1 else 1)

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Step 2: Create data generators
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
    shuffle=False  # Ensure test data order matches for accuracy calculation
)

# Step 3: Define and compile the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
print("Training the model...")
model.fit(train_data, validation_data=val_data, epochs=10)

# Step 5: Fine-tune the model
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Fine-tuning the model...")
model.fit(train_data, validation_data=val_data, epochs=5)

# Step 6: Evaluate on test data
print("Evaluating on test data...")
test_predictions = (model.predict(test_data) > 0.5).astype(int).flatten()
test_accuracy = accuracy_score(test_df['gender'].values[:len(test_predictions)], test_predictions)

print(f"Test Accuracy: {test_accuracy:.2%}")

# Step 7: Save the model
model.save('gender_recognition_model.h5')
print("Model saved as 'gender_recognition_model.h5'")
