import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# Define the paths for train, test, and validation data
train_path = 'dataset/train'
test_path = 'dataset/test'

# Data Augmentation Parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Function to check if the augmented images already exist
def already_augmented(image_file, folder_path):
    # Check if any augmented version of the image exists in the folder
    augmented_images = [f for f in os.listdir(folder_path) if f.startswith('aug_' + image_file)]
    return len(augmented_images) > 0

# Function to apply augmentation and save images back to their folders
def augment_images_in_folder(folder_path):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    for image_file in image_files:
        # Skip augmentation if the image has already been augmented
        if already_augmented(image_file, folder_path):
            print(f"Skipping augmentation for {image_file} as it has already been augmented.")
            continue
        
        image_path = os.path.join(folder_path, image_file)
        img = load_img(image_path)  # Load image
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for ImageDataGenerator
        
        # Apply augmentations and save them back to the same folder
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=folder_path, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 5:  # Limit number of augmented images per original image (5 new images)
                break

# Function to preprocess and augment images in all folders (train and test)
def preprocess_and_augment():
    # Iterate through the train and test directories
    for root_dir in [train_path, test_path]:
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for plant in os.listdir(category_path):
                    plant_path = os.path.join(category_path, plant)
                    if os.path.isdir(plant_path):
                        print(f"Processing images in: {plant_path}")
                        augment_images_in_folder(plant_path)

if __name__ == "__main__":
    # Run the preprocessing and augmentation process
    preprocess_and_augment()
    print("Preprocessing and augmentation complete!")
