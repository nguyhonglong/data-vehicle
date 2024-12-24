import os
import shutil
from sklearn.model_selection import train_test_split
import random

def create_dataset_structure(base_path):
    """Create the dataset directory structure"""
    # Create main dataset directory
    dataset_path = os.path.join(base_path, 'dataset')
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create train and val directories with their subdirectories
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_path, split)
        os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_path, 'labels'), exist_ok=True)

def split_data(source_dir, dataset_dir, train_ratio=0.8, random_state=42):
    """Split data into train and validation sets"""
    # Get all image files
    image_files = [f for f in os.listdir(source_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split the data
    train_files, val_files = train_test_split(
        image_files, 
        train_size=train_ratio,
        random_state=random_state
    )
    
    # Copy files to respective directories
    for files, split in [(train_files, 'train'), (val_files, 'val')]:
        for image_file in files:
            # Get corresponding label file
            label_file = os.path.splitext(image_file)[0] + '.txt'
            
            # Source paths
            src_image = os.path.join(source_dir, image_file)
            src_label = os.path.join(source_dir, label_file)
            
            # Destination paths
            dst_image = os.path.join(dataset_dir, split, 'images', image_file)
            dst_label = os.path.join(dataset_dir, split, 'labels', label_file)
            
            # Copy files
            if os.path.exists(src_image) and os.path.exists(src_label):
                shutil.copy2(src_image, dst_image)
                shutil.copy2(src_label, dst_label)

def main():
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'Long')
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # Create dataset structure
    create_dataset_structure(base_dir)
    
    # Split and copy data
    split_data(source_dir, dataset_dir, train_ratio=0.8)
    
    print("Dataset split completed!")
    
    # Count files in each directory
    for split in ['train', 'val']:
        n_images = len(os.listdir(os.path.join(dataset_dir, split, 'images')))
        n_labels = len(os.listdir(os.path.join(dataset_dir, split, 'labels')))
        print(f"{split} set: {n_images} images, {n_labels} labels")

if __name__ == "__main__":
    main() 