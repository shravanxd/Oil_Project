import os

# Define dataset paths (Update these with your actual dataset locations)
train_images_path = r"C:\Users\shrav\Downloads\archive\dataset\images\train"
train_labels_path = r"C:\Users\shrav\Downloads\archive\dataset\labels\train"
val_images_path = r"C:\Users\shrav\Downloads\archive\dataset\images\val"
val_labels_path = r"C:\Users\shrav\Downloads\archive\dataset\labels\val"

# Function to check missing files
def check_missing_files(image_path, label_path, dataset_name):
    """
    Checks if all images have corresponding annotations and vice versa.
    """
    image_files = {f.split(".")[0] for f in os.listdir(image_path) if f.endswith(".jpg")}
    label_files = {f.split(".")[0] for f in os.listdir(label_path) if f.endswith(".txt")}

    missing_labels = image_files - label_files  # Images without annotations
    missing_images = label_files - image_files  # Annotations without images

    print(f"\n📌 Dataset: {dataset_name}")
    print(f"Total Images: {len(image_files)}")
    print(f"Total Annotations: {len(label_files)}")

    if missing_labels:
        print(f"🚨 Missing annotation files for {len(missing_labels)} images.")
        print(f"Files: {missing_labels}")

    if missing_images:
        print(f"🚨 Missing images for {len(missing_images)} annotation files.")
        print(f"Files: {missing_images}")

    if not missing_labels and not missing_images:
        print("✅ All images and annotations are correctly paired!")

# Run checks for train and validation datasets
if __name__ == "__main__":
    check_missing_files(train_images_path, train_labels_path, "Training Set")
    check_missing_files(val_images_path, val_labels_path, "Validation Set")