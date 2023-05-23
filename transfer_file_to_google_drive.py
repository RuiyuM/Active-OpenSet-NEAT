import os

directory = "/content/drive/MyDrive/colab_dataset_NEAT/log_AL"

# Ensure the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f'Directory {directory} created.')
else:
    print(f'Directory {directory} already exists.')


