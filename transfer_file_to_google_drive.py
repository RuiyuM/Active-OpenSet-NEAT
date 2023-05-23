import shutil

# source folder (replace 'source_folder' with the name of your folder)
source = '/content/resnet_CLIP/log_AL'

# destination folder
destination = '/content/drive/MyDrive/colab_dataset_NEAT'

# move the folder using shutil.move
shutil.move(source, destination)
