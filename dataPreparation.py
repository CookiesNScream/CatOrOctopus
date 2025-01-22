#Data preparation
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
import os

def prepare_data():

  def create_dataframe_from_directory(target_dir, target_class, dataframe):

    # Get a random image path
    #random_image = random.sample(os.listdir(target_folder), 1)
    random_image = (os.listdir(target_dir), 1)

    image_paths = []
    labels = []

    
    for i in range(len(random_image[0])): #if "blah" not in somestring: 
      if ".cat" not in random_image[0][i] and "Thumbs.db" not in random_image[0][i]:
        # df = mpimg.imread(target_dir + "/" + random_image[0][i])
        # #dataframe.append(df.tolist())
        # dataframe.append(df)
        image_paths.append(target_dir + "/" + random_image[0][i])
        labels.append(target_class)

    
    df = pd.DataFrame({'filename': image_paths, 'class': labels})
    dataframe.append(df)
    return dataframe

  dataframe_test = ([])
  dataframe_training = ([])
  root = "/Users/ErenYeager"
  cat_test_dataframe = create_dataframe_from_directory(f"{root}/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals/animals/cat",
                                                      "cat", 
                                                      dataframe_test)
  cat_and_octopus_test_dataframe = create_dataframe_from_directory(f"{root}/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals/animals/octopus", 
                                                          "octopus", 
                                                          dataframe_test)
  cat_training_dataframe = create_dataframe_from_directory(f"{root}/.cache/kagglehub/datasets/crawford/cat-dataset/versions/2/CAT_03", 
                                                          "cat", 
                                                          dataframe_training)
  cat_and_octopus_training_dataframe = create_dataframe_from_directory(f"{root}/.cache/kagglehub/datasets/vencerlanz09/sea-animals-image-dataste/versions/5/Octopus", 
                                                                      "octopus", 
                                                                      dataframe_training)

  final_test_dataframe = pd.concat(dataframe_test, ignore_index=True)
  final_training_dataframe = pd.concat(dataframe_training, ignore_index=True)




  train_datagen_noAug = ImageDataGenerator(rescale=1/255.) # without data augmentation
  valid_datagen = ImageDataGenerator(rescale=1/255.)
  train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=10, # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                             shear_range=0.1, # shear the image
                                             zoom_range=0.1, # zoom into the image
                                             width_shift_range=0.1, # shift the image width ways
                                             height_shift_range=0.1, # shift the image height ways
                                             horizontal_flip=True)

  train_data = train_datagen_augmented.flow_from_dataframe(final_training_dataframe,
                                                batch_size=32, # number of images to process at a time 
                                                target_size=(224, 224), # convert all images to be 224 x 224
                                                class_mode="binary", # type of problem we're working on
                                                seed=42)

  test_data = valid_datagen.flow_from_dataframe(final_test_dataframe,
                                                batch_size=32, # number of images to process at a time 
                                                target_size=(224, 224), # convert all images to be 224 x 224
                                                class_mode="binary", # type of problem we're working on
                                                seed=42)
  return train_data, test_data