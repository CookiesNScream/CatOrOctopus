#Data preparation
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import pandas as pd
import os
from duckduckgo_search import DDGS
from fastcore.all import L

def prepare_data():

  def create_dataframe_from_directory(target_dir, target_class, dataframe, maxImageCount=750):

    # Get a random image path
    #random_image = random.sample(os.listdir(target_folder), 1)
    random_image = (os.listdir(target_dir), 1)

    image_paths = []
    labels = []

    #maxImageCount = 750 #this should be 750 or depending on the web scrape trust 
    imageCount = 0
    while (imageCount <= maxImageCount):
      for i in range(len(random_image[0])): #if "blah" not in somestring: 
        if ".cat" not in random_image[0][i] and "Thumbs.db" not in random_image[0][i]:
          # df = mpimg.imread(target_dir + "/" + random_image[0][i])
          # #dataframe.append(df.tolist())
          # dataframe.append(df)
          image_paths.append(target_dir + "/" + random_image[0][i])
          labels.append(target_class)
          imageCount += 1

    
    df = pd.DataFrame({'filename': image_paths, 'class': labels})
    dataframe.append(df)
    return dataframe
  
  def create_dataframe_from_webscrape(target_number, target_class, dataframe):
    def search_images(term, max_images=30):
      with DDGS() as ddgs:
          # Retrieve a list of search results with a specified maximum
          search_results = ddgs.images(keywords=term, max_results=max_images)
          # Extract the 'image' URLs from the search results
          image_urls = [result.get("image") for result in search_results]
          # Convert the list to a fastai L object (if needed)
          return L(image_urls)
    
    image_paths = search_images(f"{target_class} in the wild", max_images=target_number)
    labels = []
    for i in range(len(image_paths)):
      labels.append(target_class)

    df = pd.DataFrame({'filename': image_paths, 'class': labels})
    print(df)
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

  training_dataframe_supplemented = create_dataframe_from_webscrape(188, "octopus", dataframe_training)

  #test_data_suplemented1 = create_dataframe_from_directory(f"{root}/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals/animals/cat",
  #                                                    "cat", 
  #                                                    dataframe_test)
  #print(len(dataframe_test))
  final_test_dataframe = pd.concat(dataframe_test, ignore_index=True)
  final_training_dataframe = pd.concat(dataframe_training, ignore_index=True)




  train_datagen_noAug = ImageDataGenerator(rescale=1/255.) # without data augmentation
  valid_datagen = ImageDataGenerator(rescale=1/255.)
  train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20, # rotate the image slightly between 0 and 20 degrees (note: this is an int not a float)
                                             shear_range=0.2, # shear the image
                                             zoom_range=0.2, # zoom into the image
                                             width_shift_range=0.2, # shift the image width ways
                                             height_shift_range=0.2, # shift the image height ways
                                             horizontal_flip=True)

  train_data = train_datagen_noAug.flow_from_dataframe(final_training_dataframe,
                                                batch_size=32, # number of images to process at a time 
                                                target_size=(224, 224), # convert all images to be 224 x 224
                                                class_mode="binary", # type of problem we're working on
                                                seed=42,
                                                shuffle=True)

  test_data = valid_datagen.flow_from_dataframe(final_test_dataframe,
                                                batch_size=32, # number of images to process at a time 
                                                target_size=(224, 224), # convert all images to be 224 x 224
                                                class_mode="binary", # type of problem we're working on
                                                seed=42,
                                                shuffle=True)

  return train_data, test_data


'''
60 cat test i want 250
60 octo test i want 250 
1623 cat train i want 750
562 octo train i want 750
'''
prepare_data()