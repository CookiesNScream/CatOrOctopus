#This is where I will run all of the functions
from kaggleDownloads import download_data
from dataPreparation import prepare_data
from modelContruction import construct_model
from modelTraining import train_model

download_data() #Use Kaggle hub and whatnot to download the data

training_data, test_data = prepare_data() # should return into two variables or a list with two items

model = construct_model() #could take params or not idk

history = train_model(model, training_data, test_data)



