#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# Minimum imports to support running the project.
# - We use the preview Custom Vision Python API referenced [here](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/python-tutorial)
# - `yaml` for Config (see below for why we like this)
# - `os` for getting at the filesystem
# - `Augmentor` is a nice tidy image augmentation library for Python. Details [here](https://github.com/mdbloice/Augmentor).

#%%
import yaml
import os
from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry
import time
from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

#%% [markdown]
# We externalize our secrets such that we can `.gitignore` the `config.yaml` file. This allows us to avoid checking secrets into version control because doing so is clearly the devil's work. We like YAML mainly because we dislike JSON (also the devil's work).

#%%
with open("config.yaml", 'r') as yamlfile:
    cfg = yaml.load(yamlfile)

credentials = cfg['credentials']

#%% [markdown]
# We have the project to use the default `general` domain. 

#%%
training_key = credentials['training_key']
prediction_key = credentials['prediction_key']

trainer = training_api.TrainingApi(training_key)

# Create a new project
print ("Creating project...")
project = trainer.create_project("PuffyVsShell")
print(project.id)


#%%
# Make two tags in the new project
puffy_tag = trainer.create_tag(project.id, "Puffy")
shell_tag = trainer.create_tag(project.id, "Shell")


#%%
# Upload Puffy Jackets
puffy_dir = "../gear_images/insulated_jackets"
for image in os.listdir(os.fsencode(puffy_dir)):
    with open(puffy_dir + "/" + os.fsdecode(image), mode="rb") as img_data:
        print('Uploading: {}'.format(image))
        trainer.create_images_from_data(project.id, img_data.read(), [ puffy_tag.id ])
print('Done')


#%%
shell_dir = "../gear_images/hardshell_jackets"
for image in os.listdir(os.fsencode(shell_dir)):
    with open(shell_dir + "/" + os.fsdecode(image), mode="rb") as img_data:
        print('Uploading: {}'.format(image))
        trainer.create_images_from_data(project.id, img_data.read(), [ shell_tag.id ])
print('Done')

#%% [markdown]
# Train the project

#%%
print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status == "Training"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Make it the default project endpoint
trainer.update_iteration(project.id, iteration.id, is_default=True)
print ("Done!")

#%% [markdown]
# We now test with two images that were not used in training.
# 
# Puffy Jacket
# <img src="http://assets.supremenewyork.com/142654/zo/C9rZR7_iq1g1.jpg" alt="Drawing" width="200" height="200"/>
# 
# Hardshell Jacket
# <img src="https://pro.arcteryx.com/assets/images/products/mountain-guide-jacket@2x.png" alt="Drawing" width="200" height="200"/>

#%%
# Now there is a trained endpoint that can be used to make a prediction
predictor = prediction_endpoint.PredictionEndpoint(prediction_key)

test_img_url = "http://assets.supremenewyork.com/142654/zo/C9rZR7_iq1g1.jpg"
results = predictor.predict_image_url(project.id, iteration.id, url=test_img_url)
  
# Display the results.
for prediction in results.predictions:
    print ("Prediction for {}".format(test_img_url))
    print ("\t" + prediction.tag + ": {0:.2f}%".format(prediction.probability * 100))
    
    
test_img_url = "https://pro.arcteryx.com/assets/images/products/mountain-guide-jacket@2x.png"
results = predictor.predict_image_url(project.id, iteration.id, url=test_img_url)
  
# Display the results.
for prediction in results.predictions:
    print ("Prediction for {}".format(test_img_url))
    print ("\t" + prediction.tag + ": {0:.2f}%".format(prediction.probability * 100))


#%%
#Delete the project
project = trainer.delete_project(project.id)


