from flask import Flask, render_template, request

import numpy as np
import os

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model

model =load_model("C:/Users/vivek/OneDrive/Desktop/final_img_class/brain_tumor_m1.h5")
print('@@ Model loaded')

def tumor_postiv_negat(yes_no):
  test_image = load_img(yes_no, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image).round(3) # predict class horse or human
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result) # get the index of max value
  predicted_class = np.asscalar(np.argmax(result, axis=1))
  accuracy = round(result[0][predicted_class] * 100, 2)

  

  if pred == 0:
    return "No tumor" # if index 0 
  else:
    return "meningioma_tumor" # if index 1




# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
  
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/images', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred = tumor_postiv_negat(yes_no=file_path)
              
        return render_template('predict.html', pred_output = pred, user_image = file_path)
    
#Fo local system
if __name__ == "__main__":
    app.run() 
