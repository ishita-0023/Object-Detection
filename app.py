from fileinput import filename
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
from csv import writer

app = Flask(__name__)

model = load_model('models/treeclassifier.h5')

def model_predict(img_path,model):
    img = cv2.imread(img_path)
    re_size = tf.image.resize(img, (256,256))
    # plt.imshow(re_size.numpy().astype(int))
    yhat = model.predict(np.expand_dims(re_size/255, 0))
    print(yhat)
    if yhat < 0.5: 
        return 'No Tree'
    else:
        return 'Tree'

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        # get the file from post request
        f=request.files['file']

        # save the file to uploads folder
        
        basepath=os.path.dirname(os.path.realpath('__file__'))
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        
        #result
        result = model_predict(file_path, model)
        return result
    return None

if __name__=='__main__':
    app.run(debug=False,port=5926)
