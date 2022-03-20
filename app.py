import base64
from io import BytesIO
from pickle import TRUE
from flask import Flask, jsonify, request 
from flask_cors import CORS
from PIL import Image
from pylab import *
from PIL import Image, ImageChops, ImageEnhance
from keras.models import load_model
import numpy as np

app = Flask(__name__)

CORS(app)

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = 'tempresaved.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality = quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    #ela_im.save("result.jpg",'JPEG')
    return ela_im

def process(img):
    trained_model = load_model("trained.h5")
    ##display(orig_img)
    

    valid = []
    valid.append(np.array(convert_to_ela_image(img,100).resize((60, 60))).flatten() / 255.0)
    valid = np.array(valid)
    valid = valid.reshape(-1, 60, 60, 3)
    val_test = trained_model.predict(valid)
    val_test = np.argmax(val_test,axis = 0)[0]
    print(val_test)
    return val_test


@app.route('/', methods=["GET"])
def index():
    response = jsonify({"name":"jayashankar"})
    return response

@app.route('/getResult',methods=["POST"])
def result():
    data={}
    buff = BytesIO()
    res = request.files.items()
    for i in res:
        img = i[1]
        data['contentType']=img.content_type
        data['image']= base64.b64encode(img.stream.read()).decode("utf8")
        ela_img = convert_to_ela_image(img,100)
        ela_img.save(buff,format="JPEG")
        data['output'] =  base64.b64encode(buff.getvalue()).decode("utf8")
        data['result']=str(process(img))
        
    #print(data)
    return data

if __name__=="__main__":
    app.run( port=5000, debug=True)