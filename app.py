from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)

dic = ['This cell is malaria infected','This cell is not infected']

model = load_model('agada_malaria_detector.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(130,130))
	i = image.img_to_array(i)
	i = i.reshape(1, 130,130,3)
	p = (model.predict(i) > 0.5)*1
	return dic[p[0][0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)