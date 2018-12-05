import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
longitud,altura = 150 ,150
modelo = './modelo/modelo.h5'
pesos_modelo = 'modelo/pesos.h5'
cnn = keras.models.load_model(modelo)
cnn.load_weights(pesos_modelo)

##    classes=['perro','gato','pajaro'],
def predict(file): 
    x = load_img(file,target_size=(longitud,altura))
    x = img_to_array(x)
    x = np.expand_dims(x,axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    classes = ['perro','gato','pajaro','rinoceronte','gorila']
    print("es un {0}".format(classes[answer]))
    return answer
