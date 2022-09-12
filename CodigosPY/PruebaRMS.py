import Mnist_Loader
import NetworkRMS
import pickle

training_data, validation_data , test_data = Mnist_Loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=NetworkRMS.Network([784,50,10])

net.SGD( training_data, 35, 10, 0.004, 0.99, 1e-09, test_data=test_data)

archivo = open("red_pruebaRMS004.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()

#leer el archivo

archivo_lectura = open("red_prueba1.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()

net.SGD( training_data, 10, 10, 3.0, test_data=test_data)

#esquema de como usar la red :
#imagen = leer_imagen("disco.jpg")
#print(net.feedforward(imagen))