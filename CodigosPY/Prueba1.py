import Mnist_Loader
import Network
import pickle

training_data, validation_data , test_data = Mnist_Loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=Network.Network([784,45,10])

net.SGD( training_data, 5, 10, 3, test_data=test_data)

archivo = open("red_prueba1.pkl",'wb')
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