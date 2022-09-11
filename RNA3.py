import mnist_loader
import network3
import pickle
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=network3.Network([784,30,10], cost=network3.CrossEntropyCost)
net.default_weight_initializer()
net.SGD( training_data, 30, 10, 0.1, lmbda=5, evaluation_data=validation_data, monitor_evaluation_accuracy=True)
archivo = open("red_prueba3.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo
archivo_lectura = open("red_prueba3.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
exit()
