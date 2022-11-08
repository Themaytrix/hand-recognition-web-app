import tensorflow as tf
import numpy as np

##class to for inferencing model results
class KeyPointClassifier():
    def __init__(self, 
                model_path = 'hand-recognition-web-app/models/savedkeypointclassifies.tflite',
                num_threads =1):
        self.interpreter = tf.lite.Interpreter(model_path = model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        
    def __call__(self,landmarks):
        input_details_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_details_index,np.array([landmarks], dtype= np.float32))
        self.interpreter.invoke()
        output_details_index = self.output_details[0]['index']
        results = self.interpreter.get_tensor(output_details_index)
        results_index = np.argmax(np.squeeze(results))
        return results_index