import numpy as np
import torch
import onnxruntime as ort



class ONNXModel:
    def __init__(self, model_path):
        self.model = ort.InferenceSession(model_path)

    def preprocess_obs(self, obs):
        # obs shape should be :  (1, 19)
        # obs type should be  :  <class 'numpy.ndarray'>

        obs[0, 13] = obs[0, 13] / np.pi / 2
        return obs
    
    def predict(self, obs):
        # obs = obs.cpu().numpy()
        outputs = self.model.run(
            None,
            {"obs": obs},
        )
        return outputs
    
    def postprocess_actions(self, outputs):
        outputs[0][0][0] = np.clip(outputs[0][0][0], -1, 1)
        outputs[0][0][1] = np.clip(outputs[0][0][1], -1, 1)
        outputs[0][0][2] = np.clip(outputs[0][0][2], -1, 1)
        outputs[0][0][3] = np.clip(outputs[0][0][3], -1, 1)
        outputs[0][0][4] = np.clip(outputs[0][0][4], -1, 1)
        
        outputs[0][0][0] = (outputs[0][0][0] + 1) / 2 # convert the thrusts to the range [0, 1]
        outputs[0][0][1] = (outputs[0][0][1] + 1) / 2
        outputs[0][0][2] = (outputs[0][0][2] + 1) / 2
        outputs[0][0][3] = (outputs[0][0][3] + 1) / 2
        return outputs