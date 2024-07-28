import matplotlib.pyplot as plt
from onnx_model import ONNXModel
import numpy as np
import torch
import torch.nn as nn

# Define the function
# def f(z, R):
#     return 1 / (1 - (R /(4*(z-0.2)))**2)

# # Parameters
# R = 0.1016
# z_values = np.linspace(1.5, 0.25, 400)

# # Calculate f(z) for each z
# f_values = f(z_values, R)

# # Plot the function
# plt.figure(figsize=(10, 6))
# plt.plot(z_values, f_values, label='$f(z) = \\frac{1}{1 - (\\frac{R}{4z})^2}$', color='blue')
# plt.title('Plot of $f(z) = \\frac{1}{1 - (\\frac{R}{4z})^2}$ for $R = 0.10$')
# plt.xlabel('z')
# plt.ylabel('f(z)')
# plt.grid(True)
# plt.legend()
# plt.show()

# Define the following network : MLP with 512 256 and 128 hidden units. Input : 18 observations, output : 5 values
# Elu activation function

from rl_games.torch_runner import Runner
import os
import yaml
import torch
import matplotlib.pyplot as plt
import gym
import numpy as np
import onnx
import onnxruntime as ort
import rl_games.algos_torch.flatten as flatten

from IPython import embed

onnx_instance = ONNXModel("/home/m4pc/Documents/IsaacGymEnvs/isaacgymenvs/MorphingLander.onnx")

onnx_model = onnx.load("/home/m4pc/Documents/IsaacGymEnvs/isaacgymenvs/MorphingLander.onnx")
onnx.checker.check_model(onnx_model)

ort_model = ort.InferenceSession("/home/m4pc/Documents/IsaacGymEnvs/isaacgymenvs/MorphingLander.onnx")
input_data = [[0, 0, 1.27401733e+00, 0,
               0, 0,0 , 1,
               0, -0.00000000e+00, 0, 0,
               0, 0, 0,0,
               0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]

outputs = onnx_instance.predict(input_data)
outputs = onnx_instance.postprocess_actions(outputs)
print(outputs)


# np_array = np.array(input_data)
# print("dummy inputs : ", np.zeros((1, 19)).astype(np.float32))
# outputs = ort_model.run(
#     None,
#     {"obs": input_data},
# )
# print(outputs)