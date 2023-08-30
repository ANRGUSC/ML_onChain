from interpreter import *
import astor
import ast

def translate_model(file_path, class_name, output_path):
    with open(file_path, 'r') as file:
        model_code = file.read()

    module = ast.parse(model_code)
    class_def = next((node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name), None)

    if class_def is None:
        raise ValueError(f"Class {class_name} not found in file {file_path}")

    class_code = astor.to_source(class_def)
    solidity_code = py_to_solidity(class_code)

    output = f"// SPDX-License-Identifier: UNLICENSED\npragma solidity >=0.4.22 <0.9.0;\n\n{solidity_code}"
    print(output)

    with open(output_path, 'w') as file:
        file.write(output)

address = "../contracts/classifiers/"

# Use the function to translate the Perceptron and LogisticRegressionModel
translate_model("models.py", "Perceptron", address+"perceptron.sol")
translate_model("models.py", "MultiPerceptron", address+"multi_perceptron.sol")
#translate_model("models.py", "LogisticRegressionModel", address+"log_regression.sol")
