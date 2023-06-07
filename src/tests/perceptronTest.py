import sys
sys.path.append('../')
from interpreter.solGenerator import code_gen

# Single Layer Perceptron Test
def main():
    # Generate Perceptron contract
    num_weights = 5
    code_gen(model_type='perceptron', num_weights=num_weights)

    # Call Perceptron contract and compare output py-solc + web3.py



if __name__ == '__main__':
    main()