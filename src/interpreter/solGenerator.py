from textwrap import dedent

# Default function to handle unknown model types
def unknown_model_type():
    return "Unknown model type"

def perceptron_gen(**kwargs):
    model_name = kwargs.get('model_type')
    num_weights = kwargs.get('num_weights')

    contract_name = "Perceptron"

    contract = dedent(f"""\
    pragma solidity >=0.4.22 <0.9.0;
    
    contract Perceptron {{

        int[] public weights;
        int public bias;

        constructor(int[{num_weights}] memory initial_weights, int initial_bias) public {{
            weights = initial_weights;
            bias = initial_bias;
        }}

        function predict(int[{num_weights}] memory inputs) public view returns (int8) {{
            require(inputs.length == weights.length, "Length of inputs does not match length of weights");
            
            int sum = bias;
            for(uint i = 0; i < weights.length; i++) {{
                sum += weights[i] * inputs[i];
            }}

            return (sum >= 0) ? int8(1) : -1;
        }}

        function updateWeights(int[{num_weights}] memory new_weights) public {{
            require(new_weights.length == weights.length, "Length of new_weights does not match current weights");
            weights = new_weights;
        }}

        function updateBias(int new_bias) public {{
            bias = new_bias;
        }}
    }}
    """)
    # write the contract to a .sol file
    with open(f"../classifiers/{model_name}.sol", "w") as file:
        file.write(contract)

def code_gen(*args, **kwargs):
    model_func_mapping = {
        "perceptron": perceptron_gen,
    }
    model_type = kwargs.get('model_type')
    func = model_func_mapping.get(model_type, unknown_model_type)
    print(f"Generating {model_type} contract code")
    func(**kwargs)
