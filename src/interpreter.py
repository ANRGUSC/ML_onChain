from lark import Lark, Transformer, v_args
from textwrap import dedent
import re

grammar = """
    start: classdef -> concat
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: funcdef+ -> concat_line
    stmt: (assign | return | super) -> stmt
    funcdef: (constructor | forward) -> stmt

    constructor: "def __init__(" parameters ")" ":" stmt+ -> constructor
    forward: "def forward(" parameters ")" ":" stmt* -> classify
    parameters: (expr "," expr) | ((CNAME | NUMBER) "," (CNAME | NUMBER)) -> parameters
    assign: (cassign | fassign) -> stmt
    cassign: "self" "." CNAME "=" expr -> cassign
    fassign: CNAME "=" expr -> fassign

    expr: (linear | conv | sign | sigmoid | layer_pass | NUMBER | relu) -> expr
    linear: "nn.Linear(" + parameters + ")" -> linear
    relu: "F.relu(" + expr + ")" -> relu

    conv: conv2d -> stmt
    conv2d: "nn.Conv2d(" + NUMBER + ", " + NUMBER + ", " + NUMBER + ")" -> conv2d
    sign: "torch.sign(" + expr + ")" -> sign
    sigmoid: "torch.sigmoid(" + expr + ")" -> sigmoid
    dropout: "self.dropout(" + expr + ")" -> dropout
    layer_pass: ("self." + CNAME + "(" + CNAME + ")") | ("self." + CNAME + "(" + expr + ")") -> layer_pass

    return: "return " expr -> concat
    super: "super(" + ALL -> super
    NUMBER: /[0-9]+/
    CNAME: /[a-zA-Z_][a-zA-Z_0-9]*/
    ALL: /[^\\n]+/
    %import common.WS
    %ignore WS
"""


# expr can be a self transformation
# softmax
# sigmoid
# leaky_relu

# flatten: "torch.flatten(" + dim + (("," + starting_dim") | + (", " + starting_dim + ", " + end_dim))? + ")" -> flatten


@v_args(inline=True)
class SolidityTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_vars = []
        self.setter_functions = []
        self.variable_dims = {}
        self.has_sigmoid = False
        self.num_layers = 0
        self.bias_count = 0
        self.assigned_layers = 0
        self.layer_dims = []

    def concat(self, *args):
        res = ""
        for ar in args:
            if type(ar) is dict:
                res += ar['value']
            else:
                res += ar
        return res

    def concat_line(self, *args):
        return '\n'.join(args)

    def stmt(self, stmt):
        if stmt is None:
            return None
        else:
            if isinstance(stmt, dict):
                stmt = stmt['value']
            return '\t' + str(stmt)

    def contract(self, name, inherit, func2):
        self.contract_vars.append('int256[][] public biases;')
        self.contract_vars.append('int256[][] public training_data;')
        self.contract_vars.append('int public correct_Count;')
        contract_var_str = '\n\t'.join(self.contract_vars)
        setter_func_str = '\n\n\t'.join(self.setter_functions)
        return f'contract {name} {{\n\t{contract_var_str}\n\n\t{setter_func_str}\n\n{func2} }}\n'

    def constructor(self, params: tuple, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'uint256 layer_num')

        params_str = ', '.join(str(p) for p in typed_params)
        biases = "\t\tbiases = new int256[][](layer_num);\n"
        constructor = f'constructor({params_str}) {{\n ' + biases + '\n'.join(filter(None, stmts)) + '\n\t}\n'
        dataset_size = f"""
        function view_dataset_size() external view returns(uint256 size) {{
            size = training_data.length;
        }} """

        set_training_data = f"""
        function set_TrainingData(int256[] calldata d) external {{
            int256[] memory temp_d= new int256[](d.length);
            for (uint256 i = 0; i < d.length; i++) {{
                temp_d[i] = d[i];
            }}
            training_data.push(temp_d);
        }}
        """

        set_biases = f"""
        function set_Biases(uint256 layer, int256[] calldata b) external {{
            require(b.length == biases[layer].length, "Size of input biases does not match neuron number");
            biases[layer] = b;
        }}
        """

        additional_layers = ""
        for i in range(1, self.assigned_layers):
            cur_layer = f"""
          else if (layer == {i}) {{\
              weights_layer{i + 1}.push(temp_w);
          }}"""
            additional_layers += cur_layer

        set_weights = f"""
        function set_Weights(uint256 layer, int256[] calldata w) external {{
            require(layer < {self.assigned_layers}, "Layer index out of bounds");
            int256[] memory temp_w = new int256[](w.length);
            for (uint256 i = 0; i < w.length; i++) {{
                temp_w[i] = w[i];
            }}
            if (layer == 0) {{
                weights_layer1.push(temp_w);
            }} {additional_layers}
        }}

        """
        return constructor + '\n\n' + set_biases + set_weights + dataset_size + '\n\n' + set_training_data + '\n\n'

    def classify(self, params, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'int[] memory {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        stmts_string = '\n'.join(filter(None, stmts))
        res = f""" 
        function classify() public view returns (int) {{
            int correct = 0;
            for (uint256 j = 0; j < 50; j++) {{
              int256[] memory data = training_data[j];
              int256 label = data[0];

              {stmts_string}
              int256 classification;
              SD59x18 point_five = sd(0.5e18);
              if (neuronResultsLayer{self.assigned_layers}[0].gte(point_five)) {{
                  classification = int256(1e18);
              }} else {{
                  classification = int256(0e18);
              }}

              if (label == classification) {{
                  correct++;
              }}
            }}
            return correct;
        }}\n"""
        return res

    def parameters(self, x, y):
        if isinstance(x, dict):
            x = x['value']
        if isinstance(y, dict):
            y = y['value']
        return (x, y)

    def cassign(self, x, y):
        if isinstance(y, dict):
            y = y['value']
        var_type = 'int[]' if '[' in y and ']' in y else 'int'
        if (y.count('[') == 2):
            var_type = 'int[][]'
        self.contract_vars.append(f'int256[][] public weights_layer{self.assigned_layers + 1};')
        self.assigned_layers += 1
        layer_size = self.layer_dims[self.bias_count]
        assignment = f'biases[{self.bias_count}] = new int256[]({layer_size});'
        self.bias_count += 1
        sz = re.search(r'\[(.*?)\]', y).group(1)
        self.variable_dims[x] = (sz)
        return assignment

    def fassign(self, x, y):
        if isinstance(y, dict):
            y = y['value']
        return f'{y}'

    def expr(self, x):
        # If the expression matches the CNAME grammar, then it's a CNAME
        if re.match(r'[a-zA-Z_][a-zA-Z_0-9]*', x):
            return x
        else:
            # Otherwise, it's an expression
            return {'type': 'expression', 'value': x}

    def ret_val(self, x):
        if isinstance(x, dict):
            x = x['value']
        return f'return {x};'

    def super(self, x):
        return None

    def conv2d(self, *args):
        pass

    def linear(self, params):
        if (params[1] == '1'):
            res = f'int[{params[0]}]'
        else:
            res = f'int[{params[0]}][{params[1]}]'
        self.layer_dims.append(params[1])
        print("layer_dims-------", params[1])
        return res

    def layer_pass(self, layer, x):
        self.num_layers += 1
        if layer in self.variable_dims:
            if (self.num_layers == 1):
                # data operation
                res = f"SD59x18[] memory neuronResultsLayer{self.num_layers} = new SD59x18[](weights_layer{self.num_layers}.length);\n"
                matmult = f""" \
              for (uint256 n = 0; n < weights_layer{self.num_layers}.length; n++) {{
                neuronResultsLayer{self.num_layers}[n] = SD59x18.wrap(biases[{self.num_layers - 1}][n]);
                for (uint256 i = 1; i < data.length; i++) {{
                  neuronResultsLayer{self.num_layers}[n] = neuronResultsLayer{self.num_layers}[n].add(SD59x18.wrap(data[i]).mul(SD59x18.wrap(weights_layer{self.num_layers}[n][i-1])));
                }}
              }}
              \n"""
                res += matmult
                return res
            else:
                # operate on last layer
                res = f"SD59x18[] memory neuronResultsLayer{self.num_layers} = new SD59x18[](weights_layer{self.num_layers}.length);\n"
                matmult = f""" \
              for (uint256 n = 0; n < weights_layer{self.num_layers}.length; n++) {{
                neuronResultsLayer{self.num_layers}[n] = SD59x18.wrap(biases[{self.num_layers - 1}][n]);
                for (uint256 i = 0; i < weights_layer{self.num_layers - 1}.length; i++) {{
                  neuronResultsLayer{self.num_layers}[n] = neuronResultsLayer{self.num_layers}[n].add(neuronResultsLayer{self.num_layers - 1}[i].mul(SD59x18.wrap(weights_layer{self.num_layers}[n][i])));
                }}
              }}
              \n"""
                res += matmult
                return x + res
        else:
            raise ValueError('Forward pass layer not defined.')

    def sign(self, x):
        if isinstance(x, dict):
            x = x['value']
        res_num = str(self.num_layers)
        res = f"""
        for (uint256 i = 0; i < res{res_num}.length; ++i) {{
            res{res_num}[i] = ((res{res_num}[i] >= 0) ? ((res{res_num}[i] == 0) ? int(0) : int(1)) : -1);
        }}
        return res{res_num};
        """
        return x + res

    def relu(self, x):
        if isinstance(x, dict):
            x = x['value']
        res_num = str(self.num_layers)
        res = f"""
        //relu activation function
        function relu(SD59x18 x) public pure returns (SD59x18) {{
            int256 zero = 0;
            SD59x18 zero_cvt = convert(zero);
            if (x.gte(zero_cvt)) {{
                return x;
            }}
            return zero_cvt;
        }}
        """
        self.setter_functions.append(res)
        x = x[:x.rfind('}')]  # remove last bracket
        x += f"neuronResultsLayer{self.num_layers}[n] = relu(neuronResultsLayer{self.num_layers}[n]);\n"
        x += '\t\t}\n\n'
        return x

    def sigmoid(self, x):
        if isinstance(x, dict):
            x = x['value']
        if (not self.has_sigmoid):
            self.has_sigmoid = True
            res = f"""
    function sigmoid(SD59x18 x) public pure returns (SD59x18) {{
        int256 one = 1;
        SD59x18 one_cvt = convert(one);
        return (one_cvt).div(one_cvt.add((-x).exp()));
    }}
            """
            self.setter_functions.append(res)
        x = x[:x.rfind('}')]  # remove last bracket
        x += f"neuronResultsLayer{self.num_layers}[n] = sigmoid(neuronResultsLayer{self.num_layers}[n]);\n"
        x += '\t\t}\n\n'
        return x

    def dropout(self, x):
        if isinstance(x, dict):
            x = x['value']
        pass


def get_ast(expr):
    parser = Lark(grammar, parser='earley')
    return parser.parse(expr)


def py_to_solidity(expr):
    tree = get_ast(expr)
    print(tree.pretty())
    transformer = SolidityTransformer()
    return transformer.transform(tree).strip()

# Assumptions:
# Activations functions only applied on the forward pass's return statement
# Input is flat tensors: MLP only works for 1D data anyway