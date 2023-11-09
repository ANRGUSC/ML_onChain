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
    relu: "nn.ReLU(" + expr + ")" -> relu
    
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

    def constructor(self, params : tuple, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'uint256 {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        biases = "\t\tbiases = new int256[][](input_dim);\n"
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
              weights_layer{i+1}.push(temp_w);
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
        return f'function classify({params_str}) public view returns (int[] memory) {{' + '\n'.join(filter(None, stmts)) + '\n\t}\n' # TODO return stmt

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
        if(var_type == 'int'):
            assignment = f'{x} = {y};'
            res = f'''
    function set{x}({var_type} memory value) public {{
        {x} = value;
    }}'''
        elif (var_type == 'int[]'):
            sz = re.search(r'\[(.*?)\]', y).group(1)
            self.variable_dims[x] = (sz)
            assignment = f'biases[{self.bias_count}] = new int256[](1);'
            self.bias_count += 1
            res =  f'''
    function set{x}({var_type} memory value) public {{
        for (uint256 i = 0; i < value.length; ++i) {{
            {x}[i] = value[i];
        }}
    }}'''
        else: # int[][]
            matches = re.findall(r'\[(.*?)\]', y)
            if matches:
                sz1, sz2 = matches
            self.variable_dims[x] = (sz1, sz2)
            assignment = f'{x} = new int[][]({sz1});\n' + \
                f'\t\tfor (uint256 i = 0; i < {sz1}; i++) {{\n' + \
                f'\t\t\t{x}[i] = new int[]({sz2});\n' + \
                f'\t\t}}\n'
            res = f'''function set{x}({var_type} memory value) public {{
        for (uint256 i = 0; i < value.length; ++i) {{
            for (uint256 j = 0; j < value[0].length; ++j) {{
                {x}[i][j] = value[i][j];
            }}
        }}
    }}'''
        self.setter_functions.append(res)
        return assignment
    
    # Pass assignment variables into contract
    # Generate update functions for each weight array

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
        return res

    def layer_pass(self, layer, x):
        self.num_layers += 1
        if layer in self.variable_dims:
            dims = self.variable_dims[layer]
            res_type = 'int[] memory '
            res_num = str(self.num_layers)
            if (self.num_layers == 1):
                prev_res = 'x'
            else:
                prev_res = 'res' + str(self.num_layers-1)
            if (isinstance(x,dict)):
                c_type = ''
            else:
                c_type = 'int '
            if (len(dims) == 2):
                res = f"""
        {res_type}res{res_num} = new int[]({dims[1]});
        {c_type}c;
        for (uint256 i = 0; i < {dims[1]}; ++i) {{
            c = 0;
            for (uint256 j = 0; j < x.length; ++j) {{
                c += {layer}[i][j] * {prev_res}[j];
            }}
            res{res_num}[i] = c;
        }}"""
                if (not isinstance(x, dict) or isinstance(x, dict) and x['type'] == 'CNAME'):
                    return res
                else:
                    return x['value'] + res
            else: # length is 1
                res = f"""
        {res_type}res{res_num} = new int[]({1});
        {c_type}c = 0;
        for (uint256 i = 0; i < {dims[0]}; ++i) {{
            c += {layer}[i] * {prev_res}[i];
        }}
        res{res_num}[0] = c;"""
                if (not isinstance(x, dict) or isinstance(x, dict) and x['type'] == 'CNAME'):
                    return res
                else:
                    return x['value'] + res
        else: # boilerplate
            res = f"""
            for (uint256 i = 0; i < {layer}.length; ++i) {{
                int c = 0;
                for (uint256 j = 0; j < {layer}[i].length; ++j) {{
                    c += {layer}[i][j] * {x}[j];
                }}
                {x}[i] = c;
            }}"""
            return res

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
        print("IN RELU")
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
        return x + res

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
        res = f"""
        for (uint256 i = 0; i < res.length; ++i) {{
            res[i] = sigmoid(res[i]);
        }}
        return res;
        """
        return x + res
    def dropout(self, x):
        if isinstance(x, dict):
            x = x['value']
        pass

# Code Generation + Types
# Assume inputs to functions are simply x

def get_ast(expr):
    parser = Lark(grammar, parser='earley')
    return parser.parse(expr)

def py_to_solidity(expr):
    tree = get_ast(expr)
    print(tree.pretty())
    transformer = SolidityTransformer()
    return transformer.transform(tree).strip()


# TODO
# import ABDK statement, Running + ABDK int support

# TODO
# Function calls have a gas cost overhead. If we're making a prediction on a batch of datapoints, even though the evm does not have gpu support, calling predict() with a matrix of datapoints and iterating is more efficient than calling predict() n times




# Assumptions:
# Activations functions only applied on the forward pass's return statement
# Input is flat tensors: MLP only works for 1D data anyway