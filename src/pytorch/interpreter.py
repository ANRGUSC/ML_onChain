from lark import Lark, Transformer, v_args
import astpretty
from textwrap import dedent
import subprocess
import re


grammar = """
    start: classdef -> concat
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: funcdef+ -> concat_line
    stmt: (assign | return | super) -> stmt
    funcdef: (constructor | forward) -> stmt

    constructor: "def __init__(" parameters ")" ":" stmt+ -> constructor
    forward: "def forward(" parameters ")" ":" stmt* -> predict
    parameters: expr "," expr -> parameters
    assign: (cassign | fassign) -> stmt
    cassign: "self" "." CNAME "=" expr -> cassign
    fassign: CNAME "=" expr -> fassign

    expr: (linear | conv | sign | sigmoid | layer_pass | NUMBER | CNAME) -> expr
    linear: "nn.Linear(" + parameters + ")" -> linear
    
    conv: conv2d -> stmt
    conv2d: "nn.Conv2d(" + NUMBER + ", " + NUMBER + ", " + NUMBER + ")" -> conv2d
    sign: "torch.sign(" + expr + ")" -> sign
    sigmoid: "torch.sigmoid(" + expr + ")" -> sigmoid
    dropout: "self.dropout(" + CNAME + ")" -> dropout
    layer_pass: "self." + CNAME + "(" + CNAME + ")" -> layer_pass

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

    def concat(self, *args):
        return ''.join(args)

    def concat_line(self, *args):
        return '\n'.join(args)

    def stmt(self, stmt):
        if stmt is None:
            return None
        else:
            return '\t' + str(stmt)

    def contract(self, name, inherit, func2):
        contract_var_str = '\n\t'.join(self.contract_vars)
        setter_func_str = '\n\n\t'.join(self.setter_functions)
        return f'contract {name} {{\n\t{contract_var_str}\n\n\t{setter_func_str}\n\n{func2} }}\n'

    def constructor(self, params : tuple, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'uint {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        return f'constructor({params_str}) {{\n ' + '\n'.join(filter(None, stmts)) + '\n\t}\n'

    def predict(self, params, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'int[] memory {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        return f'function predict({params_str}) public view returns (int[] memory) {{' + '\n'.join(filter(None, stmts)) + '\n\t}\n' # TODO return stmt

    def parameters(self, x, y):
        return (x, y)

    def cassign(self, x, y):
        var_type = 'int[]' if '[' in y and ']' in y else 'int'
        if (y.count('[') == 2):
            var_type = 'int[][]'
        self.contract_vars.append(f'{var_type} public {x};')
        if(var_type == 'int'):
            assignment = f'{x} = {y};'
            res = f'''
            function set{x}({var_type} memory value) public {{
                {x} = value;
            }}'''
        elif (var_type == 'int[]'):
            sz = re.search(r'\[(.*?)\]', y).group(1)
            self.variable_dims[x] = (sz)
            assignment = f'{x} = new int[]({sz});'
            res =  f'''
    function set{x}({var_type} memory value) public {{
        for (uint i = 0; i < value.length; ++i) {{
            {x}[i] = value[i];
        }}
    }}'''
        else: # int[][]
            matches = re.findall(r'\[(.*?)\]', y)
            if matches:
                sz1, sz2 = matches
            self.variable_dims[x] = (sz1, sz2)
            assignment = f'{x} = new int[][]({sz1});\n' + \
                f'for (uint i = 0; i < {sz1}; i++) {{\n' + \
                f'\t{x}[i] = new int[]({sz2});\n' + \
                f'}}\n'
            res = f'''function set{x}({var_type} memory value) public {{
                for (uint i = 0; i < value.length; ++i) {{
                    for (uint j = 0; j < value[0].length; ++j) {{
                        {x}[i][j] = value[i][j];
                    }}
                }}
            }}'''
        self.setter_functions.append(res)
        return assignment
    
    # Pass assignment variables into contract
    # Generate update functions for each weight array

    def fassign(self, x, y):
        return f'FAssign {x} = {y};'

    def expr(self, x):
        return f'{x}'
    
    def ret_val(self, x):
        return f'return {x};'
    
    def super(self, x):
        return None

    def conv2d(self, *args):
        pass
    
    # TODO, update Linear to support 2d arrays
    def linear(self, params):
        if (params[1] == '1'):
            res = f'int[{params[0]}]'
        else:
            res = f'int[{params[0]}][{params[1]}]'
        return res

    def layer_pass(self, layer, x):
        if layer in self.variable_dims:
            dims = self.variable_dims[layer]
            if (len(dims) == 2):
                res = f"""
        int[][] memory res = new int[][]({dims[1]});
        for (uint i = 0; i < {dims[1]}; ++i) {{
            int c = 0;
            for (uint j = 0; j < {dims[0]}; ++j) {{
                c += {layer}[i][j] * {x}[j];
            }}
            res[i] = c;
        }}"""
                return res
            else: # length is 1
                res = f"""
        int[] memory res = new int[]({1});
        int c = 0;
        for (uint i = 0; i < {dims[0]}; ++i) {{
            c += {layer}[i] * x[i];
        }}
        res[0] = c;"""
                return res
        else:
            res = f"""
            for (uint i = 0; i < {layer}.length; ++i) {{
                int c = 0;
                for (uint j = 0; j < {layer}[i].length; ++j) {{
                    c += {layer}[i][j] * {x}[j];
                }}
                {x}[i] = c;
            }}"""
            return res

    def sign(self, x):
        res = f"""
        for (uint i = 0; i < res.length; ++i) {{
            res[i] = ((res[i] >= 0) ? ((res[i] == 0) ? int(0) : int(1)) : -1);
        }}
        return res;
        """
        return x + res

    def sigmoid(self, x):
        if (not self.has_sigmoid):
            self.has_sigmoid = True
            res = f"""
    function sigmoid(int x) public pure returns (int64) {{
        int64 x64 = ABDKMath64x64.fromInt(x);

        // Now, we compute the negative of x64.
        int64 negX64 = ABDKMath64x64.neg(x64);

        // Then, we compute e^(negX64).
        int64 expNegX64 = ABDKMath64x64.exp(negX64);

        // Next, we add 1 to expNegX64. 
        int64 onePlusExpNegX64 = ABDKMath64x64.add(ABDKMath64x64.fromInt(1), expNegX64);

        // Finally, we compute the reciprocal of onePlusExpNegX64, which gives us the result of the sigmoid function.
        int64 sigmoidResult = ABDKMath64x64.inv(onePlusExpNegX64);

        return sigmoidResult;
    }}
            """
            self.setter_functions.append(res)
        res = f"""
        for (uint i = 0; i < res.length; ++i) {{
            res[i] = sigmoid(res[i]);
        }}
        return res;
        """
        return x + res
    def dropout(self, x):
        pass

# Code Generation + Types
# Assume inputs to functions are simply x

def get_ast(expr):
    parser = Lark(grammar, parser='earley')
    return parser.parse(expr)

def py_to_solidity(expr):
    parser = Lark(grammar, parser='earley')
    tree = parser.parse(expr)
    transformer = SolidityTransformer()
    return transformer.transform(tree).strip()


# TODO
# import ABDK statement, Running + ABDK int support