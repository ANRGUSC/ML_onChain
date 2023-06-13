from lark import Lark, Transformer, v_args
import astpretty
from textwrap import dedent


grammar = """
    start: classdef -> concat
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: funcdef+ -> concat
    stmt: (assign | return | super) -> stmt
    funcdef: (constructor | forward) -> stmt

    constructor: "def __init__(" parameters ")" ":" stmt+ -> constructor
    forward: "def forward(" parameters ")" ":" stmt* -> predict
    parameters: expr "," expr -> parameters
    assign: (cassign | fassign) -> stmt
    cassign: "self" "." CNAME "=" expr -> cassign
    fassign: CNAME "=" expr -> fassign

    expr: (linear | conv | sign | layer_pass | NUMBER | CNAME) -> expr
    linear: "nn.Linear(" + parameters + ")" -> linear
    
    conv: conv2d -> stmt
    conv2d: "nn.Conv2d(" + NUMBER + ", " + NUMBER + ", " + NUMBER + ")" -> conv2d
    sign: "torch.sign(" + expr + ")" -> sign
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
    def concat(self, *args):
        return ''.join(args)

    def stmt(self, stmt):
        if stmt is None:
            return None
        else:
            return '\t' + str(stmt)

    def contract(self, name, inherit, func2):
        return f'contract {name} {{ \n {func2} }}\n'

    def constructor(self, params : tuple, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'int {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        return f'constructor({params_str}) {{\n ' + '\n'.join(filter(None, stmts)) + '\n\t}}\n'

    def predict(self, params, *stmts):
        typed_params = []
        for i in range(1, len(params)):
            typed_params.append(f'int {params[i]}')

        params_str = ', '.join(str(p) for p in typed_params)
        return f'predict({params_str}) {{\n ' + '\n'.join(filter(None,stmts)) + '\n\t}}\n'

    def parameters(self, x, y):
        return (x, y)

    def cassign(self, x, y):
        return f'CAssign {x} = {y}'

    def fassign(self, x, y):
        return f'FAssign {x} = {y}'

    def expr(self, x):
        return f'{x}'
    
    def ret_val(self, x):
        return f'\treturn {x}'
    
    def super(self, x):
        return None

    def conv2d(self, *args):
        pass
    
    def linear(self, params):
        res = f'int[{params[0]}][{params[1]}]'
        return res

    def layer_pass(self, layer, x):
        res = f"""for (int i = 0; i < {layer}.length; ++i) {{
                int c = 0;
                for (int j = 0; j < {layer}[0].length; ++j) {{
                    int c = 0;

                    {x}[i] = c;
                }}
                {x}[i] = c;
            }}"""
        return res

    def sign(self, x):
        res = f"""
            for (int i = 0; i < x.length; ++i) {{
                x[i] = ((x[i] >= 0) ? ((x[i] == 0) ? 0 : 1) : -1)
            }}
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



torch_code = """
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sign(self.fc(x))"""
# Testing the converter
tree = get_ast(torch_code)
print(tree.pretty())

output = py_to_solidity(torch_code)
print(output)
