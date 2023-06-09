from lark import Lark, Transformer, v_args
import astpretty
from textwrap import dedent


grammar = """
    start: classdef -> concat
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: funcdef+ -> suite
    stmt: (assign | return | super) -> stmt
    funcdef: (constructor | forward) -> stmt

    constructor: "def __init__(" parameters ")" ":" stmt+-> constructor
    forward: "def forward(" parameters ")" ":" stmt+-> predict
    parameters: CNAME "," CNAME -> parameters
    assign: (cassign | fassign) -> stmt
    cassign: "self" "." CNAME "=" expr -> cassign
    fassign: CNAME "=" expr -> fassign

    expr: (linear | flatten | conv | sign) -> expr
    linear: "nn.Linear(" + input_dim + ", " + output_dim + ")" -> linear
    flatten: "torch.flatten(" + dim ((+ "," + starting_dim") | (+ "," + starting_dim + "," + end_dim))? + ")" -> flatten
    conv: conv2d -> stmt
    conv2d: "nn.Conv2d(" + in_channels + ", " + out_channels + ", " + kernel_size + ")" -> conv2d
    sign: "torch.sign(" + expr + ")" -> sign
    dropout: "self.dropout(" + x + ")" -> dropout

    return: "return " expr -> ret_val
    super: "super(" + ALL -> super
    CNAME: /[a-zA-Z_][a-zA-Z_0-9]*/
    ALL: /[^\\n]+/
    %import common.WS
    %ignore WS
"""

# expr can be a self transformation
# softmax
# sigmoid
# leaky_relu

# cassign: "self" "." CNAME "=" linear
# return: "return torch.sign(self.fc(x))" -> sign
# linear: "nn.Linear(" input_dim ", " NUMBER ")" -> linear


@v_args(inline=True)
class SolidityTransformer(Transformer):
    def concat(self, *args):
        return ''.join(args)

    def suite(self, *args):
        return ''.join(args)

    def stmt(self, funcdef):
        return '\t' + str(funcdef)

    def contract(self, name, inherit, func2):
        return f'contract {name} {{ \n {func2} }}\n'

    def constructor(self, params, *stmts):
        return f'constructor({params}) {{\n ' + '\n'.join(stmts) + '\n\t}}\n'

    def predict(self, params, *stmts):
        return f'predicting({params}) {{\n ' + '\n'.join(stmts) + '\n\t}}\n'

    def parameters(self, x, y):
        return f'{x}, {y}'
    
    def literal(self, x):
        return str(x)

    def cassign(self, x, y):
        return f'CAssign {x} = {y}'

    def fassign(self, x, y):
        return f'FAssign {x} = {y}'

    def expr(self, x):
        return f'{x}'
    
    def ret_val(self, x):
        return f'\treturn {x}'
    
    def super(self, x):
        return f'\tsuper({x}'

    def flatten(self, *args):
        pass

    def conv2d(self, *args):
        pass

    def sign(self, x):
        res = dedent(f' res = int[] array2;
        for (uint i=0
                            i < x.length
                            + +i) {{
                                \t res.push((x[i] >= 0) ? ((x[i] == 0) ? 0 : 1) : -1);
                            }}')
        return res

    def dropout(self, x):
        pass



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
        x = 5

    def forward(self, x):
        y = 1
        return torch.sign(self.fc(x))"""
# Testing the converter
tree = get_ast(torch_code)
print(tree.pretty())

output = py_to_solidity(torch_code)
print(output)
