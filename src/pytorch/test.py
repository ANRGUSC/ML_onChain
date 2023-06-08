from lark import Lark, Transformer, v_args
import astpretty

# Define the grammar
grammar = """
    start: classdef
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: stmt+ 
    stmt: funcdef 
        | assign
    funcdef: constructor
        | forward
    constructor: "def __init__(" parameters ")" ":" suite -> constructor
    forward: "def forward(" parameters ")" ":" suite -> predict
    parameters: CNAME "," CNAME 
    
    

    assign: ALL
    CNAME: /[a-zA-Z_][a-zA-Z_0-9]*/
    ALL: /[^\\n]+/
    %import common.WS
    %ignore WS
"""

# Define the transformer

# cassign: "self" "." CNAME "=" linear
# return: "return torch.sign(self.fc(x))" -> sign
# linear: "nn.Linear(" input_dim ", " NUMBER ")" -> linear


@v_args(inline=True)
class SolidityTransformer(Transformer):
    def contract(name, module):
        return f'contract {name} okk {module} {{}}'

    def constructor(self, input_dim):
        return f'constructor(int input_dim) {{}}'

    def predict(self, x):
        return f'predicting{{}}'


    # def add(self, a, b):
    #     return f'({a} + {b})'

    # def mul(self, a, b):
    #     return f'({a} * {b})'

    # def number(self, n):
    #     return str(n)
    
    # def linear(self, i, o):
    #     pass



# Create the parser
parser = Lark(grammar, parser='lalr')


def py_to_solidity(expr):
    return parser.parse(expr)


torch_code = """
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sign(self.fc(x))"""
# Testing the converter
tree = py_to_solidity(torch_code)
print(tree.pretty())
