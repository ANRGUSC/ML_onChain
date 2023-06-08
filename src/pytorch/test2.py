from lark import Lark, Transformer, v_args
import astpretty

# Define the grammar
# grammar = """
#     start: classdef -> concat
#     classdef: "class" CNAME "(" ALL suite -> contract
#     suite: stmt+ -> suite
#     stmt: funcdef -> stmt
#     funcdef: (constructor | forward) -> stmt

    
#     constructor: "def __init__(" parameters ")" ":" -> constructor
#     forward: "def forward(" parameters ")" ":" -> predict
#     parameters: CNAME "," CNAME -> parameters
    
#     assign: (cassign | fassign) -> stmt
#     cassign: "self" "." CNAME "=" expr -> cassign
#     fassign: CNAME "=" ALL -> fassign

#     expr: ALL -> expr
#     return: "return" expr -> ret_val
#     super: "super(" + ALL -> super
#     CNAME: /[a-zA-Z_][a-zA-Z_0-9]*/
#     ALL: /[^\\n]+/
#     %import common.WS
#     %ignore WS
# """

grammar = """
    start: classdef -> concat
    classdef: "class" CNAME "(" ALL suite -> contract
    suite: stmt+ -> suite
    stmt: (funcdef | assign | return) -> stmt
    funcdef: (constructor | forward) -> stmt

    constructor: "def __init__(" parameters ")" ":" -> constructor
    forward: "def forward(" parameters ")" ":" -> predict
    parameters: CNAME "," CNAME -> parameters
    assign: (cassign | fassign) -> stmt
    cassign: "self" "." CNAME "=" expr -> cassign
    fassign: CNAME "=" expr -> fassign

    expr: ALL -> expr
    return: "return " expr -> ret_val
    super: "super(" + ALL -> super
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
    def concat(self, *args):
        return ''.join(args)

    def suite(self, *args):
        return ''.join(args)

    def stmt(self, funcdef):
        return str(funcdef)

    def contract(self, name, inherit, func2):
        return f'contract {name} {{ \n {func2} }}\n'

    def constructor(self, params):
        return f'constructor({params}) {{\n}}\n'

    def predict(self, params):
        return f'predicting({params}) {{\n}}\n'

    def parameters(self, x, y):
        return f'{x}, {y}'
    
    def literal(self, x):
        return str(x)

    def cassign(self, x, y):
        return f'CAssign'

    def fassign(self, x, y):
        return f'FAssign'

    def expr(self, x):
        return f'Expression'
    
    def ret_val(self, x):
        return f'return'
    
    def super(self, x):
        return f'super'


# Create the parser



def get_ast(expr):
    parser = Lark(grammar, parser='earley')
    return parser.parse(expr)

def py_to_solidity(expr):
    parser = Lark(grammar, parser='earley')
    tree = parser.parse(expr)
    transformer = SolidityTransformer()
    return transformer.transform(tree).strip()


# "super(Perceptron, self).__init__()"

# torch_code = """
# class Perceptron(nn.Module):
#     def forward(self, x):
#     def __init__(self, input_dim):

#     """

torch_code = """
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sign(self.fc(x))"""
# Testing the converter
tree = get_ast(torch_code)
print(tree.pretty())

output = py_to_solidity(torch_code)
print(output)
