from code_parser.p_ast import P_AST, parsers

class PYTHON_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0):
        super().__init__(root, code, PYTHON_AST, idx, parent, deep)

    @classmethod
    def build_ast(cls, code, lang="python"):
        the_code = bytes(code, 'utf8')
        node = parsers[lang].parse(the_code)
        the_ast = PYTHON_AST(node.root_node, the_code)
        the_ast.link_ast()
        return the_ast
    
    def check_is_function_name(self):
        if self.path.endswith("function_definition|identifier"):
            return True
        return False
    
    def check_is_function(self):
        if self.path.endswith("function_definition"):
            return True
        return False
    
    def check_is_comment(self):
        if self.path.endswith("comment"):
            return True
        return False
    
    def get_with_comment_functions(self):
        def check_function_comment(node):
            nodes = [self]
            while nodes:
                node = nodes.pop()
                if node.path.endswith("function_definition|block|expression_statement|string"):
                    return True
                nodes += node.children
            return False
        
        functions = []
        if self.path.endswith("function_definition") \
            and check_function_comment(self):
            functions += [self]
            
        for child in self.children:
            functions += child.get_with_comment_functions()

        return functions
    
    def get_function_comment(self):
        assert self.check_is_function(), "not a function"
        nodes = [self]
        comment = ""
        while nodes:
            node = nodes.pop()
            if node.path.endswith("function_definition|block|expression_statement|string"):
                comment = node.source_line
                break
            nodes += node.children
        return comment
    
    def get_function_and_comment(self):
        assert self.check_is_function(), "not a function"
        return self.source_line

if __name__ == "__main__":
    code = \
"""
from p_ast import *


class PYTHON_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0):
        super().__init__(root, code, PYTHON_AST, idx, parent, deep)

    @classmethod
    def build_ast(cls, code, lang="python"):
        the_code = bytes(code, 'utf8')
        node = parsers[lang].parse(the_code)
        the_ast = PYTHON_AST(node.root_node, the_code)
        the_ast.link_ast()
        return the_ast

    def get_functions(self):
        \"\"\"
        这个函数接收两个数作为参数，并返回它们的和。

        Args:
            a (int): 第一个加数
            b (int): 第二个加数

        Returns:
            int: 两个参数的和
        \"\"\"
        functions = []
        if self.path.endswith("function_definition"):
            functions += [self]

        for child in self.children:
            functions += child.get_functions()

        return functions
"""

    the_ast = PYTHON_AST.build_ast(code, lang="python")
    the_ast.print_path_ast()
    
    functions = the_ast.get_functions()
    for function in functions:
        print(function.get_function_name())
        print(function.source)
