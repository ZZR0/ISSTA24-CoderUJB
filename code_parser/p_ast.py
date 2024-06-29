from collections import deque
from io import StringIO
import re
import os
import tree_sitter_languages
from abc import abstractmethod
import tokenize

parsers = {}
for lang in ['java','python']:
    parsers[lang] = tree_sitter_languages.get_parser(lang)

class Node(object):
    def __init__(self, type, start_byte, end_byte, start_point, end_point, children=[]):
        self.type = type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point
        self.children = children

class P_AST(object):

    def __init__(self, root, code, cls, idx=0, parent=None, deep=0 ):
        self.type = root.type
        self.parent = parent
        if self.parent is not None:
            self.path = self.parent.path + "|" + self.type
        else:
            self.path = self.type

        self.deep = deep
        if self.type == "block":
            self.deep += 1
        
        # print(root)
        self.start_byte = root.start_byte
        self.end_byte = root.end_byte
        self.start_point = root.start_point
        self.end_point = root.end_point
        
        self.source = str(code[self.start_byte:self.end_byte], encoding="utf-8")
        self.source_line = "\n".join(str(code, encoding="utf-8").split("\n")[self.start_point[0]:self.end_point[0]+1])

        self.idx = idx
        
        if root.children:
            children = []
            prev_byte = root.children[0].start_byte
            prev_point = root.children[0].start_point
            for child in root.children:
                if child.start_byte != prev_byte:
                    children.append(Node(type="fill_in", 
                                         start_byte=prev_byte,
                                         end_byte=child.start_byte,
                                         start_point=prev_point,
                                         end_point=child.start_point,
                                         ))
                children.append(child)
                prev_byte = child.end_byte
                prev_point = child.end_point
                
            self.children = [cls(child, code, idx=i, parent=self, deep=self.deep) for i, child in
                            enumerate(children)]
            [child.set_idx(i) for i, child in enumerate(self.children)]
        else:
            self.children = []

    def modifly_source(self, source):
        self.source = source
        
    def delete_node(self):
        self.source = ""
        for child in self.children:
            child.delete_node()
    
    def link_ast(self):
        self.brother = self.parent.children if self.parent is not None else None
        self.left = None
        self.right = None
        if self.parent is not None:
            if self.idx > 0:
                self.left = self.parent.children[self.idx - 1]

            if self.idx < len(self.parent.children) - 1:
                self.right = self.parent.children[self.idx + 1]

        for child in self.children:
            child.link_ast()

    def print_ast(self, deep=0):
        print(" " * deep * 4 + self.type, self.start_point, self.end_point)
        for child in self.children:
            child.print_ast(deep + 1)

    def print_path_ast(self, deep=0):
        print("    " * deep + self.path, self.start_point, self.end_point)
        for child in self.children:
            child.print_path_ast(deep + 1)
    
    def convert_to_string(self):
        if self.children:
            code = "".join([child.convert_to_string() for child in self.children])
        else:
            code = self.source
        return code
    
    @staticmethod
    def check_is_fillin(node):
        return node.type == "fill_in"
    

    def get_functions(self):
        return self.bfs_search_all(self, lambda node: node.check_is_function(node))
    
    def get_with_comment_functions(self):
        def check_function_comment(node):
            if node.left is not None and node.left.check_is_comment():
                return True
            return False
        
        return self.bfs_search_all(self, lambda node: node.check_is_function(node) and check_function_comment(node))
    
    def get_function_comment(self):
        assert self.check_is_function(self), "not a function"
        comment = []
        while self.left is not None and (self.left.check_is_comment(self.left) or self.left.check_is_fillin(self.left)):
            if self.left.check_is_comment(self.left):
                comment.append(self.left.source_line)
            self = self.left
            
        return "\n".join(comment)
    
    def get_function_comment_nodes(self):
        assert self.check_is_function(self), "not a function"
        comments = []
        while self.left is not None and (self.left.check_is_comment(self.left) or self.left.check_is_fillin(self.left)):
            if self.left.check_is_comment(self.left):
                comments.append(self.left)
            self = self.left
        return comments
    
    def get_function_and_comment(self):
        assert self.check_is_function(self), "not a function"
        comment = self.get_function_comment()
        return comment + "\n" + self.source_line
    
    def get_function_with_name(self, name):
        for function in self.get_functions():
            if function.get_function_name() == name:
                return function
        return None
    
    def get_function_name(self):
        return self.bfs_search_one_source(self, self.check_is_function_name)
    
    @staticmethod
    def dfs_search_one(node, result_check, assert_check=None):
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = [node]
        while nodes:
            node = nodes.pop()
            if result_check(node):
                return node
            nodes.extend(node.children)
        return None
    
    @staticmethod
    def dfs_search_one_source(node, result_check, assert_check=None):
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = [node]
        while nodes:
            node = nodes.pop()
            if result_check(node):
                return node.source
            nodes.extend(node.children)
        return None

    @staticmethod
    def bfs_search_one(node, result_check, assert_check=None):
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = deque([node])
        while nodes:
            node = nodes.popleft()
            if result_check(node):
                return node
            nodes.extend(node.children)
        return None
    
    @staticmethod
    def bfs_search_one_source(node, result_check, assert_check=None):
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = deque([node])
        while nodes:
            node = nodes.popleft()
            if result_check(node):
                return node.source
            nodes.extend(node.children)
        return None
    
    @staticmethod
    def dfs_search_all(node, result_check, assert_check=None):
        results = []
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = [node]
        while nodes:
            node = nodes.pop()
            if result_check(node):
                results.append(node)
            nodes.extend(node.children)
        return results
    
    @staticmethod
    def bfs_search_all(node, result_check, assert_check=None):
        results = []
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = deque([node])
        while nodes:
            node = nodes.popleft()
            if result_check(node):
                results.append(node)
            nodes.extend(node.children)
        return results
    
    @staticmethod
    def bfs_search_all_source(node, result_check, assert_check=None):
        results = []
        if assert_check is not None:
            assert assert_check(node), "assert_check is not satisfied"
        nodes = deque([node])
        while nodes:
            node = nodes.popleft()
            if result_check(node):
                results.append(node.source)
            nodes.extend(node.children)
        return results
    
    @property
    def start_line(self):
        return self.start_point[0]
    
    @property
    def end_line(self):
        return self.end_point[0]
    
    @staticmethod
    def remove_comments_and_docstrings(source,lang):
        """
        Removes comments and docstrings from the given source code.

        Parameters:
            source (str): The source code from which to remove comments and docstrings.
            lang (str): The programming language of the source code.

        Returns:
            str: The source code with comments and docstrings removed, if applicable.
        """
        if lang in ['python']:
            """
            Returns 'source' minus comments and docstrings.
            """
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0
            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]
                ltext = tok[4]
                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))
                # Remove comments:
                if token_type == tokenize.COMMENT:
                    pass
                # This series of conditionals removes docstrings:
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string
                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line
            temp=[]
            for x in out.split('\n'):
                if x.strip()!="":
                    temp.append(x)
            return '\n'.join(temp)
        elif lang in ['ruby']:
            return source
        else:
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return " " # note: a space and not an empty string
                else:
                    return s
            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
            temp=[]
            for x in re.sub(pattern, replacer, source).split('\n'):
                if x.strip()!="":
                    temp.append(x)
            return '\n'.join(temp)

    def set_idx(self, idx):
        self.idx = idx

    @abstractmethod
    def get_packages(self):
        pass