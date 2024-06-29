from code_parser.java_ast import JAVA_AST
from code_parser.python_ast import PYTHON_AST

def get_ast(lang):
    if lang == "java":
        return JAVA_AST
    elif lang == "python":
        return PYTHON_AST
    else:
        raise Exception(f"Language {lang} is not supported")
    
class Code_AST():
    def __init__(self, code, lang) -> None:
        self.code = code
        self.lang = lang
        self.ast = get_ast(lang).build_ast(code)