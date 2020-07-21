import pandas as pd
import javalang
import os
import re

#################################################

""" CLASS METRICS """

def get_java_files():
    java_files = []
    for root, dirs, files in os.walk('./defects4j/jscomp'):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.abspath(os.path.join(root, file)))
    return java_files


def get_class(java_class):
    path, file_name = os.path.split(java_class)
    opened = open(java_class, 'r').read()
    tree = javalang.parse.parse(opened)
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        if (node.name + ".java") == file_name:
            return node
        else:
            return None


def num_methods(node):
    class_methods = []
    class_methods_count = 0
    method_statements = []
    called_methods_count = 0
    method_conditionals = []
    method_loops = []
    method_throws = []
    method_returns = []
    method_blocks = []
    words_count = []
    for member in node.body:
        if type(member) == javalang.tree.MethodDeclaration:
            class_methods_count += 1
            class_methods.append(member.name)
            for path, node in member.filter(javalang.tree.MethodInvocation):
                called_methods_count += 1
            method_statements.append(num_statements(member))
            method_conditionals.append(num_conditional(member))
            method_loops.append(num_loop(member))
            method_throws.append(num_throws(member))
            method_returns.append(num_returns(member))
    method_statements = max(method_statements) if len(method_statements) != 0 else 0
    method_conditionals = max(method_conditionals) if len(method_conditionals) != 0 else 0
    method_loops = max(method_loops) if len(method_loops) != 0 else 0
    method_throws = max(method_throws) if len(method_throws) != 0 else 0
    method_returns = max(method_returns) if len(method_returns) != 0 else 0
    return class_methods_count, called_methods_count, method_statements, method_conditionals, method_loops, method_throws, method_returns, class_methods


def num_variables(node):
    variables_count = 0
    for member in node.body:
        if type(member) == javalang.tree.FieldDeclaration:
            variables_count += 1
    return variables_count


def num_public_methods(node):
    public_methods = 0
    for member in node.body:
        if (type(member) == javalang.tree.ConstructorDeclaration or type(member) == javalang.tree.MethodDeclaration):
            if 'public' in member.modifiers:
                public_methods += 1
    return public_methods


def num_interfaces(node):
    return 0 if node.implements == None else len(node.implements)

#################################################


""" METHOD METRICS """


def num_statements(method_node):
    statements_count = 0
    for path, member in method_node.filter(javalang.tree.Statement):
        if type(member) != javalang.tree.BlockStatement:
            statements_count += 1
    return statements_count


def num_conditional(method_node):
    conditionals = 0
    for path, member in method_node.filter(javalang.tree.Statement):
        if type(member) != javalang.tree.BlockStatement and type(member) in [javalang.tree.SwitchStatement, javalang.tree.IfStatement]:
            conditionals += 1
    return conditionals


def num_loop(method_node):
    loops = 0
    for path, member in method_node.filter(javalang.tree.Statement):
        if type(member) != javalang.tree.BlockStatement and type(member) in [javalang.tree.WhileStatement, javalang.tree.ForStatement]:
            loops += 1
    return loops


def num_throws(method_node):
    throws = 0
    for path, member in method_node.filter(javalang.tree.Statement):
        if type(member) != javalang.tree.BlockStatement and type(member) == javalang.tree.ThrowStatement:
            throws += 1
    return throws


def num_returns(method_node):
    returns = 0
    for path, member in method_node.filter(javalang.tree.Statement):
        if type(member) != javalang.tree.BlockStatement and type(member) == javalang.tree.ReturnStatement:
            returns += 1
    return returns

#################################################


""" NPL METRICS """


def num_comments_and_words(class_node):
    comments = 0
    words = 0
    for path, node in class_node.filter(javalang.tree.Documented):
        if (node.documentation is not None):
            comments += 1
            words += len(re.findall('\w+', node.documentation))
    return comments, words



def avg_length_method_names(class_methods_names):
    return 0 if len(class_methods_names) == 0 else round(sum(map(len, class_methods_names)) / len(class_methods_names), 2)


#################################################


def get_buggy_classes():
    buggy_classes=[]
    for path, dirs, files in os.walk('./defects4j/framework/projects/Closure/modified_classes'):
        for f in files:
            if f.endswith(".src"):
                content=open(os.path.join(path, f), 'r').read()
                buggy_class_path=content.rstrip('\n').split('.')
                buggy_class=buggy_class_path[len(buggy_class_path)-1]
                buggy_classes.append(buggy_class)
    return buggy_classes


def is_buggy_class(class_node):
    buggy_classes = get_buggy_classes()
    if class_node.name in buggy_classes:
        return 1
    else:
        return 0

#################################################


files = get_java_files()
file_metrics = dict()

for file in files:
    class_node=get_class(file)
    if class_node != None:
        metrics=[]
        path, file_name = os.path.split(file)
        num_class_methods, called_methods, num_of_statements, num_of_conditionals, num_of_loops, num_of_throws, num_of_returns, class_methods_names =num_methods(
            class_node)
        metrics.append(num_class_methods)  # Methods
        metrics.append(num_variables(class_node))  # Fields
        # Public methods + #Called methods 
        metrics.append(num_public_methods(class_node) + called_methods)
        metrics.append(num_interfaces(class_node))  # Implemented interfaces
        metrics.append(num_of_statements)  # Statements 
        # CONDITIONAL + # LOOP statements 
        metrics.append(num_of_conditionals + num_of_loops)
        metrics.append(num_of_throws)  # Exceptions in throws clause 
        metrics.append(num_of_returns)  # Return points
        bcm, wrd = num_comments_and_words(class_node)
        metrics.append(bcm)  # Block comments
        metrics.append(avg_length_method_names(class_methods_names))  # Average lenght of method names
        metrics.append(wrd)  # Words (longest alphanumeric substrings) in block comments) 
        metrics.append(0 if num_of_statements == 0 else round(wrd / num_of_statements, 4)) # Words in comments / # Statements
        metrics.append(is_buggy_class(class_node))
        file_metrics[file_name[:-5]]=metrics

feature_vector = pd.DataFrame.from_dict(file_metrics, orient='index', columns=['MTH', 'FLD', 'RFC', 'INT', 'SZ', 'CPX', 'EX', 'RET', 'BCM', 'NML', 'WRD', 'DCM', 'buggy'])
feature_vector.index.set_names('class', inplace=True)
feature_vector = feature_vector.sort_values(by = ['class'], ascending = [True]).reset_index()
feature_vector.to_csv("./labeled-feature-vectors.csv")

