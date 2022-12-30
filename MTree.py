from __future__ import annotations
from string import ascii_letters, digits
from re import finditer
from math import nan
from typing import Callable
from multimethod import multimethod




######################################

xmin = -10
# xmin = 0
xmax = +10
dx = .01
logscale = (False, False)

######################################




valid_charT = ("!", "%", "^", "+", "-", "*", "/", "<", "=", ">", "(", ")", ",", ".", "_", "'", "Δ" " ") + tuple(ascii_letters) + tuple(digits)

operation_charTD = { # precedence: parenthesis > function > unaryR > unaryL > binary1 > binary2 > binary3
    "unaryR": ("!", "%",),
    "unaryL": ("+", "-", ),
    "binary1": ("^"),
    "binary2": ("*", "/",),
    "binary3": ("+", "-",),
}

function_charT = ("sin", "cos", "tan", "ln", "log",) # precedence: parenthesis > function > unaryR > unaryL > binary1 > binary2 > binary3


class Node:
    @multimethod
    def __init__(self): #type: ignore
        self.kuarg = {"null": True}
        
    @multimethod
    def __init__(self, data: str, placeInText: float, left: Node, right: Node, parent: Node, type: str="", **kuarg): #type: ignore
        self.data=data; self.placeInText=placeInText; self.left=left; self.right=right; self.parent=parent; self.type=type
        self.kuarg=kuarg

    @multimethod
    def __init__(self, data: str, left: Node, right: Node, parent: Node, type: str="", placeInText: int | float = nan, **kuarg):
        self.data=data; self.placeInText=placeInText; self.parent=parent; self.type=type
        self.kuarg=kuarg
        self.putleft(left)
        self.putright(right)



    def putleft(self, left: Node):
        self.left=left
        if left != null_node:
            left._putparent(self)

    def putright(self, right: Node):
        self.right=right
        if right != null_node:
            right._putparent(self)

    def replace(self, other: Node):
        self.data=other.data; self.placeInText=other.placeInText; self.left=other.left; self.right=other.right; self.parent=other.parent; self.type=other.type
        self.kuarg=other.kuarg

    def _putparent(self, parent: Node):
        self.parent=parent

    def __lt__(self, other: Node): # sort by placeInTextself.input
        return self.placeInText < other.placeInText

    def __getattribute__(self, name):
        if name != "kuarg" and "null" in self.kuarg.keys() and self.kuarg["null"] == True:
            raise Exception("Tried to get attribute of null node")
        else:
            return object.__getattribute__(self, name)

    def __repr__(self):
        from ParsePlot import Setting

        repr_left = repr(self.left) if self.left != null_node else ""
        repr_right = repr(self.right) if self.right != null_node else ""

        if self.type == "const":
            return self.data
        elif self.type == "var/const":
            return self.data
        elif self.type == "func":
            return f"{self.data}({repr_right})"
        elif self.type in ("unary_oper", "binary_oper", "oper"):
            if Setting.use_implied_multiplication and self.type == "binary_oper" and self.data == "*" and null_node not in (self.left, self.right) and (self.left.type != "const" or self.right.type != "const"):
                return f"{repr_left}{repr_right}"
            else:
                return f"{repr_left}{self.data}{repr_right}"
        else:
            raise Exception("Unknown type")
        



    def replace_matching_case(self, replace_func: Callable[[Node], Node]):
        if replace_func(self) == null_node:
            [child.replace_matching_case(replace_func) for child in (self.left, self.right) if child != null_node]
        else:
            self.replace(replace_func(self))



null_node = Node()


def treeize(text: str):
    from ParsePlot import Setting

    nodeL: list[Node] = []

    # mark every single symbol
    # parenthesis
    for _index in finditer(r"\([^\(\)]*\)", text):
        _start = _index.regs[0][0]
        _end = _index.regs[0][1]
        subnode = treeize(text[_start+1:_end-1])
        subnode.placeInText = _start
        nodeL.append(subnode)
        text = text[:_start] + "`"*(_end - _start) + text[_end:] # replace detected place with `

    # function
    for each_func in function_charT:
        for _index in finditer(each_func, text):
            _start = _index.regs[0][0]
            _end = _index.regs[0][1]
            nodeL.append(Node(text[_start:_end], null_node, null_node, null_node, "func", _start))
            text = text[:_start] + "`"*(_end - _start) + text[_end:] # replace detected place with `

    # variable/constant
    for _index in finditer("[\u0394]?[a-zA-Z](_[a-zA-Z0-9]*)?'?", text):
        _start = _index.regs[0][0]
        _end = _index.regs[0][1]
        nodeL.append(Node(text[_start:_end], null_node, null_node, null_node, "var/const", _start))
        text = text[:_start] + "`"*(_end - _start) + text[_end:] # replace detected place with `

    # number
    for _index in finditer(r"\d*[.]?\d+", text):
        _start = _index.regs[0][0]
        _end = _index.regs[0][1]
        nodeL.append(Node(text[_start:_end], null_node, null_node, null_node, "const", _start))
        text = text[:_start] + "`"*(_end - _start) + text[_end:] # replace detected place with `

    # remove whitespaces
    for _i, _char in enumerate(text):
        if _char.isspace():
            text = text[:_i] + "`" + text[_i+1:]

    # append Nodes (symbol poped out)
    for _i, _char in enumerate(text):
        if _char != "`":
            nodeL.append(Node(_char, null_node, null_node, null_node, "oper", _i))






    nodeL.sort()

    if Setting.use_implied_multiplication:
        list_changed = True
        while list_changed:
            list_changed = False
            for node_index in range(len(nodeL)-1):
                each_node = nodeL[node_index]
                post_node = nodeL[node_index+1]
                if null_node not in (each_node, post_node)\
                    and (each_node.type not in ("oper", "func") or each_node.type == "func" and each_node.right != null_node)\
                    and post_node.type != "oper":

                    middle_place = (each_node.placeInText + post_node.placeInText)/2
                    nodeL.insert(node_index+1, Node("*", null_node, null_node, null_node, "oper", middle_place))
                    list_changed = True
                    break






    # gather around function
    for node_index in range(len(nodeL)):
        each_node = nodeL[node_index]
        if each_node != null_node and each_node.data in function_charT:
            if node_index != len(nodeL)-1:
                post_node = nodeL[node_index+1]
                each_node.putright(post_node)
                nodeL[node_index+1] = null_node
            else:
                raise Exception("function operator in the last place")

    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)



    # gather around unaryR operator
    for node_index in range(len(nodeL)):
        each_node = nodeL[node_index]
        if each_node != null_node and each_node.type == "oper" and each_node.data in operation_charTD["unaryR"]:
            if node_index != 0:
                pre_node = nodeL[node_index-1]
                each_node.putleft(pre_node)
                nodeL[node_index-1] = null_node
            else:
                raise Exception("unaryR operator in the first place")

            each_node.type = "unary_oper"

    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)



    # gather around unaryL operator
    for node_index, each_node in [(node_index, each_node) for node_index, each_node in enumerate(nodeL)
                                    if each_node != null_node and each_node.type == "oper" and each_node.data in operation_charTD["unaryL"] and (node_index == 0 or (pre_node:=nodeL[node_index-1]).type == "oper")]\
                                    [::-1]:
        if each_node != null_node:
            if node_index != len(nodeL)-1:
                post_node = nodeL[node_index+1]
                each_node.putright(post_node)
                nodeL[node_index+1] = null_node
            else:
                raise Exception("unaryL operator in the last place")

            each_node.type = "unary_oper"

    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)


        

    # gather around binary1 operator
    for node_index in range(len(nodeL)):
        each_node = nodeL[node_index]
        if each_node != null_node and each_node.data in operation_charTD["binary1"] and each_node.type == "oper":
            if node_index != 0 and node_index != len(nodeL)-1:
                for pre_node_index in range(node_index-1, -1, -1):
                    pre_node = nodeL[pre_node_index]
                    if pre_node != null_node: break
                else:
                    raise Exception("Missing pre_node for a binary operation")

                post_node = nodeL[node_index+1]
                each_node.putleft(pre_node)
                each_node.putright(post_node)
                nodeL[pre_node_index] = null_node
                nodeL[node_index+1] = null_node
            else:
                raise Exception("binary operator in the first or the last place")

            each_node.type = "binary_oper"

    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)




    # gather around binary2 operator
    for node_index in range(len(nodeL)):
        each_node = nodeL[node_index]
        if each_node != null_node and each_node.data in operation_charTD["binary2"] and each_node.type == "oper":
            if node_index != 0 and node_index != len(nodeL)-1:
                for pre_node_index in range(node_index-1, -1, -1):
                    pre_node = nodeL[pre_node_index]
                    if pre_node != null_node: break
                else:
                    raise Exception("Missing pre_node for a binary operation")

                post_node = nodeL[node_index+1]
                each_node.putleft(pre_node)
                each_node.putright(post_node)
                nodeL[pre_node_index] = null_node
                nodeL[node_index+1] = null_node
            else:
                raise Exception("binary operator in the first or the last place")

            each_node.type = "binary_oper"


    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)




    # gather around binary3 operator
    for node_index in range(len(nodeL)):
        each_node = nodeL[node_index]
        if each_node != null_node and each_node.data in operation_charTD["binary3"] and each_node.type == "oper":
            if node_index != 0 and node_index != len(nodeL)-1:
                for pre_node_index in range(node_index-1, -1, -1):
                    pre_node = nodeL[pre_node_index]
                    if pre_node != null_node: break
                else:
                    raise Exception("Missing pre_node for a binary operation")

                post_node = nodeL[node_index+1]
                each_node.putleft(pre_node)
                each_node.putright(post_node)
                nodeL[pre_node_index] = null_node
                nodeL[node_index+1] = null_node
            else:
                raise Exception("binary operator in the first or the last place")

            each_node.type = "binary_oper"


    for _ in range(nodeL.count(null_node)):
        nodeL.remove(null_node)

    adam_node = nodeL[0]

    return adam_node







