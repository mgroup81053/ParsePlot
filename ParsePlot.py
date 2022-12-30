from __future__ import annotations
from math import nan, inf
from typing import Iterable, Callable
import matplotlib.pyplot as plt
import numpy as np
from numpy import float64
from math import sin, cos, tan, pi, factorial
from warnings import warn
from multimethod import multimethod

from MTree import Node, treeize, null_node



def equal(a, b):
    return abs(a-b) <= Setting.equal_epsilon

def csc(x): return 1/sin(x)
def sec(x): return 1/cos(x)
def cot(x): return 1/tan(x)

def collatz_sequence(n):
    while True:
        if n%2:
            n = 3*n+1
            yield n
        else:
            n //= 2
            yield n
def collatz_decay_time(n):
    nth_collatz_sequence = collatz_sequence(n)
    for i, object in enumerate(nth_collatz_sequence, start=1):
        if object < 3*n//2:
            return i


constant = int | float | complex
list_constant = list[int] | list[float] | list[complex]

class UnknownOperator(Exception):
    def __init__(self, oper):
        super().__init__(f"Unknown operator: {oper}")

class Setting:
    equal_epsilon = 1e-08 # assert a==b if abs(a-b) <= equal_epsilon

    use_implied_multiplication = True # if a or b is not const: ab = a*b = ab #FIXME: raise error if this is false and input is not valid
    identitive_unary_plus = True # +a = a
    floating_sign_in_multiplication = True # (+a)*b = a*(+b) = +(a*b), (-a)*b = a*(-b) = -(a*b)
    identitive_double_sign = True #  +(+a) = a, -(-a) = a
    naive_zeroth_power = True # a^0 = x^0 = tan(x)^0 = 0^0 = 1
    use_exponential_identity = True # a^1 = a
    naive_zero_multiplication = True # 0*a = x*0 = tan(x)*0 = 0
    naive_self_division = True # a/a = 0/0 = 1
    use_multiplicative_identity = True # a*1 = 1*a = a
    use_divisive_right_identity = True # a/1 = a
    sign_contribution_in_add_and_sub = True # a+(+b) = a+b, a-(+b) = a-b, a+(-b) = a-b, a-(-b) = a+b
    use_additive_identity = True # a+0 = 0+a = a
    use_subtractive_right_identity = True # a-0 = a

class Vector:
    @staticmethod
    def add(u, v):
        u = [0]*(len(v)-len(u)) + u
        v = [0]*(len(u)-len(v)) + v
        return [u_i + v_i for u_i, v_i in zip(u, v)]

    @staticmethod
    def sub(u, v):
        u = [0]*(len(v)-len(u)) + u
        v = [0]*(len(u)-len(v)) + v
        return [u_i - v_i for u_i, v_i in zip(u, v)]

    @staticmethod
    def scale(first_arg, second_arg):
        if isinstance(first_arg, constant) and isinstance(second_arg, list):
            c = first_arg
            u = second_arg
        elif isinstance(first_arg, list) and isinstance(second_arg, constant):
            c = second_arg
            u = first_arg
        else:
            raise Exception("Invalid argument(s)")

        return [u_i*c for u_i in u]

    @staticmethod
    def conv(u, v) -> list_constant:
        # out[n] = sum (i+j=n) (u[i]*v[j])
        return [sum([u[i]*v[n-i] for i in range(n + 1) if i < len(u) and n-i < len(v)])
            for n in range(len(u)-1 + len(v)-1 + 1)]


class Function:
    @multimethod
    def __init__(f, string: str): #type: ignore
        f.adam_node = treeize(string)
        f.after_init()

    @multimethod
    def __init__(f, node: Node): #type: ignore
        f.adam_node = node
        f.after_init()

    def remove_trivial_operation(f): #type: ignore
        '''replace additional operation depending on Setting'''

        if Setting.identitive_unary_plus:
            f.adam_node.replace_matching_case(lambda node:
                Node(node.right.data, node.right.left, node.right.right, node.parent, node.right.type)
                if node.type == "unary_oper" and node.data == "+" else null_node)

        if Setting.floating_sign_in_multiplication:
            # (+a)*b = +(a*b)
            f.adam_node.replace_matching_case(lambda node:
                Node("+", null_node, (a_times_b:=Node("*", node.left.right, node.right, null_node, "binary_oper")), node.parent, "unary_oper")
                    if node.type == "binary_oper" and node.data == "*"\
                        and node.left.type == "unary_oper" and node.left.data == "+"
                    else null_node)

            # a*(+b) = +(a*b)
            f.adam_node.replace_matching_case(lambda node:
                Node("+", null_node, (a_times_b:=Node("*", node.left, node.right.right, null_node, "binary_oper")), node.parent, "unary_oper")
                    if node.type == "binary_oper" and node.data == "*"\
                        and node.right.type == "unary_oper" and node.right.data == "+"
                    else null_node)

            # (-a)*b = -(a*b)
            f.adam_node.replace_matching_case(lambda node:
                Node("-", null_node, (a_times_b:=Node("*", node.left.right, node.right, null_node, "binary_oper")), node.parent, "unary_oper")
                    if node.type == "binary_oper" and node.data == "*"\
                        and node.left.type == "unary_oper" and node.left.data == "-"
                    else null_node)

            # a*(-b) = -(a*b)
            f.adam_node.replace_matching_case(lambda node:
                Node("-", null_node, (a_times_b:=Node("*", node.left, node.right.right, null_node, "binary_oper")), node.parent, "unary_oper")
                    if node.type == "binary_oper" and node.data == "*"\
                        and node.right.type == "unary_oper" and node.right.data == "-"
                    else null_node)

        if Setting.identitive_double_sign:
            # +(+a) = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.right.right.data, node.right.right.left, node.right.right.right, node.parent, node.right.right.type)
                if node.right != null_node and node.type == node.right.type == "unary_oper" and node.data == node.right.data == "+" else null_node)

            # -(-a) = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.right.right.data, node.right.right.left, node.right.right.right, node.parent, node.right.right.type)
                if node.right != null_node and node.type == node.right.type == "unary_oper" and node.data == node.right.data == "-" else null_node)

        if Setting.naive_zeroth_power:
            f.adam_node.replace_matching_case(lambda node:
                Node("1", null_node, null_node, node.parent, type="const")
                if node.type == "binary_oper" and node.data == "^" and equal(numberize_str(node.right.data), 0) else null_node)

        if Setting.use_exponential_identity:
            f.adam_node.replace_matching_case(lambda node:
                Node(node.left.data, node.left.left, node.left.right, node.parent, node.left.type)
                if node.type == "binary_oper" and node.data == "^" and equal(numberize_str(node.right.data), 1) else null_node)

        if Setting.naive_zero_multiplication:
            f.adam_node.replace_matching_case(lambda node:
                Node("0", null_node, null_node, node.parent, type="const")
                if node.type == "binary_oper" and node.data == "*" and
                    (equal(numberize_str(node.left.data), 0) or equal(numberize_str(node.right.data), 0)) else null_node)

        #FIXME: (a/a)*1 -> 1*1 -> 1, (a*1)/a -> (a*1)/a -> a/a
        if Setting.naive_self_division:
            ...

        if Setting.use_multiplicative_identity:
            # 1*a = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.right.data, node.right.left, node.right.right, node.parent, node.right.type)
                if node.type == "binary_oper" and node.data == "*" and equal(numberize_str(node.left.data), 1) else null_node)

            # a*1 = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.left.data, node.left.left, node.left.right, node.parent, node.left.type)
                if node.type == "binary_oper" and node.data == "*" and equal(numberize_str(node.right.data), 1) else null_node)

        if Setting.sign_contribution_in_add_and_sub:
            # a+(+b) = a+b
            f.adam_node.replace_matching_case(lambda node:
                Node("+", node.left, node.right.right, node.parent, "binary_oper")
                if node.right != null_node and node.type == "binary_oper" and node.right.type == "unary_oper" and node.data == "+" and node.right.data == "+" else null_node)

            # a-(+b) = a-b
            f.adam_node.replace_matching_case(lambda node:
                Node("-", node.left, node.right.right, node.parent, "binary_oper")
                if node.right != null_node and node.type == "binary_oper" and node.right.type == "unary_oper" and node.data == "-" and node.right.data == "+" else null_node)

            # a+(-b) = a-b
            f.adam_node.replace_matching_case(lambda node:
                Node("-", node.left, node.right.right, node.parent, "binary_oper")
                if node.right != null_node and node.type == "binary_oper" and node.right.type == "unary_oper" and node.data == "+" and node.right.data == "-" else null_node)

            # a-(-b) = a+b
            f.adam_node.replace_matching_case(lambda node:
                Node("+", node.left, node.right.right, node.parent, "binary_oper")
                if node.right != null_node and node.type == "binary_oper" and node.right.type == "unary_oper" and node.data == "-" and node.right.data == "-" else null_node)

        if Setting.use_additive_identity:
            # 0+a = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.right.data, node.right.left, node.right.right, node.parent, node.right.type)
                if node.type == "binary_oper" and node.data == "+" and equal(numberize_str(node.left.data), 0) else null_node)

            # a+0 = a
            f.adam_node.replace_matching_case(lambda node:
                Node(node.left.data, node.left.left, node.left.right, node.parent, node.left.type)
                if node.type == "binary_oper" and node.data == "+" and equal(numberize_str(node.right.data), 0) else null_node)

        if Setting.use_subtractive_right_identity:
            f.adam_node.replace_matching_case(lambda node:
                Node(node.left.data, node.left.left, node.left.right, node.parent, node.left.type)
                if node.type == "binary_oper" and node.data == "-" and equal(numberize_str(node.right.data), 0) else null_node)


    def after_init(f): #type: ignore
        f.remove_trivial_operation()
        f.func_type = f._get_func_type()
        f._declare_attr_by_func_type()

    @staticmethod
    def is_this_type(f: Function):
        return False

    def _get_func_type(f): #type: ignore
        '''
        get `func_type` based on `f.adam_node`
        '''

        for cls in Function.__subclasses__():
            if cls.is_this_type(f):
                return cls

        return Function

    def _declare_attr_by_func_type(f): #type: ignore
        if f.func_type == Polynomial: # constants are valid Polynomial with degree=0
            coefL = Polynomial.get_coefL(f)
            f.degree = Polynomial.get_degree(coefL)
            f.coefL = coefL[:f.degree+1]
            f.adam_node = Polynomial.get_tree(f)
            f.remove_trivial_operation()
            pass

        elif f.func_type == Trigonometric:
            f.trig_type, f.x_amp, f.y_amp, f.x_shift, f.y_shift = Trigonometric.get_amp_and_shift(f)
            f.amplitude = f.y_amp



        f.zeroL = f.func_type.get_zeroL(f)
        f.realzeroL = filter_real(f.zeroL)

    @staticmethod
    def get_zeroL(f: Function) -> list_constant: #FIXME
        return []

    def plot(f, x_range=np.arange(-10, 10, 1/16)): #type: ignore
        rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
            "xtick.major.size" : 5, "ytick.major.size" : 5,}
        with plt.rc_context(rc):
            fig, ax = plt.subplots()

            ax.plot(x_range, [f(x) for x in x_range], label=repr(f)) #BUG: exclude x s.t. f(x) makes error
            ax.scatter(f.realzeroL, [f(x) for x in f.realzeroL])

            ax.spines['left'].set_position(('data', 0.0))
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position(('data', 0.0))
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            # make arrows
            ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                    transform=ax.get_yaxis_transform(), clip_on=False)
            ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                    transform=ax.get_xaxis_transform(), clip_on=False)

            plt.legend()
            plt.show()

    def deriv(f, xL: Iterable = []): #type: ignore
        if f.func_type == Polynomial:
            if f.degree == 0:
                return Function("0")
            else:
                deriv_coefL = [(f.degree-j)*coef for j, coef in enumerate(f.coefL[:-1])]
                c_0 = deriv_coefL[-1]
                return Function("+".join([f"{str(c_i)}x^{f.degree-1-i}" for i, c_i in enumerate(deriv_coefL[:-1]) if not equal(c_i, 0)] + ([str(c_0)] if not equal(c_0, 0) else [])))
        else: #FIXME: return Polynomial approx
            raise
            # if len(xL) < 2:
            #     raise Exception("Not big enough x range")

            # yL = [f(x) for x in xL]

            # f_prime = []
            # prev_x = xL[0]
            # prev_y = yL[0]
            # for x, y in zip(xL[1:], yL[1:]):
            #     dx = x - prev_x
            #     dy = y - prev_y
            #     f_prime.append(dy/dx)

            # f_prime.append(f_prime[-1])
            # return np.array(f_prime)

    def integ(f, C=0) -> Function: #type: ignore
        if f.func_type == Polynomial:
            if f.degree == 0 and equal(f.coefL[0], 0):
                return Function(str(C))
            else:
                integ_coefL = [coef/(f.degree-j+1) for j, coef in enumerate(f.coefL)]
                return Function("+".join([f"{str(c_i)}x^{f.degree+1-i}" for i, c_i in enumerate(integ_coefL) if not equal(c_i, 0)] + ([str(C)] if not equal(C, 0) else [])))
        else: #FIXME
            raise

    def taylor(f, xL, a=0, n=5) -> Function: #type: ignore
        if f.func_type == Polynomial:
            return f
        else: #FIXME: return polynomial
            raise
        #     yL = [f(x) for x in xL]
        #     out = np.array([0 for _ in range(len(f))], float64)
        #     a_index = np.where(xL == a)[0][0]
        #     for i in range(len(yL)):
        #         t = [yL[a_index]/factorial(i)*(x-a)**i for x in xL]
        #         out += np.array(t)
        #         yL = f.deriv(xL)

        #     return out


    def __call__(f, x): #type: ignore
        if f.func_type == Polynomial:
            summation = f.coefL[-1]
            for i, c_i in enumerate(f.coefL[-2::-1], start=1):
                summation += c_i * x**i
            return summation
        elif f.func_type == Trigonometric:
            return f.trig_type((x-f.x_shift)/f.x_amp)*f.y_amp + f.y_shift
        else:
            raise #FIXME

    def __repr__(f): #type: ignore
        return repr(f.adam_node)


















class Polynomial(Function):
    @staticmethod
    def get_degree(coefL: list_constant):
        degree = len(coefL)-1
        for c_i in coefL[:-1]: 
            if c_i:
                break
            else:
                degree -= 1

        return degree

    @staticmethod
    def get_tree(f: Function):
        const_term = f.coefL[-1]
        non_const_coef = f.coefL[:-1]
        return treeize("+".join([f"{str(c_i)}*x^{i}" for i, c_i in zip(range(f.degree, 1-1, -1), non_const_coef) if c_i] + [str(const_term)]))

    @staticmethod
    def get_coefL(f: Function):
        return Polynomial._get_coefL_by_Node(f.adam_node)

    @staticmethod
    def _get_coefL_by_Node(node: Node) -> list_constant:
        if node.type == "const":
            return [numberize_str(node.data)]
        elif node.type == "var/const":
            return [1, 0]
        elif node.type == "unary_oper":
            if node.data == "-":
                return Vector.scale(Polynomial._get_coefL_by_Node(node.right), -1)
            else:
                raise UnknownOperator(node.data)
        else: # elif node.type == "binary_oper":
            if node.data == "+":
                return Vector.add(Polynomial._get_coefL_by_Node(node.left),
                    Polynomial._get_coefL_by_Node(node.right))
            elif node.data == "-":
                return Vector.sub(Polynomial._get_coefL_by_Node(node.left),
                    Polynomial._get_coefL_by_Node(node.right))
            elif node.data == "*":
                return Vector.conv(Polynomial._get_coefL_by_Node(node.left),
                    Polynomial._get_coefL_by_Node(node.right))
            elif node.data == "/":
                return Vector.scale(Polynomial._get_coefL_by_Node(node.left), node.right.data)
            elif node.data == "^":
                if node.left.type == "const": # a^b
                    return [numberize_str(node.left.data) ** numberize_str(node.right.data)]
                else: # P(x)^n
                    _temp = [1]
                    _left_coefL = Polynomial._get_coefL_by_Node(node.left)
                    for _ in range(int(node.right.data)):
                        _temp = Vector.conv(_temp, _left_coefL)

                    return _temp
            else:
                raise UnknownOperator(node.data)


    @staticmethod
    def get_zeroL(f: Function):
        abundant_x_count = 0
        for coef in f.coefL[::-1]:
            if coef != 0:
                break
            else:
                abundant_x_count += 1

        if abundant_x_count:
            modified_coefL = f.coefL[:-abundant_x_count]
        else:
            modified_coefL = f.coefL

        match len(modified_coefL)-1:
            case 0:
                a, = modified_coefL
                if a == 0:
                    zeroL = [nan,]
                else:
                    zeroL = []
            case 1:
                a, b = modified_coefL
                zeroL = [-b/a,]

            case 2:
                a, b, c = modified_coefL
                D = b**2 - 4*a*c
                zeroL = [(-b+sgn*D**0.5)/(2*a) for sgn in ((-1)**i for i in range(2))]

            case 3:
                a, b, c, d = modified_coefL
                w = (-1 + (-3)**0.5)/2
                delta_0 = b**2 - 3*a*c
                delta_1 = 2*b**3 - 9*a*b*c + 27*a**2*d
                if equal(delta_0, 0) and equal(delta_1, 0):
                    zeroL = [-b/(3*a) for _ in range(3)]
                else:
                    if equal(delta_0, 0):
                        C = (delta_1)**(1/3)
                    else:
                        C = ((delta_1 + (delta_1**2 - 4*delta_0**3))/2)**(1/3)

                    zeroL = [-(b + sgn*C + delta_0/(sgn*C))/(3*a) for sgn in (w**i for i in range(3))]

            case 4:
                a, b, c, d, e = modified_coefL
                w = (-1 + (-3)**0.5)/2
                # delta = (delta_1**2 - 4*delta_0**3)/-27
                delta_0 = c**2 - 3*b*d + 12*a*e
                delta_1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*c*e
                p = (8*a*c - 3*b**2) / (8*a**2)
                q = (b**3 - 4*a*b*c + 8*a**2*d) / (8*a**3)
                if delta_0 == delta_1 == 0:
                    zeroL = [-b/(2*a), -b/(2*a), -b/(2*a), b/(2*a)]
                else:
                    if delta_0 == 0:
                        Q = delta_1**(1/3)
                    else:
                        Q = ((delta_1 + (delta_1**2 - 4*delta_0**3)**0.5)/2)**(1/3)

                        # if delta_1 == 0:
                        # Q = (-delta_0)**0.5

                    S = (-2/3*p + (Q + delta_0/Q)/(3*a))**0.5 / 2

                    if S == 0:
                        Q *= w
                        S = (-2/3*p + (Q + delta_0/Q)/(3*a))**0.5 / 2
                    # for float error
                    if abs(S) < 1e-08:
                        Q *= w
                        S = (-2/3*p + (Q + delta_0/Q)/(3*a))**0.5 / 2



                    zeroL = [-b/(4*a) - sgn1*S + sgn2*(-4*S**2 - 2*p + sgn1*q/S)**0.5/2 for sgn1, sgn2 in ((-1, -1), (-1, 1), (1, -1), (1, 1))]

            case _:
                zeroL = []
                warn("Roots unavailable")


        return [0]*abundant_x_count + zeroL




    @staticmethod
    def is_this_type(f: Function):
        return Polynomial._is_polynomial_node(f.adam_node)

    @staticmethod
    def _is_polynomial_node(node: Node) -> bool:
        if node.type in ("var/const", "const"):
            return True
        elif node.type == "unary_oper":
            if node.data in ("+", "-"):
                return Polynomial._is_polynomial_node(node.right)
            else:
                raise UnknownOperator(node.data)
        elif node.type == "binary_oper":
            if node.data in ("+", "-", "*"):
                return Polynomial._is_polynomial_node(node.left) \
                    and Polynomial._is_polynomial_node(node.right)
            elif node.data in ("/", "^"):
                return node.left.type == node.right.type == "const" \
                    or Polynomial._is_polynomial_node(node.left) \
                        and node.right.type == "const" and is_str_integer(node.right.data) and (int(node.right.data) >= 0 if Setting.naive_zeroth_power else int(node.right.data) > 0)
            else:
                raise UnknownOperator(node.data)
        else: # elif node.type == "func" #FIXME
            return False





















class Trigonometric(Function):
    @staticmethod
    def is_this_type(f: Function):
        return Trigonometric.is_trigonometric_node(f.adam_node)

    @staticmethod
    def is_trigonometric_node(node: Node) -> bool: #FIXME
        if node.type == "unary_oper":
            if node.data in ("+", "-"):
                return Trigonometric.is_trigonometric_node(node.right)
            else:
                raise UnknownOperator(node.data)
        elif node.type == "binary_oper":
            if node.data == "*":
                if "const" in (node.left.type, node.right.type):
                    questioning_node, const_node = sorted([node.left, node.right], key=lambda _node: _node.type=="const")
                    return Trigonometric.is_trigonometric_node(questioning_node)
                else:
                    return False #FIXME
            else:
                raise UnknownOperator(node.data)
        elif node.type == "func":
            if node.data == "sin":
                right_func = Function(node.right)
                return right_func.func_type == Polynomial and right_func.degree <= 1
            else:
                return False #FIXME
        else:
            raise Exception("Unknown type")

    @staticmethod
    def get_amp_and_shift(f: Function):
        return Trigonometric._get_amp_and_shift_by_Node(f.adam_node)

    @staticmethod
    def _get_amp_and_shift_by_Node(node: Node) -> tuple[Callable, constant, constant, constant, constant]:
        if node.type == "unary_oper":
            if node.data == "+":
                return Trigonometric._get_amp_and_shift_by_Node(node.right)
            elif node.data == "-":
                trig_type, x_amp, y_amp, x_shift, y_shift = Trigonometric._get_amp_and_shift_by_Node(node.right)
                return trig_type, x_amp, -y_amp, x_shift, -y_shift
            else:
                raise UnknownOperator(node.data)
        elif node.type == "binary_oper": #FIXME
            if node.data == "+":
                if "const" in (node.left.type, node.right.type):
                    trig_node, const_node = sorted([node.left, node.right], key = lambda _node: _node.type == "const")
                    trig_type, x_amp, y_amp, x_shift, y_shift = Trigonometric._get_amp_and_shift_by_Node(trig_node)
                    return trig_type, x_amp, y_amp, x_shift, y_shift+numberize_str(const_node.data)
                else:
                    raise
            elif node.data == "-":
                raise
            elif node.data == "*":
                if "const" in (node.left.type, node.right.type):
                    trig_node, const_node = sorted([node.left, node.right], key = lambda _node: _node.type == "const")
                    trig_type, x_amp, y_amp, x_shift, y_shift = Trigonometric._get_amp_and_shift_by_Node(trig_node)
                    return trig_type, x_amp, y_amp*numberize_str(const_node.data), x_shift, y_shift*numberize_str(const_node.data)
                else:
                    raise
            else:
                raise UnknownOperator(node.data)
        elif node.type == "func": #FIXME
            right_func = Function(node.right)
            if right_func.func_type == Polynomial and right_func.degree == 1:
                if node.data == "sin": # sin(ax+b)
                    a, b = Function(node.right).coefL
                    return sin, 1/a, 1, -b/a, 0
                else:
                    raise UnknownOperator(node.data)
            else:
                raise
        else:
            raise Exception("Unexpected argument")




def filter_real(zL: list_constant):
    return [z.real for z in zL if equal(z.imag, 0)]

def is_str_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def numberize_str(string: str):
    try:
        _out = complex(string)
    except ValueError:
        return nan
    else:
        if _out.imag == 0:
            _out = _out.real

            if _out.is_integer():
                _out = int(_out)

        return _out






#BUG: +-3x^2+-2x^3+-x   ->    +- don't disappear

if __name__ == "__main__":
    f = Function(input())
    # f.integ()
    f.plot()

