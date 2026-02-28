# this py file is to implement a easy torch-engine
# this engine follows the design principle of torch 1.x but much simpler, which I call as v1
# [architecture] 
# 1. Variable class, the core class for every tensor, implements OO.
#   1.1 OO, the operator implements 1. apply the forward operator 2. record everything into the tape variable
# 2. Tape class, the core class for recording the forward operator and backward operator
from typing import List, Callable
from dataclasses import dataclass
import numpy as np

id_counter = 0

def id_generator():
    global id_counter
    id_counter += 1
    return f"v_{id_counter}"

# the Variable class is the core class for tensor, implements OO
# whenever two tensor ops, a conjugate node, tape, is created to record status
class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or id_generator()
        # print(f"Creating Variable(name={self.name}, value={value})")

    def __repr__(self):
        return f"Variable(name={self.name}, value={self.value})"
    
    def __add__(self, other):
        return add_op(self, other)

    def __mul__(self, other):
        return mul_op(self, other)

    def __sub__(self, other):
        return sub_op(self, other)

    def sin(self):
        return sin_op(self)

    def log(self):
        return log_op(self)
    
    @staticmethod
    def verbose_init(value, name=None):
        v = Variable(value, name)
        return v

# tape record input and output of the op, and the backward operator, propagate
@dataclass
class Tape:
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : Callable[[List[Variable]], List[Variable]]

Tape_list: List[Tape] = []

def reset_tape():
    global id_counter
    id_counter = 0
    global Tape_list
    Tape_list = []

def add_op(self, other):
    # forward operator
    # print(f"forward add: {self.name} and {other.name}")
    x = Variable.verbose_init(self.value + other.value)

    def add_backward(dl_doutputs):
        # print(f"backward add: {self.name} and {other.name}")
        dl_dx, = dl_doutputs
        dx_dself = Variable(1.)
        dx_dother = Variable(1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # record the backward operator and relevant inputs and outputs for later use
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=add_backward)
    Tape_list.append(tape)
    # print("Pushing tape to Tape_list:", tape)
    return x


def mul_op(self, other):
    # forward operator
    # print(f"forward mul: {self.name} and {other.name}")
    x = Variable.verbose_init(self.value * other.value)
    
    def mul_backward(dl_doutputs):
        # print(f"backward mul: {self.name} and {other.name}")
        dl_dx, = dl_doutputs
        dx_dself = other
        dx_dother = self
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # record the backward operator and relevant inputs and outputs for later use
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=mul_backward)
    Tape_list.append(tape)
    # print("Pushing tape to Tape_list:", tape)
    return x

def sub_op(self, other):
    # forward operator
    # print(f"forward sub: {self.name} and {other.name}")
    x = Variable.verbose_init(self.value - other.value)
    
    def sub_backward(dl_doutputs):
        # print(f"backward sub: {self.name} and {other.name}")
        dl_dx, = dl_doutputs
        dx_dself = Variable(1.)
        dx_dother = Variable(-1.)
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs
    # record the backward operator and relevant inputs and outputs for later use
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=sub_backward)
    Tape_list.append(tape)
    # print("Pushing tape to Tape_list:", tape)
    return x

def sin_op(self):
    # forward operator
    # print(f"forward sin: {self.name}")
    x = Variable.verbose_init(np.sin(self.value))
    
    def sin_backward(dl_doutputs):
        # print(f"backward sin: {self.name}")
        dl_dx, = dl_doutputs
        dx_dself = Variable(np.cos(self.value))
        dl_dself = dl_dx * dx_dself
        return [dl_dself]
    # record the backward operator and relevant inputs and outputs for later use
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=sin_backward)
    Tape_list.append(tape)
    # print("Pushing tape to Tape_list:", tape)
    return x

def log_op(self):
    # forward operator
    # print(f"forward log: {self.name}")
    x = Variable.verbose_init(np.log(self.value))
    
    def log_backward(dl_doutputs):
        # print(f"backward log: {self.name}")
        dl_dx, = dl_doutputs
        dx_dself = Variable(1 / self.value)
        dl_dself = dl_dx * dx_dself
        return [dl_dself]
    # record the backward operator and relevant inputs and outputs for later use    
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=log_backward)
    Tape_list.append(tape)
    # print("Pushing tape to Tape_list:", tape)
    return x

# core backward function
def grad(l: Variable, results: List):
    dl_d = {}
    dl_d[l.name] = Variable(1.0)
    for tabe in reversed(Tape_list):
        print(tabe)
        grad_outputs = [dl_d[output] for output in tabe.outputs]
        grad_inputs = tabe.propagate(grad_outputs)
        for input, grad in zip(tabe.inputs, grad_inputs): 
            if input in dl_d:
                dl_d[input] += grad
            else:
                dl_d[input] = grad
    for name, value in dl_d.items():
        print(f'd{l.name}_d{name} = {value.value}')
    return None
        


if __name__ == "__main__":
    lst = [1, 2, 3]
    for x in reversed(lst):
        print(x)
        if x == 2:
            lst.append(99)

    print("final:", lst)
    x = Variable.verbose_init(2.)
    y = Variable.verbose_init(5.)
    f = Variable.log(x) + x * y - Variable.sin(y)
    grad(f, [x, y])