import gspread
from torch import Tensor
import re
from typing import Any, Union

Cell = Union[str, int, float]

def ln(x: Tensor):
    # return pytorch ln function of x
    return x.log()


def get_tensor_sheet(values: Cell):
    equation_cells = {}
            
    def process_number(item: Cell, r: int, c: int):
        is_numeric = issubclass(type(item), (int, float))
        if is_numeric:
            # return Tensor([item]) requires_grad=True
            tensor = Tensor([item])
            tensor.requires_grad = True
            tensor.retain_grad()
            return tensor

        elif type(item) == str and item.startswith('='):
            equation_cells[(r, c)] = item
            return item

    tensors = [
        [process_number(item, r, c) for c, item in enumerate(row)]
        for r, row in enumerate(values)
    ]

    def update_formula(item: Cell, tensors: list[list[Tensor]], r: int, c: int):
        if not (type(item) == str and item.startswith('=')):
            return item

        expression = item[1:].lower()
        param_names = re.findall(r'[a-z]+\d+', expression) # find all cell names

        for cell_name in param_names:
            param_row, param_col = gspread.utils.a1_to_rowcol(cell_name)
            param_row -= 1
            param_col -= 1
            if (param_row, param_col) in equation_cells:
                return item
            
            expression = expression.replace(cell_name, f"tensors[{param_row}][{param_col}]")
        
        del equation_cells[(r, c)]

        try:
            expression = expression.replace('^', '**')
            tensor = eval(f"1*({expression})")
        except Exception as e:
            print(f"Error evaluating {expression}")
            raise e
            
        tensor.retain_grad()
        return tensor
            
    
    while len(equation_cells) > 0:
        for (r, c), formula in list(equation_cells.items()):
            tensors[r][c] = update_formula(formula, tensors, r, c)
    
    return tensors
    

def reset_grad(tensor_sheet: list[list[Tensor]]):
    for row in tensor_sheet:
        for item in row:
            if issubclass(type(item), Tensor):
                item.grad = None


def get_grads(tensor_sheet: list[list[Tensor]], cell_name: str, leaf_only=True):
    row_idx, col_idx = gspread.utils.a1_to_rowcol(cell_name)
    row_idx -= 1
    col_idx -= 1

    reset_grad(tensor_sheet)
    tensor_sheet[row_idx][col_idx].backward(retain_graph=True)

    grads = {}

    for r, row in enumerate(tensor_sheet):
        for c, item in enumerate(row):
            if issubclass(type(item), Tensor):
                if item.grad is not None:
                    if leaf_only and item.grad_fn is not None:
                        continue
                    cell_name = gspread.utils.rowcol_to_a1(r+1, c+1)
                    grads[cell_name] = item.grad.item()
    
    return grads



def percent_grads(grads: list[list[Tensor]], tensor_sheet: list[list[Tensor]], target_cell_name: str):

    target_r, target_c = gspread.utils.a1_to_rowcol(target_cell_name)
    target_r -= 1
    target_c -= 1
    target_tensor = tensor_sheet[target_r][target_c]

    # normalize grads so they represent how much a percent change in the input would change the output
    p_grads = {}
    for cell_name, grad in grads.items():
        row_idx, col_idx = gspread.utils.a1_to_rowcol(cell_name)
        row_idx -= 1
        col_idx -= 1
        tensor = tensor_sheet[row_idx][col_idx]
        p_grads[cell_name] = (grad * tensor.item() / 100) / target_tensor.item()

    return p_grads





