"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {    # 冒号后面:Dict[str, BaseSolver] 是类型注释，表示TASKS是一个字典，不过字典的 键是 字符串  值是一个类
    'detection': DetSolver,
}