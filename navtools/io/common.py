"""
|=========================================== common.py ============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/io/common.py                                                                 |
|  @brief    Common io functions.                                                                  |
|  @ref                                                                                            |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["ensure_exist"]

import os


# === ENSURE_EXIST ===
# make sure directory chosen exists
def ensure_exist(path):
    os.makedirs(os.path.realpath(path), exist_ok=True)
