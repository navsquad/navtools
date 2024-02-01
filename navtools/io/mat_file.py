'''
|========================================= mat_file.py ============================================|
|                                                                                                  |
|  Property of NAVSQUAD (UwU). Unauthorized copying of this file via any medium would be super     |
|  sad and unfortunate for us. Proprietary and confidential.                                       |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navtools/io/mat_file.py                                                               |
|  @brief    MATLAB matfile reader and writer.                                                     |
|  @ref                                                                                            |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     January 2024                                                                          |
|                                                                                                  |
|==================================================================================================|
'''

import scipy.io as sio
from .common import ensure_exist

# === savemat ===
# write dictionary to mat file
def savemat(filename, data):
  ensure_exist(os.path.dirname(os.path.realpath(filename)))
  sio.savemat(filename, data)

# === loadmat ===
# ensures scipy.io.loadmat matfile is loaded into dictionary
def loadmat(filename):
  data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
  return _check_keys(data)

# === _check_keys ===
# converts keys that are 'mat-objects' into nested dictionaries
def _check_keys(d):
  for key in d:
    if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
      d[key] = _todict(d[key])
    elif isinstance(d[key], np.ndarray) and d[key].dtype == object:
      for i in np.arange(d[key].size):
        d[key][i] = _todict(d[key][i])
  return d        

# === _todict ===
# recursive function which constructs from 'mat-objects' nested dictionaries
def _todict(matobj):
  dict = {}
  for strg in matobj._fieldnames:
    elem = matobj.__dict__[strg]
    if isinstance(elem, sio.matlab.mio5_params.mat_struct):
      dict[strg] = _todict(elem)
    else:
      dict[strg] = elem
  return dict