import numpy as np

def gauss_map(size_x, size_y=None, sigma_x=1, sigma_y=None):    
    if size_y == None:
        size_y = size_x
    if sigma_y == None:
        sigma_y = sigma_x
    
    assert isinstance(size_x, int)
    assert isinstance(size_y, int)
    
    x0 = size_x // 2
    y0 = size_y // 2
    
    x = np.arange(0, size_x, dtype=float)
    y = np.arange(0, size_y, dtype=float)[:,np.newaxis]
    
    x -= x0
    y -= y0
    
    exp_part = x**2/(2*sigma_x**2)+ y**2/(2*sigma_y**2)
    return 1/(2*np.pi*sigma_x*sigma_y) * np.exp(-exp_part)