import numpy as np
from matplotlib import pyplot as plt


class SquigglyBase:
  def __init__(self, x, y):
    assert ((len(x) > 3) and (len(y) > 3)), "Not enough x/y points!"
    assert (len(x) == len(y)), "len(x) != len(y)"
    self.x = np.array(x)
    self.y = np.array(y)

  def get_bounds(self, x):
    return min(x), max(x), max(x) - min(x)

  def squigglify(self, x, y, dx_perc=0.1, noise_strength=0.5, autocorr_perc=5):
    xmin, xmax, xrange = self.get_bounds(x)
    dx = xrange * dx_perc / 100
    new_x = np.arange(xmin, xmax + dx, dx)
    new_y = np.interp(new_x, x, y)
    noise = np.random.normal(0, noise_strength, len(new_y))
    new_y = new_y + noise

    autocorr_window = int(len(new_x) * autocorr_perc / 100)
    smoother = np.ones(autocorr_window)
    new_y = np.convolve(new_y, smoother, 'same')
    new_y_denom = np.ones_like(new_y) * autocorr_window
    new_y_denom[:autocorr_window // 2] = np.arange(autocorr_window // 2, autocorr_window)
    new_y_denom[-autocorr_window // 2:] = np.arange(autocorr_window // 2, autocorr_window)[::-1] + 1
    return new_x, new_y / new_y_denom

  def get_gridlines(self, x_bounds, y_bounds, nticks=5, extend_perc=0.03, grid_dir='x', dx_perc=0.1):
    if grid_dir == 'x':
      a, b = x_bounds, y_bounds
    elif grid_dir == 'y':
      b, a = x_bounds, y_bounds
    else:
      raise Exception("Not valid grid direction")

    new_a_bounds = (a[0] - a[2] * extend_perc, a[1] + a[2] * extend_perc)
    grid_pts_b = np.linspace(b[0], b[1], nticks)
    da = (new_a_bounds[1] - new_a_bounds[0]) * dx_perc / 100

    grid = []
    a_axis = np.arange(new_a_bounds[0], new_a_bounds[1] + da, da)
    for pt in grid_pts_b:
      grid.append((a_axis, np.ones_like(a_axis) * pt, pt))
    return grid


class SquigglyLine(SquigglyBase):
  def plot(self):
    sqx, sqy = self.squigglify(self.x, self.y)

    plt.figure(figsize=(20, 10))
    plt.plot(sqx, sqy, alpha=0.5, linewidth=3)

    for direction in ['x', 'y']:
      grid = self.get_gridlines(self.get_bounds(self.x), self.get_bounds(self.y), grid_dir=direction)
      for grid_x, grid_y, pt in grid:
        grid_x, grid_y = self.squigglify(grid_x, grid_y, noise_strength=0.05)
        args = {'c': 'black', 'linewidth': 2} if np.isclose(pt, 0, atol=0.01) else \
          {'c': "grey", 'linewidth': 1, 'linestyle': '--', 'alpha': 0.5}
        a, b = (grid_x, grid_y) if direction == 'x' else (grid_y, grid_x)
        plt.plot(a, b, **args)

    csfont = {'fontname': 'xkcd Script', 'fontsize': 30}
    hfont = {'fontname': 'xkcd Script'}
    plt.title('title', **csfont)
    plt.xlabel('xlabel', **hfont)
    plt.axis('off')
    plt.show()
