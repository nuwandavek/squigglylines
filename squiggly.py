import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

FONT = "xkcd Script"
# FONT = "Humor Sans"
AXIS_DX_PERC = 1
BLACK_AXIS = {'c': 'black', 'linewidth': 2}
GREY_AXIS = {'c': "grey", 'linewidth': 1, 'linestyle': '--', 'alpha': 0.5}


class SquigglyBase:
  def __init__(self, figsize=(20, 10)):
    self.fig, self.ax = plt.subplots(figsize=figsize, facecolor="white")
    self.lines = []

  def get_bounds(self, x):
    return min(x), max(x)

  def get_range(self, xmin, xmax):
    return xmax - xmin

  def smooth(self, new_x, new_y, autocorr_perc):
    autocorr_window = int(len(new_x) * autocorr_perc / 100)
    smoother = np.ones(autocorr_window)
    new_y = np.convolve(new_y, smoother, 'same')
    new_y_denom = np.ones_like(new_y) * autocorr_window
    new_y_denom[:autocorr_window // 2] = np.arange(autocorr_window // 2, autocorr_window)
    new_y_denom[-autocorr_window // 2:] = np.arange(autocorr_window // 2, autocorr_window)[::-1] + 1
    new_y /= new_y_denom
    return new_y

  def squigglify(self, x, y, dx_perc=0.1, noise_strength=0.5, autocorr_perc=5):
    xmin, xmax = self.get_bounds(x)
    xrange = self.get_range(xmin, xmax)
    dx = xrange * dx_perc / 100
    new_x = np.arange(xmin, xmax + dx, dx)
    new_y = np.interp(new_x, x, y)
    noisey = np.random.normal(0, noise_strength, len(new_y))
    new_y = new_y + noisey

    new_y = self.smooth(new_x, new_y, autocorr_perc)

    return new_x, new_y

  def get_gridlines(self, x_bounds, y_bounds, nticks=5, extend_perc=0.03, grid_dir='x', dx_perc=0.1):
    xrange = self.get_range(*x_bounds)
    yrange = self.get_range(*y_bounds)
    a, b, arange = (x_bounds, y_bounds, xrange) if grid_dir == 'x' else (y_bounds, x_bounds, yrange)

    new_a_bounds = (a[0] - arange * extend_perc, a[1] + arange * extend_perc)
    grid_pts_b = np.linspace(b[0], b[1], nticks)
    da = (new_a_bounds[1] - new_a_bounds[0]) * dx_perc / 100

    grid = []
    a_axis = np.arange(new_a_bounds[0], new_a_bounds[1] + da, da)
    for pt in grid_pts_b:
      grid.append((a_axis, np.ones_like(a_axis) * pt, pt))
    return grid


class SquigglyPlot(SquigglyBase):
  def __init__(self, figsize=(20, 10)):
    super().__init__(figsize)

  def draw_line(self, x, y, save_line=True, dx_perc=0.1, noise_strength=0.5, autocorr_perc=5, linewidth=3, alpha=0.75, **kwargs):
    assert ((len(x) > 3) and (len(y) > 3)), "Not enough x/y points!"
    assert (len(x) == len(y)), "len(x) != len(y)"
    x = np.array(x)
    y = np.array(y)
    xx = np.array([ele.timestamp() if isinstance(ele, datetime) else ele for ele in x])
    sqx, sqy = self.squigglify(xx, y, dx_perc, noise_strength, autocorr_perc)
    line = self.ax.plot(sqx, sqy, alpha=alpha, linewidth=linewidth, **kwargs)
    if save_line:
      self.lines.append(line[0])

  def draw_grid(self, xbounds, ybounds, legend=False):
    xtype = 'datetime' if isinstance(xbounds[0], datetime) else 'val'
    xbounds = [ele.timestamp() if xtype == 'datetime' else ele for ele in xbounds]
    xorigin = 0 if xtype == 'val' else xbounds[0]
    yorigin = 0 if ybounds[0] <= 0 <= ybounds[1] else ybounds[0]
    delx, dely = self.get_range(*xbounds) * AXIS_DX_PERC / 100, self.get_range(*ybounds) * AXIS_DX_PERC / 100

    for direction in ['x', 'y']:
      grid = self.get_gridlines(xbounds, ybounds, grid_dir=direction)
      for grid_x, grid_y, pt in grid:
        grid_x, grid_y = self.squigglify(grid_x, grid_y, noise_strength=0.005 * self.get_range(*xbounds) if direction == 'y'
                                         else 0.05 * self.get_range(*ybounds))
        args = BLACK_AXIS if np.isclose(pt, 0 if direction == 'x' else xbounds[0], atol=0.01) else GREY_AXIS
        tick_idxs = np.where(((xorigin - delx) <= grid_x) * (grid_x <= (xorigin + delx)))[0] if direction == 'x' else\
          np.where(((yorigin - 2 * dely) <= grid_x) * (grid_x <= (yorigin + 2 * dely)))[0]

        mini_grid_x, mini_grid_y = grid_x[tick_idxs], grid_y[tick_idxs]
        a, b = (grid_x, grid_y) if direction == 'x' else (grid_y, grid_x)
        mini_a, mini_b = (mini_grid_x, mini_grid_y) if direction == 'x' else (mini_grid_y, mini_grid_x)

        self.ax.plot(a, b, **args)
        self.ax.plot(mini_a, mini_b, **BLACK_AXIS)
        self.ax.text(
          xbounds[0] - delx * 3 if direction == 'x' else pt,
          pt if direction == 'x' else - dely * 8,
          str(round(pt, 1)) if (xtype == 'val' or direction == 'x') else datetime.fromtimestamp(pt).strftime("%b, %Y"),
          horizontalalignment="center",
          # rotation="horizontal" if direction == 'x' else "vertical",
          fontsize=20,
          fontname=FONT
        )
    self.ax.axis('off')
    if legend:
      xlim = xbounds[0] + 10 * delx
      ylim = ybounds[0] - 5 * dely
      for ele, line in enumerate(self.lines):
        eley = ylim - 10 * (ele + 2) * dely
        self.draw_annotations([xbounds[0] + delx, xlim], [eley, eley],
                              [xlim + 1.2 * delx, eley], line._label, c=line._color, alpha=0.75, linewidth=3, noise_strength=0.02, textbg=False)

  def draw_title(self, title):
    title_font = {'fontname': FONT, 'fontsize': 30}
    self.fig.suptitle(title, **title_font)
    self.fig.tight_layout()

  def draw_annotations(self, linexbound, lineybound, textxy, text, c='black', linewidth=1, alpha=1, noise_strength=0.1, textbg=True, fontsize=20):
    if len(linexbound) and len(lineybound):
      linexbound = [ele.timestamp() if isinstance(linexbound[0], datetime) else ele for ele in linexbound]
      x = np.linspace(linexbound[0], linexbound[-1], 20)
      y = np.geomspace(1, lineybound[-1] - lineybound[0] + 1, 20) + lineybound[0] - 1
      self.draw_line(x, y, save_line=False, noise_strength=noise_strength, c=c, linewidth=linewidth, alpha=alpha)
    if len(textxy) and text != "":
      textxy = [a.timestamp() if isinstance(a, datetime) else a for a in textxy]
      t = self.ax.text(textxy[0], textxy[1], text, fontsize=fontsize, fontname=FONT, verticalalignment="center")
      if textbg:
        t.set_bbox(dict(facecolor='#ddd', alpha=0.8, edgecolor='#eee', boxstyle="Round"))
