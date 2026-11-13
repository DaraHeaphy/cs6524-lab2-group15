import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

COLORMAP = 'turbo_r'   # near = red, far = blue
CM_TICKS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 1000]


class DepthVideoWriter:
    """Writes side-by-side frames (rectified image | depth heatmap with a
    colour bar in cm) to an mp4, using matplotlib for the heatmap and scale.

    The colour range is fixed for the whole video so colours are comparable
    between frames. The scale is logarithmic: the scenes span ~15 cm to ~4 m
    and depth uncertainty grows with Z^2, so a linear scale would squeeze all
    the near-field detail into one colour."""

    def __init__(self, path, size_hw, zmin, zmax, fps=30, title=''):
        h, w = size_hw
        dpi = 100
        self.fig, (ax_img, ax_z) = plt.subplots(
            1, 2, figsize=(2 * w / dpi + 1.6, h / dpi + 0.6), dpi=dpi,
            gridspec_kw={'wspace': 0.05})
        self.fig.subplots_adjust(left=0.01, right=0.86, top=0.93, bottom=0.01)
        self.im = ax_img.imshow(np.zeros((h, w, 3), np.uint8))
        cmap = plt.get_cmap(COLORMAP).copy()
        cmap.set_bad('black')
        self.zim = ax_z.imshow(np.full((h, w), np.nan), cmap=cmap,
                               norm=LogNorm(vmin=zmin, vmax=zmax))
        for ax in (ax_img, ax_z):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_img.set_title('Rectified frame', fontsize=9)
        ax_z.set_title('Depth (black = no estimate)', fontsize=9)
        cax = self.fig.add_axes([0.88, 0.05, 0.025, 0.85])
        cb = self.fig.colorbar(self.zim, cax=cax)
        ticks = [t for t in CM_TICKS if zmin <= t <= zmax]
        cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
        cb.ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
        cb.set_label('Depth (cm, log scale)')
        self.text = self.fig.suptitle(title, fontsize=10)
        self.title = title

        self.fig.canvas.draw()
        W, H = self.fig.canvas.get_width_height()
        self.size = (W, H)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f'Could not open video writer for {path}')

    def write(self, rect_bgr, Z, caption=''):
        self.im.set_data(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2RGB))
        self.zim.set_data(np.ma.masked_invalid(Z))
        self.text.set_text(f'{self.title}   {caption}')
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def snapshot(self, path):
        self.fig.savefig(path, dpi=150)

    def close(self):
        self.writer.release()
        plt.close(self.fig)
