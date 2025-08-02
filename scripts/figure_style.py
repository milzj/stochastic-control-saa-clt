import matplotlib.pyplot as plt
import shutil

fontsize = 12
linewidth = 2

color_styles = 2*['k']
marker_styles = ['d', 'o', 'p', 's', '<']
line_styles = ['-', '--', "-.", ':', '--']
line_styles = ['-', "-."]

plt.rcParams.update({
	"lines.linewidth": linewidth,
	"font.size": fontsize,
	"legend.frameon": True,
	"legend.fontsize": fontsize,
	"legend.framealpha": 1.0,
	"figure.figsize": [5.0, 5.0],
	"savefig.bbox": "tight",
	"savefig.pad_inches": 0.1})

if shutil.which("latex"):  # or use "pdflatex" depending on your TeX installation
    plt.rcParams.update({
	"text.usetex": True,
	"text.latex.preamble": r"\usepackage{amsfonts}",
	"font.family": "serif",
	"font.serif": "Computer Modern Roman",
	"font.monospace": "Computer Modern Typewriter"})
