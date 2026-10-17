"""Regression tests for plotted uniform-noise variances and sample geometry."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import plot_histograms_qqplots as sample_plots
from scripts import plot_true_variance as variance_plots


class PlotRegressionTests(unittest.TestCase):
    def setUp(self):
        self.style = matplotlib.rc_context({'text.usetex': False})
        self.style.__enter__()
        self.params = {'N': 50, 'REPLICATIONS': 12, 'X_EVAL': 1.0}

    def tearDown(self):
        plt.close('all')
        self.style.__exit__(None, None, None)

    def test_import_has_no_directory_creation_and_uses_project_paths(self):
        with patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir during import')):
            importlib.reload(sample_plots)
            importlib.reload(variance_plots)
        self.assertEqual(sample_plots.save_path, PROJECT_ROOT / 'data/histograms_qqplots')
        self.assertEqual(variance_plots.save_path, PROJECT_ROOT / 'data/true_variance')
        plt.rcParams['text.usetex'] = False

    def test_qq_has_one_point_series_one_fitted_line_and_common_unclipped_axes(self):
        results = {1: np.linspace(20, 50, 12), 4: np.array([-50., -9., -2., 0., 1., 2.])}
        figures = []
        with patch.object(sample_plots, '_save_figure', side_effect=lambda fig, *_: figures.append(fig)):
            sample_plots.plot_probplot_scipy_plots(results, self.params)
        self.assertEqual(len(figures), 2)
        for fig, data in zip(figures, results.values()):
            ax = fig.axes[0]
            data_lines = [line for line in ax.lines if len(line.get_xdata())]
            self.assertEqual(len(data_lines), 2)
            (osm, osr), (slope, intercept, _) = stats.probplot(data, dist='norm')
            assert_allclose(data_lines[0].get_xdata(), osm)
            assert_allclose(data_lines[0].get_ydata(), osr)
            assert_allclose(data_lines[1].get_xdata(), osm)
            assert_allclose(data_lines[1].get_ydata(), intercept + slope * osm)
            for line in data_lines:
                self.assertLessEqual(ax.get_xlim()[0], min(line.get_xdata()))
                self.assertGreaterEqual(ax.get_xlim()[1], max(line.get_xdata()))
                self.assertLessEqual(ax.get_ylim()[0], min(line.get_ydata()))
                self.assertGreaterEqual(ax.get_ylim()[1], max(line.get_ydata()))
        self.assertEqual(figures[0].axes[0].get_xlim(), figures[1].axes[0].get_xlim())
        self.assertEqual(figures[0].axes[0].get_ylim(), figures[1].axes[0].get_ylim())

    def test_histogram_keeps_tail_samples_and_handles_deterministic_results(self):
        results = {1: np.r_[np.zeros(100), 25.], 2: np.full(12, 3.)}
        figures = []
        with patch.object(sample_plots, '_save_figure', side_effect=lambda fig, *_: figures.append(fig)):
            with np.errstate(divide='raise', invalid='raise'):
                sample_plots.plot_histograms(results, self.params)
        for fig, data in zip(figures, results.values()):
            ax = fig.axes[0]
            self.assertLess(ax.get_xlim()[0], float(min(data)))
            self.assertGreater(ax.get_xlim()[1], float(max(data)))
            self.assertTrue(np.isfinite(ax.get_ylim()).all())
            for line in ax.lines:
                if len(line.get_ydata()):
                    self.assertLessEqual(max(line.get_ydata()), ax.get_ylim()[1])
        self.assertEqual([text.get_text() for text in figures[0].axes[0].get_legend().get_texts()][1:],
                         [r'$N^{1/2}(\hat{V}_{t,N}(x_t) - V_t(x_t))$', 'Normal fit'])
        self.assertEqual(len([line for line in figures[1].axes[0].lines if len(line.get_xdata())]), 0)

    def test_surfaces_use_uniform_moment_and_include_zero_propagation(self):
        params = {'T': 2, 'A': 1., 'B': 1., 'Sigma': 1.25}
        P = np.array([3., 2., 1.])
        K = np.array([-.6, -.5])
        S = np.array([2., 1., 0.])
        v = np.array([5., 2., 0.])
        surfaces, figures = [], []
        original_plot_surface = Axes3D.plot_surface
        def capture_surface(ax, time, state, values, **kwargs):
            surfaces.append((state.copy(), values.copy(), kwargs.copy()))
            return original_plot_surface(ax, time, state, values, **kwargs)
        with patch.object(Axes3D, 'plot_surface', new=capture_surface):
            with patch.object(variance_plots, '_save_figure', side_effect=lambda fig, *_: figures.append(fig)):
                variance_plots.plot_variance_surfaces(params, P, K, S, v)
        self.assertEqual(len(surfaces), 3)
        x = surfaces[0][0]
        M = params['A'] + params['B'] * K
        expected_current = 4 * x**2 * M**2 * P[1:]**2 * params['Sigma']**2
        expected_current += (4 / 5) * P[1:]**2 * params['Sigma']**4
        expected_propagated = x**2 * M**2 * S[1:] + v[1:]
        assert_allclose(surfaces[0][1], expected_propagated)
        assert_allclose(surfaces[1][1], expected_current)
        assert_allclose(surfaces[2][1], expected_current + expected_propagated)
        assert_allclose(surfaces[0][1][:, -1], 0.)
        for (_, values, kwargs), fig in zip(surfaces, figures):
            self.assertEqual(kwargs['vmin'], 0.)
            self.assertEqual(fig.axes[0].get_zlim()[0], 0.)
            self.assertGreaterEqual(fig.axes[0].get_zlim()[1], float(values.max()))
            self.assertIn(r'\sigma_{t,', fig.axes[0].get_zlabel())

    def test_plot_rejects_empty_nonfinite_or_insufficient_qq_samples(self):
        for results in ({}, {1: []}, {1: [np.nan]}, {1: [1.]}):
            with self.subTest(results=results), self.assertRaises(ValueError):
                sample_plots.plot_probplot_scipy_plots(results, self.params)
        with self.assertRaises(ValueError):
            sample_plots.plot_histograms({1: [np.inf]}, self.params)

    def test_save_creates_requested_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'nested'
            with patch.object(matplotlib.figure.Figure, 'savefig') as savefig:
                sample_plots.plot_probplot_scipy_plots({1: [-1., 0., 1.]}, self.params,
                                                     save_path=destination)
            self.assertTrue(destination.is_dir())
            self.assertEqual(savefig.call_args.args[0].parent, destination)


if __name__ == '__main__':
    unittest.main()
