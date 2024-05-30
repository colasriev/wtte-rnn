from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from six.moves import xrange

from wtte import weibull


def basic_heatmap(ax, pred, max_horizon=None, resolution=None, cmap="jet"):
    if max_horizon is None:
        max_horizon = pred.shape[0]

    if resolution is None:
        resolution = max_horizon

    ax.imshow(pred.T, origin="lower", interpolation="none", aspect="auto", cmap=cmap)
    ax.set_yticks(
        [
            x * (resolution + 0.0) / max_horizon
            for x in [0, max_horizon / 2, max_horizon - 1]
        ]
    )
    ax.set_yticklabels([0, max_horizon / 2, max_horizon - 1])
    ax.set_ylim(-0.5, resolution - 0.5)
    ax.set_ylabel("steps ahead $s$")
    return ax


def weibull_heatmap(
    fig,
    ax,
    xx,
    alpha,
    beta,
    max_horizon,
    time_to_event=None,
    censoring_indicator=None,
    title="predicted Weibull pmf $p(t,s)$",
    lw=None,
    is_discrete=True,
    resolution=None,
    cmap=None,
    color="red",
):
    """
    Adds a continuous or discrete heatmap with TTE to ax.

    Caveats:
    - axis are pixels so axis's always discrete.
      (so we want location of labels to be in middle)
    """
    if resolution is None:
        # Resolution. Defaults to 1/step. Want more with pdf.
        resolution = max_horizon

    # Discrete
    if is_discrete:
        prob_fun = weibull.pmf
    else:
        prob_fun = weibull.pdf

    if time_to_event is not None:
        if censoring_indicator is not None:
            is_censored = np.array(censoring_indicator).astype(bool)
        else:
            raise NotImplementedError("Implement me")

    # print(f"FFF {is_censored}")

    # Number of timesteps
    xx_length = len(xx)
    assert len(alpha) == xx_length
    assert len(beta) == xx_length
    assert len(time_to_event) == xx_length
    assert len(is_censored) == xx_length

    weibull_values = prob_fun(
        np.tile(np.linspace(0, max_horizon - 1, resolution), (xx_length, 1)),
        np.tile(alpha.reshape(-1, 1), (1, resolution)),
        np.tile(beta.reshape(-1, 1), (1, resolution)),
    )

    ax = basic_heatmap(ax, weibull_values, max_horizon, resolution, cmap=cmap)
    ax.set_title(title)

    def ax_add_scaled_line(
        ax, t, y, y_value_max, y_n_pixels, label, color, linestyle=None
    ):
        # Shifts and scales y to fit on an imshow as we expect it to be, i.e
        # passing through middle of a pixel
        scaled_y = ((y_n_pixels + 0.0) / y_value_max) * y
        ax.plot(t - 0.5, scaled_y, lw=lw, color=color, label=label, linestyle=linestyle)

    if time_to_event is not None:
        if not all(~is_censored):
            ax_add_scaled_line(
                ax,
                xx,
                np.ma.array(time_to_event, mask=is_censored),
                y_value_max=max_horizon,
                y_n_pixels=resolution,
                color=color,
                label="TTE",
            )
            ax_add_scaled_line(
                ax,
                xx,
                np.ma.array(time_to_event, mask=~is_censored),
                y_value_max=max_horizon,
                y_n_pixels=resolution,
                color=color,
                linestyle=":",
                label="TTE (censored)",
            )

    ax.set_xlabel("time")
    fig.tight_layout()

    return fig, ax
