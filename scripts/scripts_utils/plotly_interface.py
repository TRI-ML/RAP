from functools import partial
import os
from typing import Dict, Iterable, Tuple

import gradio as gr
import plotly.graph_objects as go
from torch.utils.data.dataloader import DataLoader

from risk_biased.utils.config_argparse import config_argparse
from risk_biased.utils.load_model import load_from_config
from risk_biased.predictors.biased_predictor import (
    LitTrajectoryPredictor,
)


def to_numpy(**kwargs):
    dic_outputs = {}
    for k, v in kwargs.items():
        dic_outputs[k] = v.detach().cpu().numpy()
    return dic_outputs


def get_scatter_data(x, mask_x, name, **kwargs):
    return [
        go.Scatter(
            x=x[k, mask_x[k], 0],
            y=x[k, mask_x[k], 1],
            showlegend=k == 0,
            name=name,
            **kwargs,
        )
        for k in range(x.shape[0])
    ]


def configuration_paths() -> Iterable[os.PathLike]:
    working_dir = os.path.dirname(os.path.realpath(__file__))
    return [
        os.path.join(
            working_dir,
            "../../risk_biased/config",
            config_file,
        )
        for config_file in ("learning_config.py", "waymo_config.py")
    ]


def load_item(index: int, data_loader: DataLoader) -> Tuple:
    (
        x,
        mask_x,
        y,
        mask_y,
        mask_loss,
        map_data,
        mask_map,
        offset,
        x_ego,
        y_ego,
    ) = data_loader.collate_fn([data_loader.dataset[index]])

    return (x, mask_x, map_data, mask_map, offset, x_ego, y_ego), y, mask_y, mask_loss


def build_data(
    predictor: LitTrajectoryPredictor,
    dataset: DataLoader,
    index: int,
    risk_level: float,
    n_samples: int,
) -> Dict[str, go.Scatter]:
    assert n_samples >= 1

    batch, y, mask_y, mask_loss = load_item(index, dataset)
    predictions = predictor.predict_step(
        batch=batch,
        risk_level=risk_level,
        n_samples=n_samples,
    )

    offset = batch[4]
    y = predictor._unnormalize_trajectory(y, offset)
    x = predictor._unnormalize_trajectory(batch[0], offset)
    numpy_data = to_numpy(
        predictions=predictions,
        y=y,
        mask_y=mask_y,
        x=x,
        mask_x=batch[1],
        map_data=batch[2],
        mask_map=batch[3],
        mask_pred=mask_loss,
    )

    x = numpy_data["x"][0]
    mask_x = numpy_data["mask_x"][0]
    y = numpy_data["y"][0]
    mask_y = numpy_data["mask_y"][0]
    pred = numpy_data["predictions"][0]
    mask_pred = numpy_data["mask_pred"][0]
    map_data = numpy_data["map_data"][0]
    mask_map = numpy_data["mask_map"][0]

    data_x = get_scatter_data(
        x,
        mask_x,
        mode="lines",
        line=dict(width=2, color="black"),
        name="Past",
    )
    ego_present = get_scatter_data(
        x=x[0:1, -1:],
        mask_x=mask_x[0:1, -1:],
        mode="markers",
        marker=dict(color="blue", size=20, opacity=0.5),
        name="Ego",
    )
    agent_present = get_scatter_data(
        x=x[1:2, -1:],
        mask_x=mask_x[1:2, -1:],
        mode="markers",
        marker=dict(color="green", size=20, opacity=0.5),
        name="Agent",
    )

    data_y = get_scatter_data(
        y,
        mask_y,
        mode="lines",
        line=dict(width=2, color="green"),
        name="Ground truth",
    )
    data_map = get_scatter_data(
        map_data,
        mask_map,
        mode="lines",
        line=dict(width=15, color="gray"),
        opacity=0.3,
        name="Centerline",
    )
    data_pred = []
    forecasts_end = []
    for i in range(n_samples):
        cur_data_pred = get_scatter_data(
            pred[:, i],
            mask_pred,
            mode="lines",
            line=dict(width=2, color="red"),
            name="Forecast",
        )
        data_pred += cur_data_pred

        forecast_end = get_scatter_data(
            pred[:, i, -1:],
            mask_pred[:, -1:],
            mode="markers",
            marker=dict(color="red", size=10, opacity=0.5, symbol="x"),
            name="Forecast end",
        )
        forecasts_end += forecast_end

    static_data = (
        data_map
        + data_x
        + data_y
        + data_pred
        + ego_present
        + agent_present
        + forecasts_end
    )

    animation_opacity = 0.5
    frames_x = [
        go.Frame(
            data=[
                go.Scatter(
                    x=x[mask_x[:, k], k, 0],
                    y=x[mask_x[:, k], k, 1],
                    mode="markers",
                    opacity=animation_opacity,
                    marker=dict(color="black", size=15),
                    showlegend=False,
                ),
                go.Scatter(
                    x=x[0:1, k, 0],
                    y=x[0:1, k, 1],
                    mode="markers",
                    opacity=animation_opacity,
                    marker=dict(color="blue", size=15),
                    showlegend=False,
                ),
            ]
        )
        for k in range(x.shape[1])
    ]

    frames_y_pred = []
    for k in range(y.shape[1]):
        cur_gt_agent_data = go.Scatter(
            x=y[1:2][mask_y[1:2, k], k, 0],
            y=y[1:2][mask_y[1:2, k], k, 1],
            mode="markers",
            opacity=animation_opacity,
            marker=dict(color="green", size=15),
        )
        cur_gt_future_data = go.Scatter(
            x=y[2:][mask_y[2:, k], k, 0],
            y=y[2:][mask_y[2:, k], k, 1],
            mode="markers",
            opacity=animation_opacity,
            marker=dict(color="black", size=15),
        )
        cur_pred_data = []
        for i in range(n_samples):
            cur_pred_data.append(
                go.Scatter(
                    x=pred[mask_pred[:, k], i, k, 0],
                    y=pred[mask_pred[:, k], i, k, 1],
                    mode="markers",
                    opacity=animation_opacity,
                    marker=dict(color="red", size=15),
                    showlegend=False,
                )
            )
        cur_ego_data = go.Scatter(
            x=y[0:1, k, 0],
            y=y[0:1, k, 1],
            mode="markers",
            opacity=animation_opacity,
            marker=dict(color="blue", size=15),
        )
        cur_data = [cur_gt_agent_data, cur_gt_future_data, *cur_pred_data, cur_ego_data]
        frame = go.Frame(data=cur_data)
        frames_y_pred.append(frame)

    return {"frames": frames_x + frames_y_pred, "data": static_data}


def prediction_plot(
    predictor: LitTrajectoryPredictor,
    dataset: DataLoader,
    index: int,
    risk_level: float,
    n_samples: int = 1,
    use_biaser: bool = True,
) -> go.Figure:
    range_radius = 70
    if use_biaser:
        risk_level = float(risk_level)
    else:
        risk_level = None
    layout = go.Layout(
        xaxis=dict(
            range=[-2 * range_radius, 2 * range_radius],
            autorange=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[-range_radius, range_radius],
            autorange=False,
            zeroline=False,
        ),
        title_text="Road Scene",
        hovermode="closest",
        width=1200,
        height=600,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                transition=dict(duration=100),
                                frame=dict(duration=100, redraw=False),
                                mode="immediate",
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    fig = go.Figure(
        **build_data(predictor, dataset, index, risk_level, n_samples),
        layout=layout,
    )

    return fig


def update_figure(
    predictor: LitTrajectoryPredictor,
    dataset: DataLoader,
    index: int,
    risk_level: float,
    n_samples: int,
    save_directory: os.PathLike,
) -> go.Figure:
    fig = prediction_plot(
        predictor, dataset, index, risk_level, n_samples, use_biaser=True
    )
    fig.update_layout(transition_duration=10)

    if save_directory != "":
        os.makedirs(save_directory, exist_ok=True)

        filepath = os.path.join(
            save_directory,
            (f"scene_{index}" f"_risk_{risk_level}" f"_n_samples_{n_samples}" ".svg"),
        )
        fig.write_image(filepath)

    return fig


def main():
    predictor, data_loader, _ = load_from_config(config_argparse(configuration_paths()))
    data_loader = data_loader.sample_dataloader()

    # `update_wrapper` is required to ensure that gradio doesn't complain about `ui_update_fn` not
    # having an attribute `.__name__`.
    ui_update_fn = partial(update_figure, predictor, data_loader)

    # Do the same thing as above but using the gradio blocks API
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Risk-Aware Prediction

            Make predictions for the green agent with a risk-seeking bias towards the ego vehicle in blue.
            The risk level is a value between 0 and 1, where 0 is not risk-seeking and 1 is the most risk-seeking.
            If "Use Biased Encoder" is unchecked, the risk level is ignored and the model will make predictions without a risk-seeking bias.

            For more information, see the paper [RAP: Risk-Aware Prediction for Robust Planning](https://arxiv.org/abs/2210.01368) published at CoRL 2022.
        """
        )
        initial_index = 27
        initial_n_samples = 10
        image = gr.Plot(ui_update_fn(initial_index, 0, initial_n_samples, ""))

        index = gr.Slider(
            minimum=0,
            maximum=len(data_loader.dataset) - 1,
            step=1,
            value=initial_index,
            label="Index",
        )
        risk_level = gr.Slider(minimum=0, maximum=1, step=0.01, label="Risk")
        n_samples = gr.Slider(
            minimum=1, maximum=20, step=1, value=initial_n_samples, label="Num Samples"
        )
        # use_biaser = gr.Checkbox(value=True, label="Use Biased Encoder")
        button = gr.Button(label="Re-sample")
        save_directory = gr.Textbox(
            value="",
            lines=1,
            max_lines=1,
            placeholder="Specify path to save image",
            label="Image save path",
        )

        # index.change(ui_update_fn, inputs=[index, risk_level, n_samples, use_biaser, save_directory], outputs=image)
        # risk_level.change(ui_update_fn, inputs=[index, risk_level, n_samples, use_biaser, save_directory], outputs=image)
        # n_samples.change(ui_update_fn, inputs=[index, risk_level, n_samples, use_biaser, save_directory], outputs=image)
        # use_biaser.change(ui_update_fn, inputs=[index, risk_level, n_samples, use_biaser, save_directory], outputs=image)
        button.click(
            ui_update_fn,
            inputs=[index, risk_level, n_samples, save_directory],
            outputs=image,
        )

    interface.launch()


if __name__ == "__main__":
    main()
