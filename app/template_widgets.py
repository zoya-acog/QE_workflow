"""
template_widgets.py
-------------------
Widget library for the advQMcalc_test Voila UI.

Provides the shared visual shell — a blue toolbar over a single content area,
mirroring the HubApp pattern from the TrueRho / AutoMD-OXtalS templates.
"""

import ipywidgets as widgets


class HubApp:
    """
    App shell: toolbar over a content area.

    Assign widgets to `content.children` then render `form`.
    """

    def __init__(self, app_name: str = "advQMcalc_test"):
        self.app_name = app_name

        from helpers import build_app_style, build_app_header

        self.content = widgets.VBox(
            children=[],
            layout=widgets.Layout(width="100%"),
        )

        self.form = widgets.VBox(
            [
                widgets.HTML(build_app_style()),
                widgets.HTML(build_app_header()),
                self.content,
            ],
            layout=widgets.Layout(
                width="min(96vw, 1600px)",
                margin="0 auto",
                padding="0 8px 24px",
            ),
        )
