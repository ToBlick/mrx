import plotly.graph_objects as go
import numpy as np
import plotly.colors as pc

base_markers = [
        'circle', 'triangle-down', 'star', 'triangle-left', 'triangle-right',
        'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw',
        'square', 'pentagon', 'triangle-up', 'hexagon', 'hexagon2',
        'cross', 'x', 'diamond', 'diamond-open', 'line-ns', 'line-ew'
    ]

colorbar = 'Viridis'
def converge_plot(err, ns, ps, qs):
    
    markers = [base_markers[i % len(base_markers)] for i in range(len(ps))]
    colors = pc.sample_colorscale(colorbar, len(qs))
    fig = go.Figure()
    for j, p in enumerate(ps):
        for k, q in enumerate(qs):
            fig.add_trace(go.Scatter(
                x=ns,
                y=err[:, j, k],
                mode='lines+markers',
                name=f'p={p}',
                marker=dict(symbol=markers[j], size=8, color=colors[k]),
                line=dict(color=colors[k], width=2),
                # marker_color=go.scatter.marker.colorscale(colorscale, color_indices[k]),
                showlegend=False,
            ))
    
    #
    for j, marker in enumerate(markers):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol=marker, color=colors[0], size=8),
            name=f'p = {ps[j]}',
            showlegend=True
        ))

    for j, color in enumerate(colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            marker=dict(symbol=None, color=color, size=8),
            name=f'q = {qs[j]}',
            showlegend=True
        ))
        # Asymptotic line
        # asymptotic = err[-1, j, 0] * (ns/ns[-1])**(-2*p)
        # fig.add_trace(go.Scatter(
        #     x=ns,
        #     y=asymptotic,
        #     mode='lines',
        #     name=f'O(1/n^{2*p})',
        #     line=dict(dash='dash', color='black'),
        #     showlegend=True
        # ))

    return fig