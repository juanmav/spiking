#
# This script is only for testing purposes. Its goal is to check
# that the environment is running smoothly.
#
#   Run:
#   > conda activate spiking
#   > python testPlot.py
#
import nest
import nest.topology as topology

nest.ResetKernel()

layer = topology.CreateLayer({
    "rows": 5,
    "columns": 5,
    "elements": "iaf_psc_alpha"
})

projection = {
    "connection_type": "divergent",
    "mask": {
        "circular": {
            "radius": 0.3
        }
    },
    'kernel': {
        'gaussian': {
            'p_center': 1.0,
            'sigma': 0.15
        }
    }
}

topology.ConnectLayers(layer, layer, projection)

fig = topology.PlotLayer(layer, nodesize=80)
fig.savefig('test0')
center = topology.FindCenterElement(layer)
topology.PlotTargets(
    center,
    layer,
    fig=fig,
    mask=projection["mask"],
    kernel=projection["kernel"],
    tgt_color='red'
)

fig.savefig('test1')
