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
nest.SetKernelStatus({"local_num_threads": 4, "print_time": True})
layer = topology.CreateLayer({
    "extent": [1.1, 1.1],
    "rows": 9,
    "columns": 9,
    "elements": "iaf_psc_alpha"
})

layer1 = topology.CreateLayer({
    "extent": [1.1, 1.1],
    "rows": 4,
    "columns": 4,
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

topology.ConnectLayers(layer1, layer, projection)
fig = topology.PlotLayer(layer, nodesize=40, nodecolor='green')
topology.PlotLayer(layer1, fig, nodesize=40, nodecolor='red')
fig.savefig('test0')
print('Saved test0 image')
center = topology.FindCenterElement(layer1)
print('got center')
topology.PlotTargets(
    center,
    layer,
    fig=fig,
    mask=projection["mask"],
    mask_color='blue',
    kernel=projection["kernel"],
    kernel_color='black',
    tgt_color='yellow'
)
print('Save test1 image')
fig.savefig('test1')
