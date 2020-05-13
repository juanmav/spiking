# https://github.com/nest/nest-simulator/issues/1581
import nest
import nest.topology as topology
import multiprocessing

local_num_threads = 1
multiprocessing_flag = True

conn_ee_dict = {
    "connection_type": "divergent",
    "mask": {
        "circular": {
            "radius": 0.1
        }
    },
    'kernel': {
        'gaussian': {
            'p_center': 1.0,
            'sigma': 0.15
        }
    },
    "weights": 0.1
}

layer_excitatory_dict = {
    "extent": [1.1, 1.1],
    "rows": 100,
    "columns": 100,
    "elements": "iaf_psc_alpha"
}

nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": local_num_threads})

layer1 = topology.CreateLayer(layer_excitatory_dict)
layer2 = topology.CreateLayer(layer_excitatory_dict)

connections = [
    (layer1, layer1, conn_ee_dict, 1),
    (layer1, layer2, conn_ee_dict, 2),
    (layer2, layer2, conn_ee_dict, 3),
    (layer2, layer1, conn_ee_dict, 4)
]


# Process the connections.
def parallel_topology_connect(parameters):
    [pre, post, projection, number] = parameters
    print(f"Connection number: {number}")
    topology.ConnectLayers(pre, post, projection)


if multiprocessing_flag:
    pool = multiprocessing.Pool(processes=4)
    pool.map(parallel_topology_connect, connections)
else:
    for [pre, post, projection, number] in connections:
        topology.ConnectLayers(pre, post, projection)

#nest.Simulate(50)
