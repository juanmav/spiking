import os
import nest
import nest.topology as topology
import numpy as np
from math import sqrt, ceil
from LayerUtils import take_poisson_layer_snapshot, Recorder, tuple_connect_and_plot_layers_with_projection
from Patterns import get_pattern_0, get_pattern_1
from RetinaUtils import image_array_to_retina
from Utils import get_simulation_prefix
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

local_num_threads = int(os.getenv("LOCAL_NUM_THREADS", 1))

nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": local_num_threads, "print_time": True})

simulation_time = int(os.getenv("SIMULATION_TIME", 250))
change_pattern_step = int(os.getenv("CHANGE_PATTERN_STEP", 250))
simulation_prefix = get_simulation_prefix()

HYPER_COLUMNS = int(os.getenv("HYPER_COLUMNS", 1))
RECEPTIVE_FIELD_DENSITY = int(os.getenv("RECEPTIVE_FIELD_DENSITY", 1))
WIDTH_HEIGHT_HYPER_COLUMN = int(os.getenv("WIDTH_HEIGHT_HYPER_COLUMN", 10))
SPATIAL_WIDTH_AND_HEIGHT = 1.0 * round(sqrt(HYPER_COLUMNS),2)

# Total neuros per combined layer, Excitatory plus Inhibitory neurons.
# Distribution is usually 1 inhibitory, 4 excitatory
TOTAL_NEURONS_PER_COMBINED_COLUMN_LAYER = int(os.getenv("TOTAL_NEURONS_PER_COMBINED_COLUMN_LAYER", 30))
TOTAL_NEURONS_COUNT = HYPER_COLUMNS * WIDTH_HEIGHT_HYPER_COLUMN ** 2 * TOTAL_NEURONS_PER_COMBINED_COLUMN_LAYER
TOTAL_EXCITATORY = TOTAL_NEURONS_COUNT * float(os.getenv("EXCITATORY_PROP", 0.8))
TOTAL_INHIBITORY = TOTAL_NEURONS_COUNT * float(os.getenv("INHIBITORY_PROP", 0.2))
# Square layer proportional to the amount.
EX_V1_WIDTH_AND_HEIGHT = ceil(sqrt(TOTAL_EXCITATORY))
IN_V1_WIDTH_AND_HEIGHT = ceil(sqrt(TOTAL_INHIBITORY))

print("EX_V1_WIDTH_AND_HEIGHT = " + str(EX_V1_WIDTH_AND_HEIGHT))
print("IN_V1_WIDTH_AND_HEIGHT = " + str(IN_V1_WIDTH_AND_HEIGHT))

# Receptive field size TODO check this.
TOTAL_RECEPTIVE_FIELD_COUNT = HYPER_COLUMNS * WIDTH_HEIGHT_HYPER_COLUMN ** 2 * RECEPTIVE_FIELD_DENSITY
RECEPTIVE_FIELD_HEIGHT_WIDTH = ceil(sqrt(TOTAL_RECEPTIVE_FIELD_COUNT))
image_size = ceil(sqrt(TOTAL_RECEPTIVE_FIELD_COUNT) / 10)

#########################################################################################

# Start Simulation from this point!
# Comment for the sake of understanding for now.
# hard code steps below

full_retina_dict = {
    "extent": [SPATIAL_WIDTH_AND_HEIGHT, SPATIAL_WIDTH_AND_HEIGHT],
    "rows": RECEPTIVE_FIELD_HEIGHT_WIDTH,
    "columns": RECEPTIVE_FIELD_HEIGHT_WIDTH,
    "elements": "poisson_generator"
}

parrot_layer_dict = {
    "extent": [SPATIAL_WIDTH_AND_HEIGHT, SPATIAL_WIDTH_AND_HEIGHT],
    "rows": RECEPTIVE_FIELD_HEIGHT_WIDTH,
    "columns": RECEPTIVE_FIELD_HEIGHT_WIDTH,
    "elements": "parrot_neuron"
}

# Layer definitions
layer_excitatory_dict = {
    "extent": [SPATIAL_WIDTH_AND_HEIGHT, SPATIAL_WIDTH_AND_HEIGHT],
    "rows": EX_V1_WIDTH_AND_HEIGHT,
    "columns": EX_V1_WIDTH_AND_HEIGHT,
    "elements": "iaf_psc_alpha"
}

layer_inhibitory_dict = {
    "extent": [SPATIAL_WIDTH_AND_HEIGHT, SPATIAL_WIDTH_AND_HEIGHT],
    "rows": IN_V1_WIDTH_AND_HEIGHT,
    "columns": IN_V1_WIDTH_AND_HEIGHT,
    "elements": "iaf_psc_alpha"
}

# Layer projections
# Retina => Parrot layer
conn_retina_parrot_dict = {
    "connection_type": "convergent",
    "mask": {
        "grid": {
            "rows": 1,
            "columns": 1
        }
    }
}

# Parrot layer => V1
conn_parrot_v1_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 0.45
        }
    },
    #'kernel': {
    #    'gaussian': {
    #        'p_center': 1.0,
    #        'sigma': 0.15
    #    }
    #},
    "weights": 5.0,
    "delays": {
        "linear": {
            "c": 0.1,
            "a": 0.2
        }
    }
}

# p(d) = c + ad
# V1 interconnections
conn_ee_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    'kernel': {
        'gaussian': {
            'p_center': 1.0,
            'sigma': 0.15
        }
    },
    "weights": 1.0,
    "delays": {
        "linear":
            {
                "c": 0.1,
                "a": 0.2
            }
    }
}


conn_ie_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    #'kernel': {
    #    'gaussian': {
    #        'p_center': 1.0,
    #        'sigma': 0.15
    #    }
    #},
    "weights": -0.5,
    "delays": {
        "linear": {
            "c": 0.1, "a": 0.2
        }
    }
}

conn_ei_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    #'kernel': {
    #    'gaussian': {
    #        'p_center': 1.0,
    #        'sigma': 0.15
    #    }
    #},
    "weights": 1.0,
    "delays":
        {
            "linear":
                {
                    "c": 0.1,
                    "a": 0.2
                }
        }
}
conn_ii_dict = {
    "connection_type": "convergent",
    "mask": {
        "circular": {
            "radius": 1.0
        }
    },
    'kernel': {
        'gaussian': {
            'p_center': 1.0,
            'sigma': 0.15
        }
    },
    "weights": -4.0,
    "delays" : { "linear" : { "c": 0.1 , "a" : 0.2 } }
}

# Log parameters.

layers = pd.DataFrame.from_dict(
    [
        {**{"name": "full_retina_dict"}, **full_retina_dict},
        {**{"name": "parrot_layer_dict"}, **parrot_layer_dict},
        {**{"name": "layer_excitatory_dict"}, **layer_excitatory_dict},
        {**{"name": "layer_inhibitory_dict"}, **layer_inhibitory_dict}
    ]
)
projections = pd.DataFrame.from_dict(
    [
        {**{"name": "conn_retina_parrot_dict"}, **conn_retina_parrot_dict},
        {**{"name": "conn_parrot_v1_dict"}, **conn_parrot_v1_dict},
        {**{"name": "conn_ee_dict"}, **conn_ee_dict},
        {**{"name": "conn_ei_dict"}, **conn_ei_dict},
        {**{"name": "conn_ie_dict"}, **conn_ie_dict},
        {**{"name": "conn_ee_dict"}, **conn_ee_dict}
    ]
)

print("Env Settings")
print(open('.env').read())
print("Layer definitions")
print(layers.to_string())
print("Projection definition")
print(projections.to_string())

# TODO copy env file.

# Retina, LGN.
full_retina_on = topology.CreateLayer(full_retina_dict)
#full_retina_off = topology.CreateLayer(full_retina_dict)

# Parrot
parrot_retina_on = topology.CreateLayer(parrot_layer_dict)
#parrot_retina_off = topology.CreateLayer(parrot_layer_dict)

# V1 new cortex
ex_on_center = topology.CreateLayer(layer_excitatory_dict)
in_on_center = topology.CreateLayer(layer_inhibitory_dict)
#ex_off_center = topology.CreateLayer(layer_excitatory_dict)
#in_off_center = topology.CreateLayer(layer_inhibitory_dict)

plotLayers = os.getenv("PLOT_LAYERS", "False") == "True"
# Connections
connections = [
    # Receptive field to parrot
    (full_retina_on, parrot_retina_on, conn_retina_parrot_dict, "1.retina_to_parrot_on", simulation_prefix, plotLayers),
    #    (full_retina_off, parrot_retina_off, conn_retina_parrot_dict, "2.retina_to_parrot_off", simulation_prefix, plotLayers),
    # Parrot to V1
    (full_retina_on, ex_on_center, conn_parrot_v1_dict, "3.parrot_to_ex_on", simulation_prefix, plotLayers),
    (full_retina_on, in_on_center, conn_parrot_v1_dict, "4.parrot_to_in_on", simulation_prefix, plotLayers),
    #(full_retina_off, ex_off_center, conn_parrot_v1_dict, "5.parrot_to_ex_off", simulation_prefix, plotLayers),
    #(full_retina_off, in_off_center, conn_parrot_v1_dict, "6.parrot_to_in_off", simulation_prefix, plotLayers),
    # Lateral connection V1
    # ON <==> ON
    #(ex_on_center, ex_on_center, conn_ee_dict, "7.ex_on_to_ex_on", simulation_prefix, plotLayers),
    (ex_on_center, in_on_center, conn_ei_dict, "8.ex_on_to_in_on", simulation_prefix, plotLayers),
    (in_on_center, ex_on_center, conn_ie_dict, "9.in_on_to_ex_on", simulation_prefix, plotLayers),
    ## OFF <==> OFF
    #(ex_off_center, ex_off_center, conn_ee_dict, "10.ex_off_to_ex_off", simulation_prefix, plotLayers),
    #(ex_off_center, in_off_center, conn_ei_dict, "11.ex_off_to_in_on", simulation_prefix, plotLayers),
    #(in_off_center, ex_off_center, conn_ie_dict, "12.in_off_to_ex_off", simulation_prefix, plotLayers),
    ## INH_ON ==> OFF
    #(in_on_center, ex_off_center, conn_ie_dict, "13.in_on_to_ex_off", simulation_prefix, plotLayers),
    #(in_on_center, in_off_center, conn_ii_dict, "14.in_on_to_in_off", simulation_prefix, plotLayers),
    ## INH_OFF => ON
    #(in_off_center, ex_on_center, conn_ie_dict, "15.in_off_to_ex_on", simulation_prefix, plotLayers),
    #(in_off_center, in_on_center, conn_ii_dict, "16.in_off_to_in_on", simulation_prefix, plotLayers)
]

simulate = os.getenv("SIMULATE", "True") == "True"
connect = os.getenv("CONNECT", "True") == "True"

if connect or simulate:
    for connection in connections:
        tuple_connect_and_plot_layers_with_projection(connection)

# ------------ Measurements Section ----------------
#recorder1 = Recorder(parrot_retina_on, 'parrot_retina_on', simulation_prefix, simulation_time)
#recorder2 = Recorder(parrot_retina_off, 'parrot_retina_off', simulation_prefix, simulation_time)
recorder3 = Recorder(ex_on_center, 'ex_on_center', simulation_prefix, simulation_time)
#recorder4 = Recorder(ex_off_center, 'ex_off_center', simulation_prefix, simulation_time)
#recorder5 = Recorder(in_on_center, 'in_on_center', simulation_prefix, simulation_time)
#recorder6 = Recorder(in_off_center, 'in_off_center', simulation_prefix, simulation_time)

# --------------------------------------------------

# Patterns from Image
# image_array_0 = np.divide(array_from_image("./images/pattern0.png"), 255)
# image_array_1 = np.divide(array_from_image("./images/pattern1.png"), 255)

# Patterns from numpy
image_array_0 = np.pad(np.kron(np.array(get_pattern_0()), np.ones((image_size, image_size))), 1, mode='edge')
image_array_1 = np.pad(np.kron(np.array(get_pattern_1()), np.ones((image_size, image_size))), 1, mode='edge')

# Just for small experiments
diff = len(image_array_0) - RECEPTIVE_FIELD_HEIGHT_WIDTH - 2
print('Diff: ' + str(diff))
image_array_0 = np.delete(image_array_0, range(0, diff), 0)
image_array_0 = np.delete(image_array_0, range(0, diff), 1)
image_array_1 = np.delete(image_array_1, range(0, diff), 0)
image_array_1 = np.delete(image_array_1, range(0, diff), 1)

print('Image rows: ' + str(len(image_array_0)))
print('Image columns: ' + str(len(image_array_0[0])))

total_time = 0
flip_flop = True
pattern = image_array_0

if simulate:
    for step in range(1, simulation_time, change_pattern_step):
        total_time = + step
        print("Total simulation time:" + str(total_time))

        if flip_flop:
            pattern = image_array_0
            flip_flop = not flip_flop
        else:
            pattern = image_array_1
            flip_flop = not flip_flop

        image_array_to_retina(pattern, full_retina_on, 'on')
        #image_array_to_retina(pattern, full_retina_off, 'off')
        take_poisson_layer_snapshot(full_retina_on, str(step)+"-full_retina_on", simulation_prefix)
        #take_poisson_layer_snapshot(full_retina_off, str(step)+"-full_retina_off", simulation_prefix)

        nest.Simulate(change_pattern_step)

    play_it = os.getenv("PLAY_IT", "False") == "True"

    #recorder1.make_video(group_frames=True, play_it=play_it)
    #recorder2.make_video(group_frames=True, play_it=play_it)
    recorder3.make_video(group_frames=True, play_it=play_it)
    #recorder4.make_video(group_frames=True, play_it=play_it)
    #recorder5.make_video(group_frames=True, play_it=play_it)
    #recorder6.make_video(group_frames=True, play_it=play_it)
