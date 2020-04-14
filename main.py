import nest
import nest.topology as topology
import numpy as np
from PIL import Image
from math import sqrt, ceil
import matplotlib.pyplot as plt

nest.ResetKernel()
#nest.SetKernelStatus({"local_num_threads": 4})
#nest.SetKernelStatus({"total_num_virtual_procs": 1})

# TODO change to a uniform distribution
# frequency_excited = 100.0
# frequency_at_rest = 10.0

simulation_time = 250
simulation_step = 1
change_pattern_step = 250

# This must represent the height and width of the input image.
RECEPTIVE_FIELD_WIDTH = 99
RECEPTIVE_FIELD_HEIGHT = 99

# Total neuros per combined layer, Excitatory plus Inhibitory neurons.
# Distribution is usually 1 inhibitory, 4 excitatory
TOTAL_NEUROS_PER_COMBINED_LAYER = 1000
TOTAL_EXCITATORY = TOTAL_NEUROS_PER_COMBINED_LAYER * 0.8
TOTAL_INHIBITORY = TOTAL_NEUROS_PER_COMBINED_LAYER * 0.2
# Square layer proportional to the amount.
EX_V1_WIDTH_AND_HEIGHT = ceil(sqrt(TOTAL_EXCITATORY))
IN_V1_WIDTH_AND_HEIGHT = ceil(sqrt(TOTAL_INHIBITORY))

POISSON_GENERATOR = "my_poisson_generator"
nest.CopyModel("poisson_generator", POISSON_GENERATOR)


# Takes an image and convert it to a 2D array.
# This method need and 101x101 image to work properly.
def array_from_image (file_path):
    im = Image.open(file_path).convert('L')
    #im.show()
    (width, height) = im.size
    ia = np.array(list(im.getdata()))
    ia = ia.reshape((height, width))
    print(ia[0])
    return ia


# Create a full retina (square 3x3 = square RETINA_WIDTH x RETINA_HEIGHT)
def create_retina():
    receptive_layer_parameters = {
        "rows": RECEPTIVE_FIELD_WIDTH,
        "columns": RECEPTIVE_FIELD_HEIGHT,
        "elements": POISSON_GENERATOR
    }
    return topology.CreateLayer(receptive_layer_parameters)


# Get neighborhood from an element, 2D array 3 x 3
def get_local_pattern(x, y, exposed_patter):
    local_pattern = np.zeros((3, 3))
    for row in range(x-1, x+2):
        for column in range(y-1, y+2):
            local_pattern[x - row][y - column] = exposed_patter[row + 1][column + 1]
    return local_pattern


# Vertical / Horizontal             Diagonal / Diagonal "wide"
# 000 000  100  111 111 111   0 3 6 9   100 110  110 0 1 3 4 =>
# 000 000  100  000 111 111             000 100  110
# 000 111  100  000 000 111             000 000  000
def calculate_surroundings_average(local_pattern):
    return np.sum(local_pattern)


#
# 000  111
# 010  101
# 000  111
#
def calculate_visual_spiking_rate_center_on(local_pattern):
    coverage = calculate_surroundings_average(local_pattern)
    frequencies = {
        6: 40,
        9: 80
    }
    return frequencies.get(coverage, 10)


def calculate_visual_spiking_rate_center_off(local_pattern):
    coverage = calculate_surroundings_average(local_pattern)
    frequencies = {
        1: 100,
        3: 70,
        4: 70,
        6: 50
    }
    return frequencies.get(coverage, 10)


def calculate_visual_spiking_rate(local_patter, center_on_or_off):
    return calculate_visual_spiking_rate_center_on(local_patter) \
        if center_on_or_off == 'on' \
        else calculate_visual_spiking_rate_center_off(local_patter)


# Set Poisson generators spiking rate using a receptive filed 3x3.
def image_to_retina(exposed_pattern, retina, center_on_or_off):
    for row in range(0, RECEPTIVE_FIELD_HEIGHT):
        for column in range(0, RECEPTIVE_FIELD_HEIGHT):
            ganglion_cells_id = topology.GetElement(retina, (row, column))
            local_pattern = get_local_pattern(row, column, exposed_pattern)
            rate = calculate_visual_spiking_rate(local_pattern, center_on_or_off)
            nest.SetStatus(ganglion_cells_id, {'rate': rate * 1.0})


def connect_and_plot_layers_with_projection(origin, target, projection, filename):
    fig, ax = plt.subplots()
    topology.ConnectLayers(origin, target, projection)
    topology.PlotLayer(origin, fig, nodesize=40, nodecolor='green')
    topology.PlotLayer(target, fig, nodesize=40, nodecolor='red')
    center = topology.FindCenterElement(origin)

    ax.set_title(filename)
    ax.legend(frameon=False, loc='lower center', ncol=2)

    plt.scatter([], [], c='green', alpha=0.3, s=40, label='Pre')
    plt.scatter([], [], c='red', alpha=0.3, s=40, label='Post')
    plt.scatter([], [], c='yellow', alpha=0.3, s=20, label='Target')
    plt.legend(loc='lower center',  bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

    if ("mask" in projection) and ("kernel" in projection):
        topology.PlotTargets(
            center,
            target,
            fig=fig,
            mask=projection["mask"],
            mask_color='blue',
            kernel=projection["kernel"],
            kernel_color='black',
            tgt_color='yellow',
            tgt_size=10
        )
    else:
        topology.PlotTargets(
            center,
            target,
            fig=fig,
            tgt_color='yellow',
            tgt_size=10
        )

    fig.savefig('./output/' + filename + '.png')


#########################################################################################

# Start Simulation from this point!
# Comment for the sake of understanding for now.
# full_retina_on = create_retina()
# full_retina_off = create_retina()
# hard code steps below

full_retina_dict = {
    "extent": [1.1, 1.1],
    "rows": RECEPTIVE_FIELD_WIDTH,
    "columns": RECEPTIVE_FIELD_HEIGHT,
    "elements": POISSON_GENERATOR
}

parrot_layer_dict = {
    "extent": [1.1, 1.1],
    "rows": RECEPTIVE_FIELD_WIDTH,
    "columns": RECEPTIVE_FIELD_HEIGHT,
    "elements": "parrot_neuron"
}

# Layer definitions
layer_excitatory_dict = {
    "extent": [1.1, 1.1],
    "rows": EX_V1_WIDTH_AND_HEIGHT,
    "columns": EX_V1_WIDTH_AND_HEIGHT,
    "elements": "iaf_psc_alpha"
}

layer_inhibitory_dict = {
    "extent": [1.1, 1.1],
    "rows": IN_V1_WIDTH_AND_HEIGHT,
    "columns": IN_V1_WIDTH_AND_HEIGHT,
    "elements": "iaf_psc_alpha"
}

# Layer projections
# Retina => Parrot layer
conn_retina_parrot_dict = {
    "connection_type": "divergent",
    "mask": {
        "circular": {
            "radius": 0.00000001
        }
    }
}

# Parrot layer => V1
conn_parrot_v1_dict = {
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
    },
    "weights": 1.0
}

# V1 interconnections
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
    "weights": 1.0
}

conn_ei_dict = {
    "connection_type": "divergent",
    "mask": {
        "circular": {
            "radius": 0.1 # incrementar la "ventana"
        }
    },
    'kernel': {
        'gaussian': {
            'p_center': 1.0,
            'sigma': 0.15 #2 a 2.5 sigmas, para aumentar la prob de conexion y no 3 por performance.
        }
    },
    "weights": 1.0
}

conn_ii_dict = {
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
    "weights": -1.0
}

conn_ie_dict = {
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
    "weights": -1.0
}

# Retina, LGN.
full_retina_on = topology.CreateLayer(full_retina_dict)
full_retina_off = topology.CreateLayer(full_retina_dict)

# Parrot
parrot_retina_on = topology.CreateLayer(parrot_layer_dict)
parrot_retina_off = topology.CreateLayer(parrot_layer_dict)

# V1 new cortex
ex_on_center = topology.CreateLayer(layer_excitatory_dict)
in_on_center = topology.CreateLayer(layer_inhibitory_dict)
ex_off_center = topology.CreateLayer(layer_excitatory_dict)
in_off_center = topology.CreateLayer(layer_inhibitory_dict)


# Connections
# Receptive field to parrot
connect_and_plot_layers_with_projection(full_retina_on, parrot_retina_on, conn_retina_parrot_dict, "1.retina_to_parrot_on")
connect_and_plot_layers_with_projection(full_retina_off, parrot_retina_off, conn_retina_parrot_dict, "2.retina_to_parrot_off")

# Parrot to V1
connect_and_plot_layers_with_projection(parrot_retina_on, ex_on_center, conn_parrot_v1_dict, "3.parrot_to_ex_on")
connect_and_plot_layers_with_projection(parrot_retina_on, in_on_center, conn_parrot_v1_dict, "4.parrot_to_in_on")

connect_and_plot_layers_with_projection(parrot_retina_off, ex_off_center, conn_parrot_v1_dict, "5.parrot_to_ex_off")
connect_and_plot_layers_with_projection(parrot_retina_off, in_off_center, conn_parrot_v1_dict, "6.parrot_to_in_off")

# Cross connection V1
# ON <==> ON
connect_and_plot_layers_with_projection(ex_on_center, ex_on_center, conn_ee_dict, "7.ex_on_to_ex_on")
connect_and_plot_layers_with_projection(ex_on_center, in_on_center, conn_ei_dict, "8.ex_on_to_in_on")
connect_and_plot_layers_with_projection(in_on_center, ex_on_center, conn_ei_dict, "9.in_on_to_ex_on")

# OFF <==> OFF
connect_and_plot_layers_with_projection(ex_off_center, ex_off_center, conn_ee_dict, "10.ex_off_to_ex_off")
connect_and_plot_layers_with_projection(ex_off_center, in_off_center, conn_ei_dict, "11.ex_off_to_in_on")
connect_and_plot_layers_with_projection(in_off_center, ex_off_center, conn_ie_dict, "12.in_off_to_ex_off")

# INH_ON ==> OFF
connect_and_plot_layers_with_projection(in_on_center, ex_off_center, conn_ie_dict, "13.in_on_to_ex_off")
connect_and_plot_layers_with_projection(in_on_center, in_off_center, conn_ii_dict, "14.in_on_to_in_off")

# INH_OFF => ON
connect_and_plot_layers_with_projection(in_off_center, ex_on_center, conn_ie_dict, "15.in_off_to_ex_on")
connect_and_plot_layers_with_projection(in_off_center, in_on_center, conn_ii_dict, "16.in_off_to_in_on")


# ------------ Measurements Section ----------------

spike_detector = nest.Create('spike_detector', params={
        "withgid": True,
        "withtime": True,
        "to_file": True,
    })


nest.Connect(nest.GetLeaves(parrot_retina_on)[0], spike_detector)
print(nest.GetLeaves(parrot_retina_on)[0][0])
# --------------------------------------------------------------------------------------------

# TODO which are the neurons's spikes representative? Ex? In? both?


# Patterns
image_array_0 = np.divide(array_from_image("./images/pattern0.png"), 255)
image_array_1 = np.divide(array_from_image("./images/pattern1.png"), 255)

total_time = 0
flip_flop = True
pattern = image_array_0

#for step in range(1, simulation_time, change_pattern_step):
#total_time = + step
print("Total simulation time:" + str(total_time))

if flip_flop:
    pattern = image_array_0
    flip_flop = not flip_flop
else:
    pattern = image_array_1
    flip_flop = not flip_flop

image_to_retina(pattern, full_retina_on, 'on')
image_to_retina(pattern, full_retina_off, 'off')

nest.Simulate(250)


# TODO implement Parrot neuron (proxy)
# check sigma of distributions.

# Inspect status when simulation is "paused"
# Debugging purposes
full_retina_on_ids = nest.GetNodes(full_retina_on)
full_retina_on_rates = [nest.GetStatus(x, 'rate') for x in full_retina_on_ids]
full_retina_on_rates = np.reshape(full_retina_on_rates, (-1, RECEPTIVE_FIELD_WIDTH))
print(full_retina_on_rates)

