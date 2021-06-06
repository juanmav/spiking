import os
import nest
import nest.topology as topology
import numpy as np
from math import sqrt, ceil
from LayerUtils import take_poisson_layer_snapshot, Recorder, tuple_connect_and_plot_layers_with_projection
from Patterns import get_pattern_0, get_pattern_1
from RetinaUtils import image_array_to_retina
from Utils import get_simulation_prefix
from Projections import conn_ii_dict, conn_ee_dict, conn_ei_dict, conn_ie_dict, conn_parrot_v1_dict, conn_retina_parrot_dict
import pandas as pd
import subprocess
from mpi4py import MPI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

local_num_threads = int(os.getenv("LOCAL_NUM_THREADS", 1))

nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": local_num_threads, "print_time": True})

simulation_time = int(os.getenv("SIMULATION_TIME", 250))
change_pattern_step = int(os.getenv("CHANGE_PATTERN_STEP", 250))

HYPER_COLUMNS = int(os.getenv("HYPER_COLUMNS", 1))

simulation_prefix = get_simulation_prefix(HYPER_COLUMNS, simulation_time)

RECEPTIVE_FIELD_DENSITY = int(os.getenv("RECEPTIVE_FIELD_DENSITY", 1))
WIDTH_HEIGHT_HYPER_COLUMN = int(os.getenv("WIDTH_HEIGHT_HYPER_COLUMN", 10))
SPATIAL_WIDTH_AND_HEIGHT = 1.0 * round(sqrt(HYPER_COLUMNS), 2)

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

#
max_spiking_rate = int(os.getenv("MAX_SPIKING_RATE", 100))
min_spiking_rate = int(os.getenv("MIN_SPIKING_RATE", 10))

#########################################################################################

# Start Simulation from this point!
# Comment for the sake of understanding for now.
# hard code steps below

retina_dict = {
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

# Log parameters.
layers = pd.DataFrame.from_dict(
    [
        {**{"name": "retina_dict"}, **retina_dict},
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
        {**{"name": "conn_ii_dict"}, **conn_ii_dict}
    ]
)

filename = './output/' + simulation_prefix + '/envvars.txt'
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'a+') as f:
    print("Env Settings")
    print(open('.env').read(), file=f)
    print("Layer definitions")
    print(layers.to_csv(sep='\t'), file=f)
    print("Projection definition")
    print(projections.to_csv(sep='\t'), file=f)

# Retina, LGN.
retina_on = topology.CreateLayer(retina_dict)
#retina_off = topology.CreateLayer(retina_dict)

# Parrot
parrot_retina_on = topology.CreateLayer(parrot_layer_dict)
#parrot_retina_off = topology.CreateLayer(parrot_layer_dict)

# V1 new cortex
ex_on_center = topology.CreateLayer(layer_excitatory_dict)
#in_on_center = topology.CreateLayer(layer_inhibitory_dict)
#ex_off_center = topology.CreateLayer(layer_excitatory_dict)
#in_off_center = topology.CreateLayer(layer_inhibitory_dict)

# Connections
connections = [
    # Receptive field to parrot
    (retina_on, parrot_retina_on, conn_retina_parrot_dict, "1.retina_to_parrot_on"),
    # (retina_off, parrot_retina_off, conn_retina_parrot_dict, "2.retina_to_parrot_off"),
    # Parrot to V1
    (retina_on, ex_on_center, conn_parrot_v1_dict, "3.parrot_to_ex_on"),
    #(retina_on, in_on_center, conn_parrot_v1_dict, "4.parrot_to_in_on"),
    # (retina_off, ex_off_center, conn_parrot_v1_dict, "5.parrot_to_ex_off"),
    # (retina_off, in_off_center, conn_parrot_v1_dict, "6.parrot_to_in_off"),
    # Lateral connection V1
    # ON <==> ON
    #(ex_on_center, ex_on_center, conn_ee_dict, "7.ex_on_to_ex_on"),
    #(ex_on_center, in_on_center, conn_ei_dict, "8.ex_on_to_in_on"),
    #(in_on_center, ex_on_center, conn_ie_dict, "9.in_on_to_ex_on"),
    #(in_on_center, in_on_center, conn_ii_dict, "10.in_on_to_in_on"),
    ## OFF <==> OFF
    # (ex_off_center, ex_off_center, conn_ee_dict, "11.ex_off_to_ex_off"),
    # (ex_off_center, in_off_center, conn_ei_dict, "12.ex_off_to_in_on"),
    # (in_off_center, ex_off_center, conn_ie_dict, "13.in_off_to_ex_off"),
    # (in_off_center, in_off_center, conn_ii_dict, "14.in_off_to_in_off"),
    ## INH_ON ==> OFF
    # (in_on_center, ex_off_center, conn_ie_dict, "15.in_on_to_ex_off"),
    # (in_on_center, in_off_center, conn_ii_dict, "16.in_on_to_in_off"),
    ## INH_OFF => ON
    # (in_off_center, ex_on_center, conn_ie_dict, "17.in_off_to_ex_on"),
    # (in_off_center, in_on_center, conn_ii_dict, "18.in_off_to_in_on")
]

simulate = os.getenv("SIMULATE", "True") == "True"
connect = os.getenv("CONNECT", "True") == "True"
plotLayers = os.getenv("PLOT_LAYERS", "False") == "True"

if connect or simulate:
    for connection in connections:
        tuple_connect_and_plot_layers_with_projection(connection, simulation_prefix, plotLayers)

group_frames = int(os.getenv("GROUP_FRAMES", 0))
# ------------ Measurements Section ----------------
recorder1 = Recorder(parrot_retina_on, 'parrot_retina_on', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
# recorder2 = Recorder(parrot_retina_off, 'parrot_retina_off', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
recorder3 = Recorder(ex_on_center, 'ex_on_center', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
# recorder4 = Recorder(ex_off_center, 'ex_off_center', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
#recorder5 = Recorder(in_on_center, 'in_on_center', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
# recorder6 = Recorder(in_off_center, 'in_off_center', simulation_prefix, simulation_time, group_frames, max_spiking_rate)
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
    for step in range(change_pattern_step, simulation_time + 1, change_pattern_step):
        total_time = + step
        print("Total simulation time:" + str(total_time))

        if flip_flop:
            pattern = image_array_0
            flip_flop = not flip_flop
        else:
            pattern = image_array_1
            flip_flop = not flip_flop

        image_array_to_retina(pattern, retina_on, 'on', max_spiking_rate, min_spiking_rate)
        #image_array_to_retina(pattern, retina_off, 'off', max_spiking_rate, min_spiking_rate)
        take_poisson_layer_snapshot(retina_on, str(step) + "-retina_on", simulation_prefix)
        #take_poisson_layer_snapshot(retina_off, str(step)+"-retina_off", simulation_prefix)

        nest.Simulate(change_pattern_step)

    play_it = os.getenv("PLAY_IT", "False") == "True"

    eeg1 = recorder1.make_video(play_it=play_it, local_num_threads=local_num_threads)
    # eeg2 =  recorder2.make_video(group_frames=group_frames, play_it=play_it, local_num_threads=local_num_threads)
    eeg3 = recorder3.make_video(play_it=play_it, local_num_threads=local_num_threads)
    # eeg4 = recorder4.make_video(group_frames=group_frames, play_it=play_it, local_num_threads=local_num_threads)
    #eeg5 = recorder5.make_video(group_frames=group_frames, play_it=play_it, local_num_threads=local_num_threads)
    # eeg6 = recorder6.make_video(group_frames=group_frames, play_it=play_it, local_num_threads=local_num_threads)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        x_coordinates = np.arange(eeg1.size)

        plt.plot(x_coordinates, eeg1, label='parrot')
        #plt.plot(x_coordinates, eeg3, label='ex_on_center')
        #plt.plot(x_coordinates, eeg5, label='ex_in_center')
        #plt.plot(x_coordinates, eeg3 + eeg5, label='total')
        plt.legend()
        plt.savefig(f'./output/{simulation_prefix}/total_eeg.png')

open_it = os.getenv("OPEN_IT", "False") == "True"
if open_it:
    subprocess.call('xdg-open ' + './output/' + simulation_prefix, shell=True)
