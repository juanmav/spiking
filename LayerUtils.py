import nest
import nest.topology as topology
import numpy as np
import pandas as pd
import png
import subprocess
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def take_poisson_layer_snapshot(layer, layer_name, simulation_prefix):
    layer_ids = nest.GetNodes(layer)
    layer_size = len(layer_ids[0])
    layer_rates = [nest.GetStatus(x, 'rate') for x in layer_ids]
    layer_rates = np.reshape(layer_rates, (-1, int(layer_size ** (1 / 2.0))))
    layer_rates = layer_rates.astype('uint8')
    image = png.from_array(layer_rates, mode='L')
    folder = './output/' + simulation_prefix + '/poisson_layer/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    image.save(folder + str(layer_name) + '.png')


class Recorder:

    def __init__(self, layer, layer_name, simulation_prefix, simulation_time):
        self.layer = layer
        self.layer_first_id = nest.GetLeaves(self.layer)[0][0]
        self.layer_size = len(nest.GetLeaves(self.layer)[0])
        self.simulation_time = simulation_time
        self.simulation_prefix = simulation_prefix
        folder = './output/' + self.simulation_prefix + '/spike_detector/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        label = folder + layer_name
        self.spike_detector = nest.Create('spike_detector', params={
            "withgid": True,
            "withtime": True,
            "to_file": True,
            "label": label
        })
        nest.Connect(nest.GetLeaves(self.layer)[0], self.spike_detector)

        self.filename = label + '-' + str(self.spike_detector[0]) + '-*.gdf'
        self.output_folder = self.filename.split('.gdf')[0] + '/'

    # TODO create all in memory and read the file and make a +1 to the position.
    def make_video(self, group_frames=True, play_it=True):
        print('This should be call after simulation.')
        data = pd.concat(
            [
                pd.read_csv(f, '\t', header=None, usecols=[0, 1],names=['neuron', 'time'])
                for f in glob.glob(self.filename)
            ]
        )
        if group_frames:
            frames = self.simulation_time
            data['time'] = data['time'].astype(int)
        else:
            frames = self.simulation_time * 10
            data['time'] = data['time'] * 10

        data.sort_values(by=['time'], inplace=True)
        ids = list(range(self.layer_first_id, self.layer_first_id + self.layer_size))
        array = np.zeros(len(ids), dtype=np.dtype('uint8'))
        grouped = data.groupby(['time'])

        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # for step in range(1, frames):
        #    self.process_image(step, ids, array, grouped)
        Parallel(n_jobs=-3)(delayed(self.process_image)(step, ids, array, grouped) for step in range(1, frames))

        subprocess.call(
            'ffmpeg -i ' + self.output_folder + '%d.png -vf "setpts=(1/3)*PTS"  -threads 4 ' + self.output_folder + '0video.webm',
            shell=True
        )

        if play_it:
            subprocess.call('xdg-open ' + self.output_folder + '0video.webm', shell=True)

    def process_image(self, step, ids, array, grouped):
        print("Frame: " + str(step))
        image_frame_dict = dict(zip(ids, array.T))

        if step in grouped.groups:
            for row, data in grouped.get_group(step).iterrows():
                # print(int(data.neuron))
                neuron = int(data.neuron)
                image_frame_dict[neuron] = 1
        else:
            print('nothing')

        image_frame_array = [value for key, value in image_frame_dict.items()]
        square = np.reshape(image_frame_array, (-1, int(self.layer_size ** (1 / 2.0))))
        square = np.kron(square, np.ones((10, 10)))  # "Zoom/Expand" array.
        image = png.from_array(square.astype('uint8'), mode='L;1')
        image.save(self.output_folder + str(step) + '.png')


def connect_and_plot_layers_with_projection(origin, target, projection, filename, simulation_prefix):
    fig, ax = plt.subplots()
    topology.ConnectLayers(origin, target, projection)
    topology.PlotLayer(target, fig, nodesize=40, nodecolor='red')
    topology.PlotLayer(origin, fig, nodesize=40, nodecolor='green')
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
    folder = './output/' + simulation_prefix + '/layer/'
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(folder + filename + '.png')