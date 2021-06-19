import nest
import nest.topology as topology
import numpy as np
import pandas as pd
import png
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from mpi4py import MPI
import multiprocessing
from matplotlib import cm
from PIL import Image


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

    def __init__(self, *args, **kwargs):
        print(kwargs)
        self.colmap = cm.get_cmap('viridis', 256)
        self.lut = (self.colmap.colors[..., 0:3] * 255).astype(np.uint8)
        if 're_process' not in kwargs:
            layer, layer_name, simulation_prefix, simulation_time, group_frames, max_spiking_rate = args
            self.layer = layer
            self.layer_name = layer_name
            self.layer_first_id = nest.GetLeaves(self.layer)[0][0]
            self.layer_size = len(nest.GetLeaves(self.layer)[0])
            self.simulation_time = simulation_time
            self.group_frames = group_frames
            self.max_spiking_rate = max_spiking_rate
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

            self.filename = label + '-*.gdf'
            self.output_folder = self.filename.split('-*.gdf')[0] + '/'
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            if rank == 0:
                with open('./output/' + self.simulation_prefix + '/recorder.txt', 'a+') as f:
                    print(f'r = LayerUtils.Recorder.re_process({layer}, \'{layer_name}\', \'{simulation_prefix}\', {simulation_time}, {self.layer_first_id}, {self.layer_size}, {self.group_frames}, {self.max_spiking_rate})', file=f)
                    print('r.make_video(play_it=True, local_num_threads=4)', file=f)
        else:
            layer, layer_name, simulation_prefix, simulation_time, layer_first_id, layer_size, group_frames, max_spiking_rate = args
            self.layer = layer
            self.layer_name = layer_name
            self.layer_first_id = layer_first_id
            self.layer_size = layer_size
            self.simulation_time = simulation_time
            self.group_frames = group_frames
            self.simulation_prefix = simulation_prefix
            self.max_spiking_rate = max_spiking_rate
            folder = './output/' + self.simulation_prefix + '/spike_detector/'
            label = folder + layer_name
            self.filename = label + '-*.gdf'
            self.output_folder = self.filename.split('-*.gdf')[0] + '/'


    @classmethod
    def re_process(cls, layer, layer_name, simulation_prefix, simulation_time, layer_first_id, size, group_frames, max_spiking_rate):
        return cls(layer, layer_name, simulation_prefix, simulation_time, layer_first_id, size, group_frames, max_spiking_rate, re_process=True)

    # TODO create all in memory and read the file and make a +1 to the position.
    def make_video(self, play_it=True, local_num_threads=4):
        # https://www.youtube.com/watch?v=36nCgG40DJo HPC
        # https://www.youtube.com/watch?v=RR4SoktDQAw Threads python.

        print('This should be call after simulation.')
        spikes_data = pd.concat(
            [
                pd.read_csv(f, '\t', header=None, usecols=[0, 1], names=['neuron', 'time'])
                for f in glob.glob(self.filename)
            ]
        )
        # TODO check with there are nulls spikes or/and file concat
        spikes_data = spikes_data.dropna()

        if self.group_frames:
            frames = self.simulation_time / self.group_frames
            spikes_data['time'] = spikes_data['time'] / self.group_frames
            spikes_data['time'] = spikes_data['time'].astype(int)
        else:
            frames = self.simulation_time * 10
            spikes_data['time'] = spikes_data['time'] * 10

        spikes_data.sort_values(by=['time'], inplace=True)
        ids = list(range(self.layer_first_id, self.layer_first_id + self.layer_size))
        array = np.zeros(len(ids), dtype=np.dtype('uint8'))
        grouped = spikes_data.groupby(['time'])

        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        frames_per_rank = int(frames / size)
        from_frame = frames_per_rank * rank
        to_frame = (rank + 1) * frames_per_rank

        parameters = [[step, ids, array, grouped] for step in range(from_frame, to_frame)]
        pool = multiprocessing.Pool(processes=local_num_threads)
        localresult = pool.map(self.process_image, parameters)

        spikes_data = comm.gather(localresult, root=0)
        # Synchronization point
        comm.barrier()

        if rank == 0:
            ffmpeg_command_line = 'ffmpeg -i ' + self.output_folder + '%d.png -vf "setpts=(1/1)*PTS"  -threads 8 ' + self.output_folder + f'../../{self.layer_name}.mp4'
            print(ffmpeg_command_line)
            subprocess.call(
                ffmpeg_command_line,
                shell=True
            )
            eeg = np.array(spikes_data).flatten()
            if play_it:
                subprocess.call('xdg-open ' + self.output_folder + f'../../{self.layer_name}.mp4', shell=True)
            return eeg

    def process_image(self, parameters):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        [step, ids, array, grouped] = parameters
        print("[" + str(rank) + "]" + "Frame: " + str(step))
        image_frame_dict = dict(zip(ids, array.T))

        if step in grouped.groups:
            for row, data in grouped.get_group(step).iterrows():
                neuron = int(data.neuron)
                image_frame_dict[neuron] = image_frame_dict[neuron] + 1
        else:
            print('There is not spikes in this frame')

        image_frame_array = [value for key, value in image_frame_dict.items()]
        square = np.reshape(image_frame_array, (-1, int(self.layer_size ** (1 / 2.0))))
        spike_count = np.sum(square)
        # "Zoom/Expand" array to make images "nicer"
        square = np.kron(square, np.ones((10, 10)))
        # here magic
        rgb_per_spike = ((255*1000) / (self.group_frames * self.max_spiking_rate))
        square = square * (rgb_per_spike if rgb_per_spike < 255 else 255)
        square = square.astype(np.uint8)
        result = np.zeros((*square.shape, 3), dtype=np.uint8)
        np.take(self.lut, square, axis=0, out=result)
        Image.fromarray(result).save(self.output_folder + str(step) + '.png')
        return spike_count


def connect_and_plot_layers_with_projection(origin, target, projection, filename, simulation_prefix, plot=True):
    print(f"Connecting: {filename} start")
    topology.ConnectLayers(origin, target, projection)
    print(f"Connecting: {filename} end")
    if plot:
        fig, ax = plt.subplots()
        topology.PlotLayer(target, fig, nodesize=40, nodecolor='red')
        topology.PlotLayer(origin, fig, nodesize=40, nodecolor='green')
        center = topology.FindCenterElement(origin)

        ax.set_title(filename)
        ax.legend(frameon=False, loc='lower center', ncol=2)

        plt.scatter([], [], c='green', alpha=0.3, s=40, label='Pre')
        plt.scatter([], [], c='red', alpha=0.3, s=40, label='Post')
        plt.scatter([], [], c='yellow', alpha=0.3, s=20, label='Target')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

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


def tuple_connect_and_plot_layers_with_projection(parameters, simulation_prefix, plot):
    [origin, target, projection, filename] = parameters
    connect_and_plot_layers_with_projection(origin, target, projection, filename, simulation_prefix, plot=plot)
