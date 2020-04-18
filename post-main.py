import pandas as pd
import numpy as np
import png

# TODO deprecate this file.

# TODO create configuration file for shared values.
simulation_time = 250
frames = simulation_time * 10
layer_first_id = 39210
layer_size = 784

data = pd.read_csv(
    'spike_detector-59463-0.gdf',
    '\t',
    header=None,
    usecols=[0, 1],
    names=['neuron', 'time']
)
data['time'] = data['time'] * 10
data.sort_values(by=['time'], inplace=True)
ids = list(range(layer_first_id, layer_first_id + layer_size))
array = np.zeros(len(ids), dtype=np.dtype('uint8'))
grouped = data.groupby(['time'])

for step in range(1, frames):
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
    square = np.reshape(image_frame_array, (-1, int(layer_size ** (1 / 2.0))))
    square = np.kron(square, np.ones((10, 10)))  # "Zoom/Expand" array.
    image = png.from_array(square.astype('uint8'), mode='L;1')
    image.save('./output/spikedetector/' + str(step) + '.png')

# TODO add convert images to a video
# ffmpeg -i %d.png -vf "setpts=(1/5)*PTS" video.webm
#