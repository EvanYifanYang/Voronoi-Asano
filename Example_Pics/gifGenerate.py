from PIL import Image

frames = [Image.open(f'{i}.png') for i in range(53)]

extra_pause_frames = 15
for _ in range(extra_pause_frames):
    frames.append(frames[-1])


frames[0].save(
    'Asano.gif',
    save_all=True,
    append_images=frames[1:],
    duration=300,
    loop=0
)