from napari.utils.colormaps import Colormap

positions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
colors = [
    (0.0,       0.0,      0.0,      1.0),  # black
    (0.149112,  0.160734, 0.396078, 1.0),  # deep blue
    (0.294641,  0.391785, 0.466667, 1.0),  # bluish gray
    (0.792157,  0.476975, 0.245413, 1.0),  # warm orange
    (0.988235,  0.826425, 0.333287, 1.0),  # warm yellow
    (1.0,       1.0,      1.0,      1.0),  # white
]

sky = Colormap(colors=colors, controls=positions, name="sky")

positions = [0.0, 1.0/3.0, 2.0/3.0, 1.0]
colors = [
    (0.000000, 0.000000, 0.000000, 1.0),  # black
    (0.345098, 0.109804, 0.000000, 1.0),  # dark brown
    (0.737255, 0.501961, 0.000000, 1.0),  # golden
    (0.988235, 0.988235, 0.501961, 1.0),  # pale yellow
]

afm_gold = Colormap(colors=colors, controls=positions, name="afm_gold")