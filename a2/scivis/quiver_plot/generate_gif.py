import imageio
images = [imageio.v2.imread(f'images/1.1.{i}.png') for i in range(4, 14, 2)]
imageio.mimsave('magnitude_proportional.gif', images, duration=2)

images = [imageio.v2.imread(f'images/1.1.{i}.png') for i in range(5, 14, 2)]
imageio.mimsave('same_sized_with_colour.gif', images, duration=2)
