import os

examples_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(examples_dir, "160808/")

images_names = []

for path, subdirs, _ in os.walk(dataset_dir):
    for dir in sorted(subdirs):
        for f in sorted(os.listdir(os.path.join(path, dir))):
            if os.path.isfile(os.path.join(path, dir, f)):
                images_names.append(os.path.join(path, dir, f))

# group images names by pairs
images_pairs = zip(*[images_names[i::2] for i in range(2)])

# write to file
with open('image_pairs.txt', 'w') as fp:
    fp.write('\n'.join('%s %s' % x for x in images_pairs))