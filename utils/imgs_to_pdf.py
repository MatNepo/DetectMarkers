import os
from PIL import Image

from utils.tree2list import tree2list


def main(folder="DICT_5X5_1000"):
    path_list = [i.path for i in tree2list(folder)]
    images = [Image.open(path).convert("RGB") for path in path_list]
    images[0].save(os.path.join(folder, f"out_{folder}.pdf"), save_all=True, append_images=images[1:])

if __name__ == '__main__':
    main()
