import os
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
from itertools import product

fig_size = 4


def _draw_clock_face(radius):
    plt.figure(figsize=(fig_size, fig_size))
    plot = plt.subplot()
    plot.set_xlim([-1.8, 1.8])
    plot.set_ylim([-1.8, 1.8])
    plot.set_xlabel("x", size=14)
    plot.set_ylabel("y", size=14)
    plot.set_aspect("equal")
    plt.axis("off")

    # Make the clock frame
    if random.random() < 0.5:
        x = radius * np.cos(np.linspace(0, 2 * np.pi, 1000))
        y = radius * np.sin(np.linspace(0, 2 * np.pi, 1000))
        plt.plot(x, y, linewidth=4 * random.random(), c="black")
    else:
        line_width = random.randint(1, 4)
        rand_radius = random.uniform(1, 1.4)
        corner_radius = random.uniform(0, 0.3)

        if random.random() < 0.5:
            corner_radius = 0

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.8, 1.8])
        ax.set_aspect("equal")

        rect = patches.FancyBboxPatch(
            (-radius * rand_radius, -radius * rand_radius),
            2 * radius * rand_radius,
            2 * radius * rand_radius,
            linewidth=line_width,
            edgecolor="black",
            facecolor="none",
            boxstyle=patches.BoxStyle.Round(pad=corner_radius),
        )
        ax.add_patch(rect)

    # Make the clock face
    if random.random() < 0.5:
        rand_length = random.random()
        rand_width = random.random()
        for s_ in range(60):
            line_length = (0.1 if s_ % 5 == 0 else 0.05) * rand_length
            line_width = (4 if s_ % 5 == 0 else 2) * rand_width
            x1 = np.sin(np.radians(360 * (s_ / 60))) * radius
            x2 = np.sin(np.radians(360 * (s_ / 60))) * (radius - line_length)
            y1 = np.cos(np.radians(360 * (s_ / 60))) * radius
            y2 = np.cos(np.radians(360 * (s_ / 60))) * (radius - line_length)

            plt.plot([x1, x2], [y1, y2], linewidth=line_width, c="black")

    # Make the clock numbers
    if random.random() < 0.5:
        rand_font_size = random.randint(10, 20)
        for h_ in range(1, 13, 1):
            x = np.sin(np.radians(360 * (h_ / 12))) * radius * 0.8
            y = np.cos(np.radians(360 * (h_ / 12))) * radius * 0.8
            plt.text(
                x,
                y,
                str(h_),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=rand_font_size,
            )


def _calculate_clock_hands(hour, minute, second):
    deg_second = (second / 60) * 360
    deg_minute = (minute / 60) * 360 + (1 / 60) * 360 * (second / 60)
    deg_hour = (hour / 12) * 360 + (1 / 12) * 360 * (minute / 60)
    return deg_second, deg_minute, deg_hour


def create_clock(hour, minute, second, directory):
    radius = random.uniform(0.5, 1.5)
    _draw_clock_face(radius)

    deg_second, deg_minute, deg_hour = _calculate_clock_hands(hour, minute, second)

    second_hand_length = random.uniform(0.8, 0.9)
    minute_hand_length = random.uniform(0.7, 0.9)
    hour_hand_length = random.uniform(0.4, 0.6)

    if random.random() < 0.5:
        color = random.choice(["red", "black"])
        x_second = np.sin(np.radians(deg_second)) * radius * second_hand_length
        y_second = np.cos(np.radians(deg_second)) * radius * second_hand_length
        plt.plot([0, x_second], [0, y_second], linewidth=2 * random.random(), c=color)

    x_minute = np.sin(np.radians(deg_minute)) * radius * minute_hand_length
    y_minute = np.cos(np.radians(deg_minute)) * radius * minute_hand_length
    plt.plot([0, x_minute], [0, y_minute], linewidth=5 * random.random(), c="black")

    x_hour = np.sin(np.radians(deg_hour)) * radius * hour_hand_length
    y_hour = np.cos(np.radians(deg_hour)) * radius * hour_hand_length
    plt.plot([0, x_hour], [0, y_hour], linewidth=5 * random.random(), c="black")

    plt.axis("off")

    clock_fname = os.path.join(
        directory, f"clock-{hour:02d}.{minute:02d}.{second:02d}.png"
    )
    plt.savefig(clock_fname)
    plt.close("all")
    return clock_fname


def main(dir_name, index_fname, generate_num):
    if os.path.isdir(dir_name):
        print(f"Directory {dir_name} already exists. Removing...")
        shutil.rmtree(dir_name)

    os.mkdir(dir_name)
    print(f"Created directory {dir_name}.")

    hours = range(0, 12)
    minutes = range(0, 60)
    seconds = range(0, 60)
    times = [x for x in product(hours, minutes, seconds)]

    # Pick a random subset of the times
    random.shuffle(times)
    times = times[:generate_num]

    with open(index_fname, "w") as index_file:
        TOTAL_IMAGES = len(times)
        with tqdm(total=TOTAL_IMAGES, desc="Generating images", unit="image") as pbar:
            for t in times:
                pbar.update(1)
                clock_fname = create_clock(t[0], t[1], t[2], dir_name)
                index_str = f"{clock_fname}\t{t[0]}\t{t[1]}\t{t[2]}\n"
                index_file.write(index_str)

    print(f"Created {len(times)} clocks.")


if __name__ == "__main__":
    dir_name = "clocks"
    index_fname = "clocks_all.txt"
    generate_num = 1000
    main(dir_name, index_fname, generate_num)
