"""This is for generating organize the data for plotting."""

import time

time_start = time.perf_counter()

dimensions = [2, 3, 4, 5, 6, 7, 8]
data_dict = {
    "our_sigmoid": [],
    "our_tanh": [],
    "our_maxpool_dlp": [],
}

print("[INFO] Start organizing data...")

for method in [
    "our_sigmoid",
    "our_tanh",
    "our_maxpool_dlp",
]:
    for d in dimensions:
        for n in [3]:

            fpath = f"./polytopes_{d}d_{n}_{method}.txt"

            print(f"[INFO] Loading data from {fpath}...")

            try:
                file = open(fpath, "r")
            except FileNotFoundError:
                print(f"[WARNING] {fpath} Does not found and it is skipped.")
                continue

            lines = file.readlines()  # noqa
            file.close()
            lines = [line.split("\t")[0] for line in lines]
            data = [float(num) for num in lines]

            data_dict[method].append(data)

with open("data.txt", "w") as file:
    file.write(str(data_dict))

print(f"[INFO] Data organized and saved to data.txt.")
print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
