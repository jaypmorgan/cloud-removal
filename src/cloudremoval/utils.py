from pathlib import Path
import urllib.request


def download(url, output):
    def progress(block_num, block_size, total_size):
        if block_num % 10 == 0:
            down_size = f"{(block_num*block_size)/1000000000:.4f}/{total_size/1000000000:.4f} GB"
            perc_size = f"{round(100*((block_num*block_size)/total_size), 4):.4f}"
            print(
                f"Downloading {Path(url).name}: {down_size} ({perc_size}%)",
                end="\r",
                flush=True,
            )

    urllib.request.urlretrieve(url, output, progress)
    return output
