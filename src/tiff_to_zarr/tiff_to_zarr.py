import zarr
from numcodecs import Zstd
import os
import click
import sys
from dask_jobqueue import LSFCluster
from dask.distributed import Client, LocalCluster
import time
from tiff_to_zarr.tiff_stack import TiffStack
from tiff_to_zarr.tiff_volume import TiffVolume


@click.command("tiff-to-zarr")
@click.option(
    "--src",
    "-s",
    type=click.Path(exists=True),
    help="Input tiff file location, or tiff stack directory path.",
)
@click.option("--dest", "-d", type=click.STRING, help="Output .zarr file path.")
@click.option(
    "--num_workers", "-w", default=100, type=click.INT, help="Number of dask workers"
)
@click.option(
    "--cluster",
    "-c",
    default="",
    type=click.STRING,
    help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'",
)
@click.option(
    "--zarr_chunks",
    "-zc",
    nargs=3,
    default=(128, 128, 128),
    type=click.INT,
    help="Chunk size for (z, y, x) axis order. z-axis is normal to the tiff stack plane. Default (64, 128, 128)",
)
@click.option(
    "--slab_layers",
    "-sl",
    default=1,
    type=click.INT,
    help="Separation of a slab with thickness=chunk_size(along z axis) into a set of layers, to avoid running out of RAM. Default num_layers=1",
)
@click.option(
    "--axes",
    "-a",
    nargs=3,
    default=("z", "y", "x"),
    type=str,
    help="Metadata axis names. Order matters. \n Example: -a z y x",
)
@click.option(
    "--translation",
    "-t",
    nargs=3,
    default=(0.0, 0.0, 0.0),
    type=float,
    help="Metadata translation(offset) value. Order matters. \n Example: -t 1.0 2.0 3.0",
)
@click.option(
    "--scale",
    "-s",
    nargs=3,
    default=(1.0, 1.0, 1.0),
    type=float,
    help="Metadata scale value. Order matters. \n Example: -s 1.0 2.0 3.0",
)
@click.option(
    "--units",
    "-u",
    nargs=3,
    default=("nanometer", "nanometer", "nanometer"),
    type=str,
    help="Metadata unit names. Order matters. \n Example: -t nanometer nanometer nanometer",
)
def cli(
    src,
    dest,
    num_workers,
    cluster,
    zarr_chunks,
    axes,
    translation,
    scale,
    units,
    slab_layers,
):

    # create a dask client to submit tasks
    if cluster == "":
        print("Did not specify which instance of the dask client to use!")
        sys.exit(0)
    elif cluster == "lsf":
        num_cores = 1
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory="/scratch/$USER/",
        )

    elif cluster == "local":
        cluster = LocalCluster()

    client = Client(cluster)
    with open(
        os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w"
    ) as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    if os.path.isdir(src):
        tiff_volume = TiffStack(src, axes, scale, translation, units)
    elif src.endswith(".tif") or src.endswith(".tiff"):
        tiff_volume = TiffVolume(src, axes, scale, translation, units)

    z_store = zarr.NestedDirectoryStore(dest)
    z_root = zarr.open(store=z_store, mode="a")
    z_arr = z_root.require_dataset(
        name="s0",
        shape=tiff_volume.shape,
        dtype=tiff_volume.dtype,
        chunks=zarr_chunks,
        compressor=Zstd(level=6),
    )

    # write in parallel to zarr using dask
    client.cluster.scale(num_workers)
    tiff_volume.write_to_zarr(z_arr, client, slab_layers)
    client.cluster.scale(0)
    # populate zarr metadata
    tiff_volume.populate_zarr_attrs(z_root)


if __name__ == "__main__":
    cli()
