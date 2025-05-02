if __name__ == "__main__":
    tiff_dir_name = r"\\192.168.1.192\worm-tools\Jinghao-Wang\test\rawdata\20240914_test\w2_100_2024-09-14_21-19-32\0_Camera-Red_VSC-10629"

#%%
import dask.array as da
import glob
import numpy as np
import tifffile
from pathlib import Path
#%%
def _process_tif(chunck, output_dir, compression="none", level=0, forground_threshold=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = chunck[0]  # 获取文件名
    image = tifffile.imread(filename)
    image[image<forground_threshold] = 0
    output_file_name = output_dir / Path(filename).name

    tifffile.imwrite(
        output_file_name,
        image,
        compression=compression,
        compressionargs=(
            {"level": level} if compression in ("zlib", "adobe_deflate") else None
        ),
        photometric="minisblack",
        metadata={"axes": "YX"},
    )
    return output_file_name

def copy_tiff_using_dask(tiff_dir_name:str, output_dir_name:str, compression="none", level=0, forground_threshold=0):
    tiff_name_format = Path(tiff_dir_name) / "*.tif"
    filenames = da.from_array(glob.glob(str(tiff_name_format)), chunks=(1,))
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filenames = filenames.map_blocks(
        _process_tif,
        dtype=object,  # 输出类型为对象（字符串）
        meta=np.array([], dtype=object),  # 元数据定义
        chunks=(1,),  # 输出分块形状为 1D（每个分块生成一个文件名）
        output_dir=output_dir,
        compression=compression,
        level=level,
        forground_threshold=forground_threshold,
    ).compute()