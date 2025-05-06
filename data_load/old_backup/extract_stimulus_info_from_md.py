if __name__ == "__main__":
    folder_path = r'H:\Process_temporary\WJH\olfactory\analysis\olfactory\20240619_wen0065\neuronal_activity'

#%%
import os
import re
import numpy as np
import pandas as pd
import h5py 

#%%
def experiment_extract(md_content):
    '''
    Extracts the experiment information from the markdown file.
    '''
    pattern = r"- (\w+ \w+|\w+): \[(\d+),(\d+)\]" # Find patterns matching for stimulus & buffer intervals
    matches = re.findall(pattern, md_content)
    df = pd.DataFrame(matches, columns=['state', 'start', 'end'])  
    df['start'] = df['start'].astype(int)-1
    df['end'] = df['end'].astype(int)-1
    return df

def extract_experiment_data(markdown_content, start_header, end_header):
    """
    从Markdown内容中提取指定标题范围内的实验数据，并按worm名称分割和提取实验信息。

    参数:
    - markdown_content (str): 完整的Markdown内容。
    - start_header (str): 起始的二级标题（例如 "## 1116"）。
    - end_header (str): 结束的二级标题（例如 "## 1220"）。

    返回:
    - dict: 包含每个worm实验数据的字典，键为worm name，值为提取的实验数据。
    """
    # 匹配从 start_header 到 end_header 之间的内容
    pattern = rf"({start_header}.*?)({end_header}|$)"
    match = re.search(pattern, markdown_content, flags=re.DOTALL)

    if not match:
        print(f"未找到从 {start_header} 到 {end_header} 的内容。")
        return {}

    section = match.group(1)

    sections = re.split(r"(#### w\d+)", section)

    experiment_data = {}
    for i in range(1, len(sections), 2):
        worm_name = sections[i].strip()
        worm_content = sections[i + 1].strip()
        print(f"processing worms: {worm_name}")

        key = re.search(r"w\d+", worm_name).group()
        df = experiment_extract(worm_content)
        experiment_data[key] = df

    return experiment_data

if __name__ == "__main__":
    with open(folder_path+ '/README.md', 'r') as f:
        markdown_content = f.read()
    
    experiment_df_0619 = extract_experiment_data(markdown_content, "## 0619", "## 0901")