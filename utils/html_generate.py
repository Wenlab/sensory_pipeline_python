import os

def generate_html(directory):
    # HTML模板的头部
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Directory Listing</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .file-list { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background-color: #f0f0f0; }
            .directory { margin-bottom: 20px; }
            .image { margin-bottom: 20px; }
            img { max-width: 100%; height: auto; }
            a { text-decoration: none; color: blue; }
            .html-content { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; background-color: #f9f9f9; }
            .text-content { border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; background-color: #f0f0f0; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>Directory Listing</h1>
    """

    # 添加返回上级目录的链接
    parent_directory = os.path.abspath(os.path.join(directory, os.pardir))
    if os.path.exists(parent_directory) and parent_directory != directory:
        html_content += f'<div class="file-list"><a href="../index.html">Go to Parent Directory</a></div>\n'

    html_content += """
        <div class="file-list">
            <h2>Directories</h2>
    """

    # 构建子目录列表
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            # 如果是子目录，则添加链接到子目录列表
            html_content += f'<p>Directory: <a href="{filename}/index.html">{filename}</a></p>\n'

    html_content += """
        </div>
        <div class="file-list">
            <h2>Files</h2>
    """

    # 构建文件目录列表
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # 如果文件是图片（支持PNG、JPG、JPEG、GIF）或者HTML文件，则添加到文件目录列表
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) or (filename.lower().endswith('.html') and filename.lower() != 'index.html') or filename.lower().endswith('.txt'):
                html_content += f'<p><a href="#{filename}">{filename}</a></p>\n'
            # 其他文件类型则添加为超链接
            else:
                if filename.lower() != 'index.html':
                    html_content += f'<p><a href="{filename}">{filename}</a></p>\n'

    html_content += """
        </div>
    """

    # 遍历目录中的所有文件并生成内容
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            # 如果文件是图片（支持PNG、JPG、JPEG、GIF），则直接显示
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                html_content += f'<div class="image" id="{filename}"><h3>{filename}</h3><img src="{filename}" alt="{filename}"></div>\n'
            # 如果文件是HTML文件且不是index.html，则嵌入内容
            elif filename.lower().endswith('.html') and filename.lower() != 'index.html':
                with open(filepath, 'r', encoding='utf-8') as html_file:
                    embedded_html = html_file.read()
                    html_content += f'<div class="html-content" id="{filename}"><h3>{filename}</h3>{embedded_html}</div>\n'
            # 如果文件是文本文件，则嵌入内容
            elif filename.lower().endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as text_file:
                    text_content = text_file.read()
                    html_content += f'<div class="text-content" id="{filename}"><h3>{filename}</h3><pre>{text_content}</pre></div>\n'

    # HTML模板的尾部
    html_content += """
    </body>
    </html>
    """

    # 将生成的HTML保存为文件
    output_path = os.path.join(directory, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML file generated at: {output_path}")

def generate_html_for_all_subdirectories(root_directory):
    # 递归遍历所有子目录并生成HTML
    for current_directory, subdirs, files in os.walk(root_directory):
        generate_html(current_directory)

if __name__ == "__main__":
    # 用户输入根目录
    root_directory = r"I:\WJH\flavor\raw_image"
    os.makedirs(root_directory, exist_ok=True)  # 确保根目录存在
    if os.path.isdir(root_directory):
        generate_html_for_all_subdirectories(root_directory)
    else:
        print("The specified root directory does not exist.")