import os, glob, sys
import shutil

def get_md_path():
    return sys.argv[1]

def get_html_path():
    mdPath = get_md_path()
    return mdPath.replace("md", "html")

def check_html_directory():
    htmlPath =get_html_path()
    if not os.path.exists(htmlPath):
        raise Exception(str(htmlPath) + " directory not exists, please create it before")
    else:
        htmlFiles=os.listdir(htmlPath)
        # if dir exists and html exist, delete them and regenerate
        if (len(htmlFiles)>0):
             for file in htmlFiles:
                if file.endswith(".html"):
                    os.remove(os.path.join(htmlPath, file))
       

def get_all_markdown_files(inputDir):
    mdFiles=list()
    for file in os.listdir(inputDir):
        if file.endswith(".md"):
            #mdFiles.append(os.path.join(inputDir, file))
            mdFiles.append(file)
            #print(os.path.join(inputDir, file))
    print(mdFiles)
    return mdFiles

def create_html_files(mdFiles):
    for mdFile in mdFiles:
        html_file = os.path.join(get_html_path(), str(mdFile).replace(".md",".html"))
        # keep this logging
        print(html_file)
        shutil.copyfile("html/template.html", html_file)

        with open(html_file, "r") as f:
            # TODO, need to get this right
            dir_depth = html_file.count("/")
            relative_md_file_path = '../' * dir_depth+get_md_path()+mdFile
            print(relative_md_file_path)
            new_text = f.read().replace("../md/file.md", relative_md_file_path)
        with open(html_file, 'w') as f:
            f.write(new_text)
        # replace string .md file in the htmlfile

def get_current_dir():
    return os.path.dirname(os.path.realpath(__file__))

def mv_image_files():
    os.chdir(get_md_path())
    types = ('*.png', '*.jpg') # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    print(files_grabbed)

    for imgFile in files_grabbed:
        target_image_file = os.path.join(get_current_dir(),get_html_path(), imgFile)
        src_img_file = os.path.join(get_current_dir(),get_md_path(), imgFile)
        # keep this logging
        print("======================")
        print(src_img_file)
        print(target_image_file)
        print()
        shutil.move(src_img_file, target_image_file)