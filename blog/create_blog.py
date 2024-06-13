#!/usr/bin/python

import sys
import os, glob
import shutil

#
def help_menu():
    print('''
        arg1: md dir. For instance, md/ref/python/pandas
    ''')

def check_arg():
    print(len(sys.argv))
    if len(sys.argv)==1:
        help_menu()
        sys.exit(0)

def get_md_path():
    return sys.argv[1]

def get_html_path():
    mdPath = get_md_path()
    return mdPath.replace("md", "html")

def check_html_directory():
    htmlPath =get_html_path()
    if not os.path.exists(htmlPath):
        raise Exception(str(htmlPath) + " directory not exists, please create it before")

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
            relative_md_file_path = '../' * dir_depth+get_md_path()+"/"+mdFile
            print(relative_md_file_path)
            new_text = f.read().replace("../md/file.md", relative_md_file_path)
        with open(html_file, 'w') as f:
            f.write(new_text)
        # replace string .md file in the htmlfile


def main():
    check_arg()
    check_html_directory()
    create_html_files(get_all_markdown_files(get_md_path()))

if __name__ == "__main__":
    main()


