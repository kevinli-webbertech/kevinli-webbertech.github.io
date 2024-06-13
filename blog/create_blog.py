#!/usr/bin/python

import sys
import os, glob

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
    for file in os.listdir(inputDir):
        if file.endswith(".md"):
            print(os.path.join(inputDir, file))

def main():
    check_arg()
    check_html_directory()
    get_all_markdown_files(get_md_path())

if __name__ == "__main__":
    main()


