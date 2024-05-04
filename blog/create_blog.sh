#!/bin/bash

mdPath=./md
htmlPath=./html
fileName=$1

if [[ -z $1 ]];then
  echo "pleaes provide file name in .md format"
fi

mdFilePath=$mdPath/$fileName

echo "generating new blog html file"

#cp $htmlPath/template.html $htmlPath/$fileName.html
#sed s/file.md/$fileName/ 
echo ${mdFilePath}
#echo `sed s/.md/.html/ $fileName`