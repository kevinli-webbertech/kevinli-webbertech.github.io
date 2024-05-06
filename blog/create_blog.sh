#!/bin/bash

mdPath=./md
htmlPath=./html
mdFileName=$1

if [[ -z $1 ]];then
  echo "pleaes provide file name in .md format"
  exit 1
fi

# strip off .md
mdFilePath=${mdPath}/${mdFileName}

#sed s/file.md/$fileName/ 

echo "md file path is: ${mdFilePath}"

echo "generating new blog html file"

#echo htmlFileName=${htmlPath}/`echo ${mdFileName}| cut -d "." -f 1`.html

#htmlFileName=`echo ${mdFilePath}| cut -d "." -f 1`

htmlFileName=`echo ${mdFilePath}`

echo "html file path is: ${htmlFileName}"
#cp ${htmlPath/}/template.html $htmlPath/${htmlFileName}



#https://stackoverflow.com/questions/13210880/replace-one-substring-for-another-string-in-shell-script
#echo `sed s/.md/.html/ $fileName`