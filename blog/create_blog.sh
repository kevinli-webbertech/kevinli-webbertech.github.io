#!/bin/bash

mdPath="md"
htmlPath="html"
mdFileName=$1

if [[ -z $1 ]];then
  echo "pleaes provide file name in .md format"
  exit 1
fi

mdFilePath=${mdPath}/${mdFileName}
echo "md file path is: ${mdFilePath}"

# GET MD FILE
mdFileName=`cut -d "/" -f 2 <<< $mdFileName`

# GET HTML FILE
htmlFileName=`sed -E s/"md"/"html"/ <<< ${mdFileName}`
echo "html file path is: ${htmlPath}/${htmlFileName}"

# COPY HTML FILE
cp ${htmlPath/}/template.html $htmlPath/${htmlFileName}

# SED TEMPLATE
echo "debugging"
echo ${mdFileName}  
echo ${htmlPath}/${htmlFileName}
sed -i s/"file.md"/"${mdFileName}"/g ${htmlPath}/${htmlFileName}

#https://stackoverflow.com/questions/13210880/replace-one-substring-for-another-string-in-shell-script
#echo `sed s/.md/.html/ $fileName`