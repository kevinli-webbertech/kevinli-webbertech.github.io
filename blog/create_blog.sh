#!/bin/bash

mdPath="md"
htmlPath="html"
mdDir=$1


if [[ -z $1 ]];then
  echo "please provide a file in .md format"
  exit 1
fi

getHtmlDir() {
  return
}

getAllMarkdownfiles() {
  files=$(ls $mdDir/*.md)
  echo $files
}

getMDWithoutHtml() {
  mdFiles=$(getAllMarkdownfiles)
  for file in $mdFiles; do
      echo "==="
      echo $file
      htmlPath=$(getHTMLFilePath $file)
      echo "****"
      echo $htmlPath
      echo "\n"
  done
}

getHTMLFilePath() {
  mdFilePath=$1
  echo "xxxx"
  echo -e "mdFilePath: $mdFilePath\n"
  # GET MD FILE
  mdFile=`echo ${mdFilePath} |rev | cut -d '/' -f 1|rev`

  # GET HTML FILE
  htmlFileName=`sed -e s/"md"/"html"/ <<< ${mdFile}`
  echo -e "htmlFilePath is: ${htmlPath}/${htmlFileName}\n"
}

  # COPY HTML FILE
 # cp ${htmlPath}/template.html $htmlPath/${htmlFileName}

  # SED TEMPLATE
  #echo "debugging"
  #echo ${mdFileName}  
  #echo ${htmlPath}/${htmlFileName}

  #sed -i s/"file.md"/"${mdFileName}"/g ${htmlPath}/${htmlFileName}
  #https://stackoverflow.com/questions/13210880/replace-one-substring-for-another-string-in-shell-script
  #echo `sed s/.md/.html/ $fileName`


#mdFiles=$(getAllMarkdownfiles)
echo "testing...."
#echo $mdFiles
getMDWithoutHtml
#create_html
