#!/bin/bash

# TODO need to fix this file

#URL="$1"
URL="https://kevinli-webbertech.github.io/blog.html"
BASE_URL="https://kevinli-webbertech.github.io/"

broken_link=()

if [ -z "$URL" ]; then
  echo "Please provide a URL as an argument."
  exit 1
fi


# TODO some links are returning 200 but it was actually just blank pages

check_link() {
  local link="$1"
  status_code=$(curl -s -o /dev/null -w "%{http_code}" "$link")
 
  #if [[ "$status_code" -ge 400 ]]; then
  #  echo "Broken link: $link (Status code: $status_code)"
  if [[ "$status_code" -ge 300 && "$status_code" -lt 400 ]]; then
    echo "Redirection link: $link (Status code: $status_code)"
  elif [[ "$status_code" -eq 404 ]]; then
    echo "$link is broken.., add it to the final set."
    echo $status_code
    broken_link+=("$link")
    #echo ${broken_link[@]}
  else
    echo "$link, status_code: $status_code"
  fi
}

check_empty_page() {
  local content=$(curl -s "$URL")
  markdown_content=$(grep "markdown-body" <<< $content)
  echo $markdown_content
}

# Core function
extract_links() {
  local content=$(curl -s "$URL")
  
  # Extracts href attributes from <a> tags
  links=$(grep -oE 'href="[^"]+"' <<< "$content" | sed 's/href="//g' | sed 's/"//g')

  echo "debugging here..."

  for link in $links; do
    # Check only http or https links
    if [[ "$link" == http* ]]; then
        check_link "$link"
    else
       link=$BASE_URL$link
       #echo "full link: $link"
       check_link "$link"
    fi
  done

  echo "==============Printing Broken Links======================"
  #echo "${broken_link[@]}" 
  for i in "${broken_link[@]}"
  do
    echo "$i"
  done
}

#extract_links
check_empty_page
