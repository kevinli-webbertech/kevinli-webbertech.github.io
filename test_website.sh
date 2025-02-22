#!/bin/bash

#URL="$1"
URL="https://kevinli-webbertech.github.io/blog.html"

if [ -z "$URL" ]; then
  echo "Please provide a URL as an argument."
  exit 1
fi

check_link() {
  local link="$1"
  status_code=$(curl -s -o /dev/null -w "%{http_code}" "$link")

  if [[ "$status_code" -ge 400 ]]; then
    echo "Broken link: $link (Status code: $status_code)"
  elif [[ "$status_code" -ge 300 && "$status_code" -lt 400 ]]; then
    echo "Redirection link: $link (Status code: $status_code)"
  elif [[ "$status_code" -eq 404 ]]; then
    echo "$link is broken..."
  fi
}

extract_links() {
  local content=$(curl -s "$URL")
  
  # Extracts href attributes from <a> tags
  links=$(grep -oE 'href="[^"]+"' <<< "$content" | sed 's/href="//g' | sed 's/"//g')

  echo "debugging here..."

  for link in $links; do
    echo $link
    # Check only http or https links
    if [[ "$link" == http* ]];
    then
        check_link "$link"
    fi
  done
}

extract_links
