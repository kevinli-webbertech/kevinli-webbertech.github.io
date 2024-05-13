echo "commiting code to github..."
echo $(pwd)
# TODO figure out project root

export project_root="/home/xiaofengli/git/kevinli-webbertech.github.io"
echo $project_root
cd $project_root
git pull --no-ff --no-edit
git add *
git commit -m "updating code"
git push

echo "pushing code to github..."
