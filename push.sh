echo "commiting code to github..."

project_root=$(pwd)
echo $project_root
cd $project_root
git pull --no-ff --no-edit
git add *
git commit -m "updating code"
git push

echo "pushing code to github..."
