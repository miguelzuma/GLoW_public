git fetch && git merge -X theirs origin/main --no-commit --no-ff
commit_id=$(git rev-parse origin/main)

jupyter nbconvert --clear-output --inplace ./notebooks/*.ipynb
make clean && make all

rsync -ahAX \
      --info=progress2 \
      ./sphinx_doc/build/html/ ./docs

git add *
git commit -m "doc updated for commit $commit_id"
git push