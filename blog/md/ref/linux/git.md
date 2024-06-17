# Git Ref

## Cheatsheet

**SETUP**

Configuring user information used across all local repositories

`git config --global user.name “[firstname lastname]”`

set a name that is identifiable for credit when review version history

`git config --global user.email “[valid-email]”`

set an email address that will be associated with each history marker

`git config --global color.ui auto`

set automatic command line coloring for Git for easy reviewing


**SETUP & INIT**

Configuring user information, initializing and cloning repositories

`git init`

initialize an existing directory as a Git repository

`git clone [url]`

retrieve an entire repository from a hosted location via URL


**STAGE & SNAPSHOT**

Working with snapshots and the Git staging area

`git status`

show modified files in working directory, staged for your next commit

`git add [file]`

add a file as it looks now to your next commit (stage)

`git reset [file]`

unstage a file while retaining the changes in working directory

`git diff`

diff of what is changed but not staged

`git diff --staged`

diff of what is staged but not yet commited

`git commit -m “[descriptive message]”`

commit your staged content as a new commit snapshot


**BRANCH & MERGE**

Isolating work in branches, changing context, and integrating changes

`git branch`

list your branches. a * will appear next to the currently active branch

`git branch [branch-name]`

create a new branch at the current commit

`git checkout`

switch to another branch and check it out into your working directory

`git merge [branch]`

merge the specified branch’s history into the current one

`git log`

show all commits in the current branch’s history

**TEMPORARY COMMITS**

Temporarily store modified, tracked files in order to change branches

`git stash`

Save modified and staged changes

`git stash list`

list stack-order of stashed file changes

`git stash pop`

write working from top of stash stack

`git stash drop`

discard the changes from top of stash stack

**SHARE & UPDATE**

Retrieving updates from another repository and updating local repos

`git remote add [alias] [url]`

add a git URL as an alias

`git fetch [alias]`

fetch down all the branches from that Git remote

`git merge [alias]/[branch]`

merge a remote branch into your current branch to bring it up to date

`git push [alias] [branch]`

Transmit local branch commits to the remote repository branch

`git pull`

fetch and merge any commits from the tracking rem

**TRACKING PATH CHANGES**

Versioning file removes and path changes

`git rm [file]`

delete the file from project and stage the removal for commit

`git mv [existing-path] [new-path]`

change an existing file path and stage the move

`git log --stat -M`

show all commit logs with indication of any paths that moved


**INSPECT & COMPARE**

Examining logs, diffs and object information

`git log`

show the commit history for the currently active branch

`git log branchB..branchA`

show the commits on branchA that are not on branchB

`git log --follow [file]`

show the commits that changed file, even across renames

`git diff branchB...branchA`

show the diff of what is in branchA that is not in branchB

`git show [SHA]`

show any object in Git in human-readable format

**IGNORING PATTERNS**

Preventing unintentional staging or commiting of files

```
logs/
*.notes
pattern*/
```

matches or wildcard globs.

Save a file with desired paterns as .gitignore with either direct string 

`git config --global core.excludesfile [file]`

system wide ignore patern for all local repositories

**REWRITE HISTORY** 

Rewriting branches, updating commits and clearing history

`git rebase [branch]`

apply any commits of current branch ahead of specified one

`git reset --hard [commit]`

clear staging area, rewrite working tree from specified commit

`git revert [commit]`

Create a new commit that undoes the changes made in the specified commit.

`git rebase -i [commit]`

Interactively rebase the current branch onto [commit].

`git reset --soft [commit]`

Move the HEAD pointer to [commit], staging all changes.

`git reset --hard [commit]`

Move the HEAD pointer to [commit], discarding all changes in the working directory.

`git commit --amend -m "[new message]"`

Change the commit message of the most recent commit.

**Optimizing and managing commit history**

`git merge --ff-only [branch]`

Perform a fast-forward merge, failing if not possible.

`git rebase -i [commit]`

Squash commits into one.

`git clone --depth 1 [url]`

Perform a shallow clone with a history truncated to the latest commit.


**Authentication and configuration with tokens**

`git config --global credential.helper store`

Save your credentials to a file.

`git config --global user.name "your-username"`

Set your GitHub username.

`git remote set-url origin https://<token>@github.com/username/repo.git`

Set the remote URL with the access token for authentication.

## Github Access 

* Use token to access a project

If you want to join someone's project, that project owner has to add your github id as collaborator. Then in the project setting, the project owner will generate a token. For example, a `classic token` and send it to the collaborator. The collaborator will use the above way to set password. The id is your github username, and password is the token.

`git config --global credential.helper store`

Set your credentials to be stored.

`git remote set-url origin https://<token>@github.com/username/repo.git`

Use the token as the password when prompted.

* Use SSH

Generate a public key and save it to the github ssh to allow the server to know your machine information. 

`ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`

Generate a new SSH key.

`ssh-add ~/.ssh/id_rsa`

Add your SSH key to the SSH agent.

`cat ~/.ssh/id_rsa.pub`

Copy your SSH public key and add it to your GitHub account.

## Trigger github.io to rebuild

`git commit -m 'rebuild pages' --allow-empty`
`git push origin <branch_name>`

## Ref

* <https://education.github.com/git-cheat-sheet-education.pdf>
