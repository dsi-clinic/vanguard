
# general
mkfile_path := $(abspath $(firstword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
current_abs_path := $(subst Makefile,,$(mkfile_path))

# pipeline constants
# PROJECT_NAME
project_name := "vanguard"
project_dir := "$(current_abs_path)"

# environment variables
include .env



open-ssh-login-node:
	code --remote ssh-remote+fe.ds ~/vanguard

