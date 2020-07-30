#!/usr/bin/env bash

function strip_file_extension() {
  #1: file path
  # assumes there is a "/" in argument
  local file_path
  local file_name
  file_path="${1}"
  file_name="${file_path##*/}"
  # strip out file extension
  echo "${file_name%.*}"
}

function cp_to_bin() {
  #1: path to source file
  #2: path to target bin
  local source_path
  local path_to_bin
  local target_path
  source_path="${1}"
  path_to_bin="${2}"
  target_path="${path_to_bin}/$(strip_file_extension "${source_path}")"
  sudo cp "${source_path}" "${target_path}"
  sudo chmod +x "${target_path}"
}

function main() {
  #1: (optional) full path to bin directory. Default: /usr/local/bin
  local path_to_bin="/usr/local/bin"
  cp_to_bin dev_tools/doctest-on.sh "${path_to_bin}"

  # test install
  command -v doctest-on
}

main "$@" || exit 1