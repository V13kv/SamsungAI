#!/bin/bash

outputFolder=$1
#fileCnt=$2

mkdir -p "$outputFolder"

for dir in ./ishiharaPlates/*
do
	dirName=${dir##*/}
	mkdir -p "$outputFolder/$dirName"

	find "$dir" | #sort -R | #tail -$fileCnt |
		while read -r file; do
			cp "$dir/$file" "$outputFolder/$dirName"			
		done
done
